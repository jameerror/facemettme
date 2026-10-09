"""Upload-once video sessions, frame inspection and selected-person previews."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

import cv2
import numpy as np
from flask import Blueprint, abort, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from engine import read_image
from settings import REFERENCE_MAX_MB

VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def options(values):
    mode = values.get('mode', 'auto')
    try:
        size = int(values.get('size', 640))
    except (ValueError, TypeError):
        raise ValueError('ขนาดตรวจจับไม่ถูกต้อง')
    if mode not in {'auto', 'cpu', 'cuda'} or size not in {320, 640, 1024}:
        raise ValueError('ตัวเลือกประมวลผลไม่ถูกต้อง')
    return mode, size


def face_indices(value, count):
    try:
        ids = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        raise ValueError('รายการใบหน้าไม่ถูกต้อง')
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 or i >= count for i in ids):
        raise ValueError('กรุณาเลือกใบหน้าที่ตรวจพบในเฟรมนี้อย่างน้อยหนึ่งคน')
    return list(dict.fromkeys(ids))


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('อ่านวิดีโอไม่ได้')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = count / fps if fps > 0 else 0
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('อ่านความยาววิดีโอไม่ได้ กรุณาแปลงเป็น MP4 แล้วลองใหม่')
        return dict(duration=duration, fps=fps, width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def extract_frame(path, seconds):
    import imageio_ffmpeg
    # A video's displayed frame spans until the next frame. Seeking after the last
    # frame's PTS can otherwise return no image despite being inside the duration.
    metadata = video_metadata(path)
    seek_seconds = max(0, math.floor(seconds*metadata['fps']+1e-6)/metadata['fps']-1e-6)
    try:
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-ss', str(seek_seconds),
                                 '-i', str(path), '-map', '0:v:0', '-frames:v', '1',
                                 '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True,
                                timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        raise ValueError('ดึงเฟรมใช้เวลานานเกินไป กรุณาเลือกเวลาอื่นหรือลองใหม่')
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR) if result.stdout else None
    if result.returncode or frame is None:
        raise ValueError('ดึงเฟรมนี้ไม่ได้ กรุณาเลือกเวลาอื่น')
    if frame.shape[0]*frame.shape[1] > 40_000_000:
        raise ValueError('เฟรมวิดีโอใหญ่เกิน 40 ล้านพิกเซล')
    return frame


class VideoEditor:
    def __init__(self, engine, inference_lock, is_processing, data_root):
        self.engine = engine
        self.inference_lock = inference_lock
        self.is_processing = is_processing
        self.data_root = data_root
        self.lock = threading.RLock()
        self.videos = {}

    def get(self, video_id):
        asset = self.videos.get(video_id)
        if asset is None:
            abort(404, description='ไม่พบวิดีโอ กรุณาอัปโหลดใหม่')
        return asset

    @contextmanager
    def operation(self, video_id):
        with self.lock:
            asset = self.get(video_id)
            if asset['busy'] or asset['claimed'] or self.is_processing() or not self.inference_lock.acquire(blocking=False):
                abort(409, description='มีงานกำลังประมวลผล กรุณารอให้เสร็จก่อน')
            asset['busy'] = True
        try:
            yield asset
        finally:
            with self.lock:
                asset['busy'] = False
            self.inference_lock.release()

    def identity(self, asset, frame_id, ids):
        frame = asset.get('frame')
        if not frame or frame['id'] != frame_id:
            raise ValueError('เฟรมตัวอย่างเปลี่ยนไป กรุณาดึงเฟรมและเลือกใบหน้าใหม่')
        selected = face_indices(ids, len(frame['faces']))
        return np.stack([frame['faces'][i].normed_embedding.copy() for i in selected])

    def claim(self, video_id, frame_id, ids, selection):
        with self.lock:
            asset = self.get(video_id)
            if asset['busy'] or asset['claimed']:
                abort(409, description='วิดีโอนี้กำลังถูกใช้งาน')
            identities = self.identity(asset, frame_id, ids) if selection == 'selected' else None
            asset['claimed'] = True
            return asset['path'], identities

    def release(self, video_id, consume=False):
        with self.lock:
            asset = self.videos.get(video_id)
            if not asset:
                return
            if consume:
                shutil.rmtree(asset['folder'], ignore_errors=True)
                self.videos.pop(video_id, None)
            else:
                asset['claimed'] = False

    def blueprint(self, settings):
        bp = Blueprint('video_editor', __name__)

        @bp.errorhandler(ValueError)
        def bad_input(error):
            return jsonify(error=str(error)), 400

        @bp.errorhandler(409)
        @bp.errorhandler(404)
        def request_error(error):
            return jsonify(error=error.description), error.code

        @bp.post('/api/videos')
        def upload():
            file = request.files.get('target')
            extension = Path(file.filename or '').suffix.lower() if file else ''
            if extension not in VIDEO_EXT:
                raise ValueError('กรุณาเลือกไฟล์วิดีโอที่รองรับ')
            video_id = uuid.uuid4().hex
            folder = self.data_root()/'editor'/video_id
            folder.mkdir(parents=True)
            path = folder/('video'+extension)
            try:
                file.save(path, buffer_size=1024*1024)
                if path.stat().st_size > settings().max_upload_mb*1024**2:
                    raise RequestEntityTooLarge(description=f'วิดีโอต้องไม่เกิน {settings().max_upload_mb} MB')
                metadata = video_metadata(path)
                with self.lock:
                    self.videos[video_id] = dict(folder=folder, path=path, metadata=metadata,
                                                 busy=False, claimed=False, frame=None)
                return jsonify(id=video_id, **metadata), 201
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise

        @bp.post('/api/videos/<video_id>/frame')
        def inspect_frame(video_id):
            values = request.get_json(silent=True) or {}
            mode, size = options(values)
            try:
                seconds = float(values.get('seconds', 0))
            except (ValueError, TypeError):
                raise ValueError('เวลาเฟรมไม่ถูกต้อง')
            with self.operation(video_id) as asset:
                if not math.isfinite(seconds) or not 0 <= seconds < asset['metadata']['duration']:
                    raise ValueError('เวลาเฟรมอยู่นอกช่วงวิดีโอ')
                image = extract_frame(asset['path'], seconds)
                engine = self.engine()
                backend = engine.load(mode, size, lambda *args: None)
                faces = engine.faces(image)
                frame_id = uuid.uuid4().hex
                folder = asset['folder']/frame_id
                folder.mkdir()
                assets = {'original.png': folder/'original.png', 'detected.jpg': folder/'detected.jpg'}
                marked = image.copy()
                thumbnails = []
                try:
                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = map(int, face.bbox)
                        margin = max(8, int((x2-x1)*.2))
                        crop = image[max(0,y1-margin):min(image.shape[0],y2+margin),
                                     max(0,x1-margin):min(image.shape[1],x2+margin)]
                        if not crop.size:
                            raise ValueError('ตำแหน่งใบหน้าไม่ถูกต้อง')
                        filename = f'face-{i}.jpg'
                        assets[filename] = folder/filename
                        if not cv2.imwrite(str(assets[filename]), crop):
                            raise ValueError('บันทึกภาพใบหน้าไม่ได้')
                        cv2.rectangle(marked, (x1,y1), (x2,y2), (154,245,209), 2)
                        cv2.putText(marked, str(i+1), (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .8, (154,245,209), 2)
                        thumbnails.append(dict(id=i, thumbnail=f'/api/videos/{video_id}/frames/{frame_id}/{filename}'))
                    if not cv2.imwrite(str(assets['original.png']), image) or not cv2.imwrite(str(assets['detected.jpg']), marked):
                        raise ValueError('บันทึกเฟรมไม่ได้')
                except Exception:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise
                old_frame = asset['frame']
                asset['frame'] = dict(id=frame_id, seconds=seconds, faces=faces, assets=assets, folder=folder)
                if old_frame:
                    shutil.rmtree(old_frame['folder'], ignore_errors=True)
                return jsonify(frame_id=frame_id, seconds=seconds, faces=thumbnails, backend=backend,
                               frame_time=math.floor(seconds*asset['metadata']['fps']+1e-6)/asset['metadata']['fps'],
                               original=f'/api/videos/{video_id}/frames/{frame_id}/detected.jpg')

        @bp.post('/api/videos/<video_id>/preview')
        def preview(video_id):
            mode, size = options(request.form)
            selection = request.form.get('selection', 'selected')
            if selection not in {'selected', 'all'}:
                raise ValueError('โหมดใบหน้าไม่ถูกต้อง')
            source = request.files.get('source')
            extension = Path(source.filename or '').suffix.lower() if source else ''
            if extension not in IMAGE_EXT:
                raise ValueError('กรุณาเลือกรูปอ้างอิง')
            with self.operation(video_id) as asset:
                frame = asset['frame']
                frame_id = request.form.get('frame_id')
                if not frame or frame['id'] != frame_id:
                    raise ValueError('กรุณาดึงเฟรมตัวอย่างใหม่ก่อน')
                identities = self.identity(asset, frame_id, request.form.get('face_ids')) if selection == 'selected' else None
                path = asset['folder']/('reference'+extension)
                try:
                    source.save(path, buffer_size=1024*1024)
                    if path.stat().st_size > REFERENCE_MAX_MB*1024**2:
                        raise RequestEntityTooLarge(description=f'รูปอ้างอิงต้องไม่เกิน {REFERENCE_MAX_MB} MB')
                    engine = self.engine()
                    backend = engine.load(mode, size, lambda *args: None)
                    reference = engine.source(path)
                    result, count, _ = engine.swap(read_image(frame['assets']['original.png']), reference, selection, identities)
                    if not count:
                        raise ValueError('ไม่พบใบหน้าที่เลือก กรุณาเลือกเฟรมที่เห็นหน้าชัดขึ้น')
                    filename = 'result.png'
                    output = frame['folder']/filename
                    if not cv2.imwrite(str(output), result):
                        raise ValueError('บันทึกตัวอย่างไม่ได้')
                    frame['assets'][filename] = output
                    return jsonify(result=f'/api/videos/{video_id}/frames/{frame_id}/{filename}', faces=count, backend=backend)
                finally:
                    path.unlink(missing_ok=True)

        @bp.get('/api/videos/<video_id>/frames/<frame_id>/<filename>')
        def image(video_id, frame_id, filename):
            with self.lock:
                frame = self.get(video_id)['frame']
                if not frame or frame['id'] != frame_id or filename not in frame['assets']:
                    abort(404)
                return send_file(frame['assets'][filename])

        @bp.delete('/api/videos/<video_id>')
        def remove(video_id):
            with self.lock:
                asset = self.get(video_id)
                if asset['busy'] or asset['claimed']:
                    abort(409, description='รอให้งานวิดีโอหยุดก่อนลบ')
                self.release(video_id, consume=True)
            return jsonify(ok=True)

        return bp
