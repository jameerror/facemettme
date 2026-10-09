"""Face Swap Me TT Me: local and Linux cloud browser application."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import secrets
import shutil
import threading
import uuid
import webbrowser

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
import cv2

from engine import FaceEngine, Cancelled, MODEL_ROOT, SWAP_MODEL, read_image, process_video
from settings import DATA_ROOT, REFERENCE_MAX_MB, load_settings
from video_editor import VideoEditor

ROOT = Path(__file__).resolve().parent
DATA = DATA_ROOT
DATA.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['EXECUTION_THREAD_COUNT'] = 4
settings = load_settings()


def configure(server_settings):
    app.config['SERVER_SETTINGS'] = server_settings
    app.config['MAX_CONTENT_LENGTH'] = server_settings.max_request_bytes
    app.config['TRUSTED_HOSTS'] = list(server_settings.trusted_hosts)


configure(settings)
TOKEN = secrets.token_urlsafe(32)
jobs = {}
lock = threading.RLock()
executor = ThreadPoolExecutor(max_workers=1)
engine = FaceEngine()
inference_lock = threading.Lock()
editor = VideoEditor(lambda: engine, inference_lock,
                     lambda: any(job['state'] in {'queued', 'running'} for job in jobs.values()), lambda: DATA)
app.register_blueprint(editor.blueprint(lambda: app.config['SERVER_SETTINGS']))
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


@app.before_request
def access_control():
    # Health only reports process liveness, never files, tokens or model details.
    if request.path == '/healthz' and request.method == 'GET':
        return None
    if request.method in {'POST', 'DELETE'} and not secrets.compare_digest(request.headers.get('X-App-Token', ''), TOKEN):
        abort(403)


@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


@app.get('/')
def index():
    return render_template('index.html', token=TOKEN, cloud=app.config['SERVER_SETTINGS'].cloud,
                           execution_thread_count=app.config['EXECUTION_THREAD_COUNT'])


@app.get('/healthz')
def health():
    return jsonify(status='ok')


@app.get('/api/status')
def status():
    import onnxruntime as ort
    return jsonify(model_ready=SWAP_MODEL.is_file() and (MODEL_ROOT/'models'/'buffalo_l').is_dir(),
                   providers=ort.get_available_providers(), max_upload_mb=app.config['SERVER_SETTINGS'].max_upload_mb,
                   max_reference_mb=REFERENCE_MAX_MB)


def public(job):
    return {key: value for key, value in job.items() if key not in {'cancel', 'folder'}}


def update(job_id, **changes):
    with lock:
        jobs[job_id].update(changes)


def worker(job_id, source, target, mode, size, selection, kind, identities=None, video_id=None, thread_count=4):
    with inference_lock:
        return process_job(job_id, source, target, mode, size, selection, kind, identities, video_id, thread_count)


def process_job(job_id, source, target, mode, size, selection, kind, identities=None, video_id=None, thread_count=4):
    with lock:
        job = jobs[job_id]
        cancel, folder = job['cancel'], job['folder']
    output = folder / ('result.png' if kind == 'image' else 'result.mp4')

    def progress(percent, message):
        update(job_id, progress=round(percent, 1), message=message)

    try:
        if cancel.is_set():
            raise Cancelled()
        update(job_id, state='running')
        backend = engine.load(mode, size, progress)
        update(job_id, backend=backend)
        if cancel.is_set():
            raise Cancelled()
        face = engine.source(source)
        if kind == 'image':
            progress(20, 'กำลังสลับใบหน้า')
            result, count, _ = engine.swap(read_image(target), face, selection)
            if not count:
                raise ValueError('ไม่พบใบหน้าในภาพเป้าหมาย')
            if not cv2.imwrite(str(output), result):
                raise ValueError('บันทึกภาพไม่สำเร็จ')
            details = {'faces': count}
        else:
            details = process_video(engine, face, target, output, selection, progress, cancel, identities, thread_count)
        if cancel.is_set():
            raise Cancelled()
        update(job_id, state='done', progress=100, message='เสร็จแล้ว', details=details,
               result=f'/api/jobs/{job_id}/result', kind=kind)
    except Cancelled:
        output.unlink(missing_ok=True)
        update(job_id, state='cancelled', message='ยกเลิกแล้ว')
    except Exception as exc:
        logging.exception('Job %s failed', job_id)
        output.unlink(missing_ok=True)
        update(job_id, state='error', message=str(exc))
    finally:
        source.unlink(missing_ok=True)
        if video_id:
            editor.release(video_id, consume=True)
        else:
            target.unlink(missing_ok=True)


@app.post('/api/jobs')
def create_job():
    source, target = request.files.get('source'), request.files.get('target')
    video_id = request.form.get('video_id')
    if not source or not source.filename or (not video_id and (not target or not target.filename)):
        return jsonify(error='กรุณาเลือกรูปอ้างอิงและไฟล์เป้าหมาย'), 400
    source_ext = Path(source.filename).suffix.lower()
    target_ext = '.mp4' if video_id else Path(target.filename).suffix.lower()
    if source_ext not in IMAGE_EXT or target_ext not in IMAGE_EXT | VIDEO_EXT:
        return jsonify(error='นามสกุลไฟล์ไม่รองรับ'), 400
    mode, selection = request.form.get('mode', 'auto'), request.form.get('selection', 'largest')
    try:
        size = int(request.form.get('size', '640'))
        thread_count = int(request.form.get('execution_thread_count', app.config['EXECUTION_THREAD_COUNT']))
    except ValueError:
        return jsonify(error='ขนาดตรวจจับหรือจำนวนเธรดไม่ถูกต้อง'), 400
    if not 1 <= thread_count <= 32:
        return jsonify(error='จำนวนเธรดต้องเป็นจำนวนเต็ม 1–32'), 400
    if mode not in {'auto', 'cpu', 'cuda'} or selection not in {'largest', 'all', 'selected'} or size not in {320, 640, 1024}:
        return jsonify(error='ตัวเลือกไม่ถูกต้อง'), 400
    if selection == 'selected' and not video_id:
        return jsonify(error='กรุณาอัปโหลดวิดีโอแล้วเลือกใบหน้าจากเฟรมตัวอย่าง'), 400
    identities = None
    with lock:
        if any(job['state'] in {'queued', 'running'} for job in jobs.values()):
            return jsonify(error='มีงานกำลังทำอยู่ กรุณารอหรือยกเลิกก่อน'), 409
        if video_id:
            try:
                target_path, identities = editor.claim(video_id, request.form.get('frame_id'),
                                                        request.form.get('face_ids'), selection)
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
        job_id = uuid.uuid4().hex
        folder = DATA / job_id
        try:
            folder.mkdir()
        except Exception:
            if video_id:
                editor.release(video_id)
            raise
        jobs[job_id] = dict(id=job_id, state='queued', progress=0, message='รอประมวลผล',
                            backend='', folder=folder, cancel=threading.Event())
    try:
        source_path = folder / ('source'+source_ext)
        if not video_id:
            target_path = folder / ('target'+target_ext)
        source.save(source_path, buffer_size=1024*1024)
        if not video_id:
            target.save(target_path, buffer_size=1024*1024)
        if source_path.stat().st_size > REFERENCE_MAX_MB * 1024**2:
            raise RequestEntityTooLarge(description=f'รูปอ้างอิงต้องไม่เกิน {REFERENCE_MAX_MB} MB')
        if target_path.stat().st_size > app.config['SERVER_SETTINGS'].max_upload_mb * 1024**2:
            raise RequestEntityTooLarge(description=f"ไฟล์เป้าหมายต้องไม่เกิน {app.config['SERVER_SETTINGS'].max_upload_mb} MB")
        if not source_path.stat().st_size or not target_path.stat().st_size:
            raise ValueError('ไฟล์ว่างเปล่า')
        executor.submit(worker, job_id, source_path, target_path, mode, size, selection,
                        'image' if target_ext in IMAGE_EXT else 'video', identities, video_id, thread_count)
    except Exception as exc:
        with lock:
            jobs.pop(job_id, None)
        shutil.rmtree(folder)
        if video_id:
            editor.release(video_id)
        if isinstance(exc, RequestEntityTooLarge):
            raise
        return jsonify(error=str(exc)), 400
    return jsonify(id=job_id), 202


@app.get('/api/jobs/<job_id>')
def get_job(job_id):
    with lock:
        if job_id not in jobs:
            abort(404)
        return jsonify(public(jobs[job_id]))


@app.post('/api/jobs/<job_id>/cancel')
def cancel_job(job_id):
    with lock:
        if job_id not in jobs:
            abort(404)
        jobs[job_id]['cancel'].set()
    return jsonify(ok=True)


@app.get('/api/jobs/<job_id>/result')
def result(job_id):
    with lock:
        job = jobs.get(job_id)
        if not job or job['state'] != 'done':
            abort(404)
        path = job['folder'] / ('result.png' if job['kind'] == 'image' else 'result.mp4')
    return send_file(path, as_attachment=request.args.get('download') == '1', download_name='face-swap-me-tt-me'+path.suffix)


@app.delete('/api/jobs/<job_id>')
def delete_job(job_id):
    with lock:
        job = jobs.get(job_id)
        if not job:
            abort(404)
        if job['state'] in {'queued', 'running'}:
            return jsonify(error='กรุณายกเลิกและรอให้งานหยุดก่อนลบ'), 409
        shutil.rmtree(job['folder'])
        del jobs[job_id]
    return jsonify(ok=True)


@app.errorhandler(RequestEntityTooLarge)
def too_large(exc):
    if exc.description != RequestEntityTooLarge.description:
        return jsonify(error=exc.description), 413
    return jsonify(error=f"ไฟล์เป้าหมายต้องไม่เกิน {app.config['SERVER_SETTINGS'].max_upload_mb} MB และรูปอ้างอิงต้องไม่เกิน {REFERENCE_MAX_MB} MB"), 413


if __name__ == '__main__':
    import argparse
    from waitress import serve
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Bind address for local or cloud access')
    parser.add_argument('--port', type=int)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--execution-thread-count', type=int, choices=range(1, 33), default=4,
                        metavar='1-32', help='Default parallel video frame workers (default: 4)')
    args = parser.parse_args()
    try:
        settings = load_settings(args.host, args.port)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    configure(settings)
    app.config['EXECUTION_THREAD_COUNT'] = args.execution_thread_count
    local_host = '[::1]' if settings.host == '::1' else '127.0.0.1'
    url = f'http://{local_host}:{settings.port}'
    print(f'Face Swap Me TT Me: {url}', flush=True)
    if not args.no_browser and not settings.cloud:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    serve(app, host=settings.host, port=settings.port, threads=6,
          max_request_body_size=app.config['MAX_CONTENT_LENGTH'], channel_timeout=300)
