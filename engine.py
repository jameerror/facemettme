"""Local face swapping with explicit runtime selection and video audio preservation."""
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import math
import subprocess
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps
from settings import MODEL_ROOT

ROOT = Path(__file__).resolve().parent
SWAP_MODEL = MODEL_ROOT / 'inswapper_128.onnx'
DETECTION_ANGLES = (0, 90, 180, 270, 45, 135, 225, 315)
DETECTION_MIRRORS = (False, True)


class Cancelled(Exception):
    pass


def read_image(path):
    # Pillow applies camera EXIF rotation; OpenCV alone can misread orientation.
    with Image.open(path) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError('ภาพใหญ่เกิน 40 ล้านพิกเซล กรุณาย่อภาพก่อน')
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def providers_for(mode, available):
    if mode not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('โหมดประมวลผลไม่ถูกต้อง')
    cuda = 'CUDAExecutionProvider' in available
    if mode == 'cuda' and not cuda:
        raise ValueError('ไม่พบ NVIDIA CUDA runtime ให้เลือก CPU หรือติดตั้ง GPU runtime')
    if mode != 'cpu' and cuda:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def face_tiles(width, height):
    """Overlapping, bounded search windows, including the image edges."""
    longest = max(width, height)
    if longest <= 256:
        return []
    side = max(256, math.ceil(longest / 4), min(640, math.ceil(longest / 2)))

    def starts(length):
        end = max(0, length - side)
        return sorted(set([*range(0, end, max(1, int(side * .75))), end]))

    return [(x, y, min(x + side, width), min(y + side, height))
            for y in starts(height) for x in starts(width)]


def same_face_box(a, b):
    """Suppress both duplicate boxes and partial boxes at crop boundaries."""
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return (intersection / max(area_a + area_b - intersection, 1) >= .35 or
            intersection / max(min(area_a, area_b), 1) >= .7)


def rotated_view(image, angle):
    """Rotate without cutting corners; return the map back to source pixels."""
    if angle == 0:
        return image, np.eye(2, 3, dtype=np.float32)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), angle, 1.)
    cosine, sine = abs(matrix[0,0]), abs(matrix[0,1])
    out_width = int(math.ceil(width*cosine + height*sine - 1e-6))
    out_height = int(math.ceil(height*cosine + width*sine - 1e-6))
    matrix[:,2] += [(out_width-width)/2, (out_height-height)/2]
    rotated = cv2.warpAffine(image, matrix, (out_width,out_height),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
    return rotated, cv2.invertAffineTransform(matrix)


def detection_views(image):
    for angle in DETECTION_ANGLES:
        rotated, inverse = rotated_view(image, angle)
        for mirrored in DETECTION_MIRRORS:
            if not mirrored:
                yield angle, False, rotated, inverse
                continue
            # Compose the reflection with the rotation's inverse. Mirroring
            # also reverses eye/mouth labels, which must be restored separately.
            reflection = np.array([[-1,0,rotated.shape[1]-1],[0,1,0],[0,0,1]],float)
            mapping = np.vstack([inverse,[0,0,1]]) @ reflection
            yield angle, True, cv2.flip(rotated,1), mapping[:2]


class FaceEngine:
    def __init__(self):
        self.analyzer = self.swapper = None
        self.key = None
        self.backend = 'ยังไม่ได้โหลด'

    def load(self, mode, size, progress):
        import onnxruntime as ort
        if mode != 'cpu' and hasattr(ort, 'preload_dlls'):
            ort.preload_dlls(directory='')
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        if not SWAP_MODEL.is_file():
            raise ValueError('ไม่พบ models/inswapper_128.onnx กรุณารัน setup_models.py ก่อน')
        if not (MODEL_ROOT / 'models' / 'buffalo_l').is_dir():
            raise ValueError('ไม่พบโมเดล buffalo_l กรุณารัน setup_models.py ก่อน')
        providers = providers_for(mode, ort.get_available_providers())
        key = (tuple(providers), size)
        if key == self.key:
            return self.backend
        progress(1, 'กำลังโหลดโมเดล')

        def create(selected):
            analyzer = FaceAnalysis(name='buffalo_l', root=str(MODEL_ROOT),
                                    allowed_modules=['detection', 'recognition'], providers=selected)
            analyzer.prepare(ctx_id=0 if selected[0].startswith('CUDA') else -1,
                             det_size=(size, size))
            swapper = get_model(str(SWAP_MODEL), providers=selected, download=False)
            if swapper is None:
                raise ValueError('ไฟล์โมเดลสลับใบหน้าไม่ถูกต้อง')
            sessions = [swapper.session] + [model.session for model in analyzer.models.values()]
            using_cuda = all('CUDAExecutionProvider' in session.get_providers() for session in sessions)
            return analyzer, swapper, using_cuda

        try:
            analyzer, swapper, cuda = create(providers)
        except Exception:
            if mode != 'auto' or providers[0] == 'CPUExecutionProvider':
                raise
            analyzer, swapper, cuda = create(['CPUExecutionProvider'])
        if mode == 'cuda' and not cuda:
            raise ValueError('CUDA โหลดไม่สำเร็จ ตรวจไดรเวอร์ CUDA/cuDNN หรือเลือก CPU')
        if mode == 'auto' and providers[0].startswith('CUDA') and not cuda:
            analyzer, swapper, cuda = create(['CPUExecutionProvider'])
        self.analyzer, self.swapper = analyzer, swapper
        self.key = key
        self.backend = 'NVIDIA GPU (CUDA)' if cuda else 'CPU'
        return self.backend

    def faces(self, image):
        from insightface.app.common import Face
        original = list(self.analyzer.get(image))
        original_ids = {id(face) for face in original}
        faces = []
        candidates = list(original)
        height, width = image.shape[:2]
        # Run only detection on tiles; compute recognition once per new face
        # after merging so overlapping tiles do not repeat expensive embeddings.
        regions = [(0,0,width,height), *face_tiles(width, height)]
        for region_index, (left, top, right, bottom) in enumerate(regions):
            crop = image[top:bottom,left:right]
            for angle, mirrored, view, inverse in detection_views(crop):
                if region_index == 0 and angle == 0 and not mirrored:
                    continue  # Already detected by analyzer.get above.
                boxes, landmarks = self.analyzer.det_model.detect(view, max_num=0)
                if landmarks is None:
                    continue
                for box, points in zip(boxes, landmarks):
                    x1,y1,x2,y2 = box[:4]
                    corners = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]],np.float32)
                    mapped = cv2.transform(corners,inverse)[0]
                    bbox = np.r_[mapped.min(axis=0),mapped.max(axis=0)]
                    bbox = np.clip(bbox,[0,0,0,0],[right-left,bottom-top]*2)
                    bbox += [left,top,left,top]
                    points = np.asarray(points,np.float32)
                    if mirrored:
                        points = points[[1,0,2,4,3]]
                    kps = cv2.transform(points[None],inverse)[0] + [left,top]
                    if not np.isfinite(bbox).all() or not np.isfinite(kps).all():
                        continue
                    if np.any(bbox[2:] - bbox[:2] < 6):
                        continue
                    candidates.append(Face(bbox=bbox.astype(np.float32),
                                           kps=kps.astype(np.float32), det_score=float(box[4])))
        # An upside-down face can be weakly detected with incorrect landmark
        # ordering. Prefer the strongest orientation, even over the first pass.
        for face in sorted(candidates, key=lambda face: getattr(face,'det_score',1.), reverse=True):
            if any(same_face_box(face.bbox, other.bbox) for other in faces):
                continue
            if id(face) not in original_ids:
                self.analyzer.models['recognition'].get(image, face)
            faces.append(face)
        return sorted(faces, key=lambda face: float(face.bbox[0]))

    def recover_faces(self, image, faces, boxes):
        """Retry missed tracks at crop resolution without changing shared models."""
        from copy import copy
        from face_tracker import overlap
        if self.analyzer is None:
            return faces
        recovered = list(faces)
        height, width = image.shape[:2]
        for box in boxes:
            if any(overlap(box, face.bbox) >= .25 for face in recovered):
                continue
            center = (box[:2] + box[2:]) / 2
            # Include context and room for motion; the model resizes this crop
            # to its normal input size, making small faces easier to detect.
            radius = np.maximum(box[2:] - box[:2], 24)
            left, top = np.maximum(np.floor(center - radius), 0).astype(int)
            right, bottom = np.minimum(np.ceil(center + radius), [width, height]).astype(int)
            if right - left < 24 or bottom - top < 24:
                continue
            for candidate in self.analyzer.get(image[top:bottom, left:right]):
                # InsightFace's Face returns None for missing magic methods,
                # so Python's copy protocol cannot reconstruct it reliably.
                face = type(candidate)(candidate) if isinstance(candidate, dict) else copy(candidate)
                face.bbox = np.asarray(face.bbox).copy() + np.array([left, top, left, top])
                face.kps = np.asarray(face.kps).copy() + np.array([left, top])
                if overlap(box, face.bbox) < .2:
                    continue
                if any(overlap(face.bbox, other.bbox) >= .25 for other in recovered):
                    continue
                recovered.append(face)
        return sorted(recovered, key=lambda face: float(face.bbox[0]))

    def source(self, path):
        faces = self.faces(read_image(path))
        if len(faces) != 1:
            raise ValueError(f'รูปอ้างอิงต้องมีใบหน้าเดียว ตรวจพบ {len(faces)} ใบหน้า')
        return faces[0]

    def swap(self, image, source, selection, identity=None):
        faces = self.faces(image)
        if selection == 'all':
            targets = faces
        elif selection == 'selected':
            if identity is None:
                raise ValueError('กรุณาเลือกใบหน้าจากเฟรมตัวอย่างก่อน')
            identities = np.atleast_2d(identity)
            pairs = sorted(((float(np.dot(face.normed_embedding, embedding)), face_index, person_index)
                            for face_index, face in enumerate(faces)
                            for person_index, embedding in enumerate(identities)), reverse=True)
            used_faces, used_people, targets = set(), set(), []
            for score, face_index, person_index in pairs:
                if score < 0.35:
                    break
                if face_index not in used_faces and person_index not in used_people:
                    targets.append(faces[face_index])
                    used_faces.add(face_index)
                    used_people.add(person_index)
        elif identity is None:
            targets = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[:1]
            if targets:
                identity = targets[0].normed_embedding.copy()
        else:
            # Match the original person on every frame, so face order changes do not switch identity.
            candidates = [(float(np.dot(f.normed_embedding, identity)), f) for f in faces]
            best = max(candidates, key=lambda pair: pair[0]) if candidates else None
            targets = [best[1]] if best and best[0] >= 0.35 else []
        result, count = self.swap_targets(image, source, targets)
        return result, count, identity

    def swap_targets(self, image, source, targets):
        result = image.copy()
        for target in targets:
            result = self.swapper.get(result, target, source, paste_back=True)
        return result, len(targets)


def tracked_frames(engine, source, capture, selection, cancel, identities, thread_count):
    from face_tracker import FaceTracker
    tracker = FaceTracker(selection, identities)
    pool = ThreadPoolExecutor(max_workers=thread_count, thread_name_prefix='swap-frame')

    def detect(frame):
        if cancel.is_set():
            raise Cancelled()
        return engine.faces(frame)

    def render(frame, targets):
        if cancel.is_set():
            raise Cancelled()
        result = engine.swap_targets(frame, source, targets)
        if cancel.is_set():
            raise Cancelled()
        return result

    def result(future):
        while not wait([future], timeout=.1).done:
            if cancel.is_set():
                raise Cancelled()
        if cancel.is_set():
            raise Cancelled()
        return future.result()

    try:
        while True:
            batch = []
            for _ in range(thread_count):
                if cancel.is_set():
                    raise Cancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                batch.append((frame, pool.submit(detect, frame)))
            if not batch:
                break
            rendered = []
            # Identity and optical flow are updated only in timeline order,
            # even if inference workers finish their observations out of order.
            for frame, observation in batch:
                targets = tracker.select(frame, result(observation), engine.recover_faces)
                rendered.append(pool.submit(render, frame, targets))
            for future in rendered:
                yield result(future)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def processed_frames(engine, source, capture, selection, cancel, identities, thread_count):
    """Keep a bounded window of frames in flight; yield strictly in input order."""
    if isinstance(engine, FaceEngine):
        yield from tracked_frames(engine, source, capture, selection, cancel, identities, thread_count)
        return
    identity = identities
    pending = deque()
    pool = ThreadPoolExecutor(max_workers=thread_count, thread_name_prefix='swap-frame')

    def swap(frame, selected):
        if cancel.is_set():
            raise Cancelled()
        result = engine.swap(frame, source, selection, selected)
        if cancel.is_set():
            raise Cancelled()
        return result

    exhausted = False
    try:
        while not exhausted or pending:
            if cancel.is_set():
                raise Cancelled()
            while not exhausted and len(pending) < thread_count:
                if cancel.is_set():
                    raise Cancelled()
                ok, frame = capture.read()
                if not ok:
                    exhausted = True
                    break
                # Establish the legacy principal identity in timeline order.
                if selection == 'largest' and identity is None:
                    result, count, identity = swap(frame, None)
                    yield result, count
                else:
                    pending.append(pool.submit(swap, frame, identity))
            if pending:
                future = pending.popleft()
                while not wait([future], timeout=.1).done:
                    if cancel.is_set():
                        raise Cancelled()
                result, count, _ = future.result()
                yield result, count
    finally:
        # Wait for active inference before the shared model/files can be reused.
        pool.shutdown(wait=True, cancel_futures=True)


def process_video(engine, source, target, output, selection, progress, cancel, identities=None, thread_count=4):
    import imageio_ffmpeg
    if type(thread_count) is not int or not 1 <= thread_count <= 32:
        raise ValueError('จำนวนเธรดต้องเป็นจำนวนเต็ม 1–32')
    if cancel.is_set():
        raise Cancelled()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    normalized = output.parent / 'normalized.mp4'
    silent = output.parent / 'silent.mp4'
    process = None

    def run(command):
        # File-backed stderr prevents pipe deadlocks on long clips; polling enables cancellation.
        nonlocal process
        if cancel.is_set():
            raise Cancelled()
        with (output.parent / 'ffmpeg.log').open('w+b') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while process.poll() is None:
                if cancel.wait(0.2):
                    process.terminate()
                    process.wait()
                    raise Cancelled()
            if process.returncode:
                log.seek(0)
                detail = log.read().decode('utf-8', 'replace')[-1200:]
                raise ValueError('แปลงวิดีโอไม่สำเร็จ: ' + detail)

    capture = writer = None
    try:
        progress(3, 'กำลังเตรียมวิดีโอและปรับอัตราเฟรม')
        # Normalize variable frame rate/rotation first; preserve the original audio timeline.
        run([ffmpeg, '-y', '-i', str(target), '-map', '0:v:0', '-an', '-vf',
             'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-fps_mode', 'cfr', '-c:v', 'libx264',
             '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(normalized)])
        capture = cv2.VideoCapture(str(normalized))
        if not capture.isOpened():
            raise ValueError('อ่านวิดีโอไม่ได้')
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError('อัตราเฟรมหรือขนาดวิดีโอไม่ถูกต้อง')
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('สร้างไฟล์วิดีโอไม่ได้')
        frame_count = swapped_frames = 0
        frames = processed_frames(engine, source, capture, selection, cancel, identities, thread_count)
        try:
            for result, count in frames:
                if cancel.is_set():
                    raise Cancelled()
                writer.write(result)
                frame_count += 1
                swapped_frames += int(count > 0)
                progress(min(94, 5 + 89 * frame_count / max(total, frame_count)),
                         f'ประมวลผลเฟรม {frame_count:,} / {total:,} · {thread_count} เธรด')
        finally:
            frames.close()
        capture.release()
        writer.release()
        capture = writer = None
        if not frame_count or not swapped_frames:
            raise ValueError('ไม่พบใบหน้าเป้าหมายในวิดีโอ')
        if total and frame_count < total - 1:
            raise ValueError('วิดีโออ่านได้ไม่ครบเฟรม กรุณาแปลงไฟล์แล้วลองใหม่')
        progress(95, 'กำลังบันทึก MP4 พร้อมเสียงต้นฉบับ')
        run([ffmpeg, '-y', '-i', str(silent), '-i', str(target), '-map', '0:v:0',
             '-map', '1:a:0?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-t', str(frame_count/fps),
             '-movflags', '+faststart', str(output)])
        return {'frames': frame_count, 'swapped_frames': swapped_frames, 'fps': round(fps, 3),
                'thread_count': thread_count}
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        normalized.unlink(missing_ok=True)
        silent.unlink(missing_ok=True)
