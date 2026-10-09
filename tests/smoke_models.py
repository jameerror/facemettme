"""Optional integration check using the model author's bundled demo image."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
import subprocess
import cv2
import imageio_ffmpeg
from insightface.data import get_image
from engine import FaceEngine, process_video
from settings import DATA_ROOT

folder = DATA_ROOT / 'smoke'
folder.mkdir(parents=True, exist_ok=True)
engine = FaceEngine()
print('Backend:', engine.load('cpu', 640, lambda *args: print(*args)), flush=True)
image = get_image('t1')
faces = engine.faces(image)
assert len(faces) >= 2, 'Demo should contain multiple faces'
x1, y1, x2, y2 = map(int, faces[0].bbox)
margin = 30
crop = image[max(0,y1-margin):y2+margin, max(0,x1-margin):x2+margin]
reference = folder/'reference.png'
cv2.imwrite(str(reference), crop)
source = engine.source(reference)
result, count, _ = engine.swap(image, source, 'all')
assert count >= 2
assert cv2.absdiff(image, result).sum() > 0
cv2.imwrite(str(folder/'result.png'), result)
frame = folder/'target.png'
cv2.imwrite(str(frame), image)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
clip = folder/'target.mp4'
subprocess.run([ffmpeg, '-y', '-loop', '1', '-i', str(frame), '-f', 'lavfi', '-i',
                'sine=frequency=440:duration=1', '-t', '1', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-r', '3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                str(clip)], check=True, capture_output=True)
details = process_video(engine, source, clip, folder/'result.mp4', 'all',
                        lambda *args: print(*args, flush=True), threading.Event())
assert details['swapped_frames'] == details['frames'] == 3
print('REAL MODEL IMAGE + VIDEO PASSED:', count, 'faces;', details, flush=True)
