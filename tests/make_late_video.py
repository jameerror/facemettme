"""Generate a small demo where one person is absent until the final frame."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from insightface.data import get_image
from engine import FaceEngine
from settings import DATA_ROOT

folder = DATA_ROOT/'smoke'
folder.mkdir(parents=True, exist_ok=True)
engine = FaceEngine()
engine.load('cpu', 640, lambda *args: None)
image = get_image('t1')
people = engine.faces(image)
person = people[1]
x1,y1,x2,y2 = map(int, person.bbox)
masked = image.copy()
masked[max(0,y1-20):y2+20, max(0,x1-20):x2+20] = 0
path = folder/'late-person.mp4'
height, width = image.shape[:2]
width += width % 2
height += height % 2
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 3, (width,height))
assert writer.isOpened()
for frame in [masked, masked, image]:
    padded = cv2.copyMakeBorder(frame, 0,height-frame.shape[0],0,width-frame.shape[1],cv2.BORDER_CONSTANT)
    writer.write(padded)
writer.release()
print('Late-person video:', path, flush=True)
