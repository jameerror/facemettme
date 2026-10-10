"""Real-model check: recover one intentionally missed face in a video frame."""
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from insightface.data import get_image
from engine import FaceEngine, process_video
from settings import DATA_ROOT

folder=DATA_ROOT/'smoke'
folder.mkdir(parents=True,exist_ok=True)
engine=FaceEngine()
print('Tracking backend:',engine.load('cpu',640,lambda *args:None),flush=True)
image=get_image('t1')
people=engine.faces(image)
selected=people[1].normed_embedding.copy()
source=engine.source(folder/'reference.png')
height,width=image.shape[:2]
target=folder/'tracking-target.mp4'
writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),3,(width,height))
assert writer.isOpened()
for marker in [0,100,200]:
    frame=image.copy()
    frame[-24:,:24]=marker
    writer.write(frame)
writer.release()
detect=engine.faces
misses=[]

def detect_with_one_miss(frame):
    faces=detect(frame)
    if 70 < frame[-12:,:12].mean() < 130:
        filtered=[face for face in faces if np.dot(face.normed_embedding,selected)<.7]
        assert len(filtered)==len(faces)-1
        misses.append(True)
        return filtered
    return faces

engine.faces=detect_with_one_miss
render=engine.swap_targets
for selection,expected in [('selected',1),('all',len(people))]:
    misses.clear()
    counts=[]
    def render_and_count(frame,source,targets):
        counts.append(len(targets))
        return render(frame,source,targets)
    engine.swap_targets=render_and_count
    details=process_video(engine,source,target,folder/f'tracking-{selection}-result.mp4',selection,
                          lambda *args:None,threading.Event(),np.array([selected]) if selection=='selected' else None,4)
    assert misses==[True]
    assert details['frames']==details['swapped_frames']==3,details
    assert counts==[expected]*3,counts
    print('REAL TRACKING PASSED:',selection,'one detector miss recovered; all target faces swapped in 3/3 frames with 4 workers',flush=True)
