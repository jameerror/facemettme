"""Compare whole-frame and tiled detection on small model-author demo faces."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import FaceEngine, same_face_box
from settings import DATA_ROOT

engine = FaceEngine()
print('Backend:',engine.load('cpu',640,lambda *args:None),flush=True)
reference = cv2.imread(str(DATA_ROOT/'smoke'/'reference.png'))
assert reference is not None,'Run tests/smoke_models.py first'
canvas = np.full((1080,1920,3),180,np.uint8)
patch = cv2.resize(reference,(56,64),interpolation=cv2.INTER_AREA)
expected = []
for left,top in [(180,160),(920,500),(1770,930)]:
    canvas[top:top+64,left:left+56] = patch
    expected.append(np.array([left,top,left+56,top+64]))
baseline = engine.analyzer.get(canvas)
faces = engine.faces(canvas)
print('Whole-frame detections:',len(baseline),'Tiled detections:',len(faces),flush=True)
assert len(faces)>len(baseline),(len(baseline),len(faces))
assert all(any(same_face_box(box,face.bbox) for face in faces) for box in expected)
assert all(face.normed_embedding is not None for face in faces)
print('REAL SMALL-FACE CHECK PASSED: all three distant faces detected with embeddings',flush=True)
