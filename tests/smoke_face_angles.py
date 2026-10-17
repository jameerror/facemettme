"""Real-model roll-angle detection and alignment, using the bundled demo crop."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import FaceEngine, DETECTION_ANGLES
from settings import DATA_ROOT

engine = FaceEngine()
print('Backend:',engine.load('cpu',640,lambda *args:None),flush=True)
reference = cv2.imread(str(DATA_ROOT/'smoke'/'reference.png'))
assert reference is not None,'Run tests/smoke_models.py first'
canvas = np.full((256,256,3),127,np.uint8)
scale = 144/max(reference.shape[:2])
reference = cv2.resize(reference,None,fx=scale,fy=scale)
h,w = reference.shape[:2]
canvas[(256-h)//2:(256-h)//2+h,(256-w)//2:(256-w)//2+w] = reference
source = engine.analyzer.get(canvas)[0]
baseline_hits = baseline_matches = improved_hits = 0
for angle in DETECTION_ANGLES:
    matrix = cv2.getRotationMatrix2D((127.5,127.5),angle,1.)
    frame = cv2.warpAffine(canvas,matrix,(256,256),borderValue=(127,127,127))
    baseline = engine.analyzer.get(frame)
    faces = engine.faces(frame)
    baseline_hits += bool(baseline)
    baseline_matches += any(float(np.dot(face.normed_embedding,source.normed_embedding))>=.35
                            for face in baseline)
    assert len(faces)==1,(angle,len(faces))
    target = faces[0]
    score = float(np.dot(target.normed_embedding,source.normed_embedding))
    assert score>=.35,(angle,score)
    improved_hits += 1
    if angle in (90,180):
        result,count = engine.swap_targets(frame,source,faces)
        assert count==1 and np.any(result!=frame)
    print('PASS angle:',angle,'identity score:',round(score,3),flush=True)
assert improved_hits==8
print('REAL ANGLE CHECK PASSED: baseline',baseline_hits,'/8; multi-angle',improved_hits,'/8',flush=True)
print('Correct identity: baseline',baseline_matches,'/8; multi-angle',improved_hits,'/8',flush=True)
