"""Verify real recognition/alignment when only reflected detections are used.

This checks reflection geometry, not real-world profile/pitch coverage.
"""
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import FaceEngine
from settings import DATA_ROOT

engine = FaceEngine()
engine.load('cpu',640,lambda *args:None)
image = cv2.imread(str(DATA_ROOT/'smoke'/'reference.png'))
assert image is not None,'Run tests/smoke_models.py first'
reference = engine.analyzer.get(image)[0]
with patch('engine.DETECTION_MIRRORS',(True,)), patch('engine.DETECTION_ANGLES',(0,)), \
        patch.object(engine.analyzer,'get',return_value=[]):
    faces = engine.faces(image)
assert len(faces)==1,len(faces)
similarity = float(np.dot(reference.normed_embedding,faces[0].normed_embedding))
assert similarity>=.35,similarity
assert np.linalg.norm(faces[0].kps-reference.kps,axis=1).mean()<10
print('REAL MIRROR-ONLY DETECTION PASSED: original coordinates and identity; similarity',
      round(similarity,3),flush=True)
