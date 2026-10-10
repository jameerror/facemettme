import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from engine import FaceEngine, processed_frames
from face_tracker import FaceTracker


def person(score=1., x=30):
    return SimpleNamespace(bbox=np.array([x,30,x+40,80],np.float32),
                           kps=np.array([[x+10,42],[x+30,42],[x+20,55],[x+12,68],[x+28,68]],np.float32),
                           normed_embedding=np.array([score,np.sqrt(max(0,1-score*score))]))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120,140,3),100,np.uint8)

    def test_short_detection_gap_recovers_then_expires(self):
        for selection, identities in [('selected', [[1.,0.]]), ('all', None)]:
            tracker = FaceTracker(selection, identities)
            self.assertEqual(len(tracker.select(self.frame,[person()])),1)
            with patch.object(tracker,'predict',return_value=person()):
                for _ in range(3):
                    self.assertEqual(len(tracker.select(self.frame,[])),1)
                self.assertEqual(tracker.select(self.frame,[]),[])
                self.assertEqual(tracker.select(self.frame,[]),[])
            self.assertEqual(len(tracker.select(self.frame,[person()])),1)

    def test_score_dip_keeps_same_position_for_only_short_gap(self):
        tracker = FaceTracker('selected',[[1.,0.]])
        tracker.select(self.frame,[person()])
        with patch.object(tracker,'predict',return_value=None):
            for _ in range(3):
                self.assertEqual(len(tracker.select(self.frame,[person(.3)])),1)
            self.assertEqual(tracker.select(self.frame,[person(.3)]),[])

    def test_other_person_or_changed_position_is_not_recovered(self):
        for detected in [person(0.), person(.3,85)]:
            tracker = FaceTracker('selected',[[1.,0.]])
            tracker.select(self.frame,[person()])
            with patch.object(tracker,'predict',return_value=person()):
                # A different person overlapping the tracked location blocks
                # carry-forward; a weak match elsewhere is never accepted.
                self.assertEqual(tracker.select(self.frame,[detected]),[])

    def test_scene_cut_and_invalid_flow_reset_continuity(self):
        tracker = FaceTracker('selected',[[1.,0.]])
        tracker.select(self.frame,[person()])
        # No optical-flow attempt across a cut.
        with patch.object(tracker,'predict',wraps=tracker.predict) as predict:
            self.assertEqual(tracker.select(np.full_like(self.frame,255),[]),[])
            predict.assert_not_called()
        tracker.select(self.frame,[person()])
        with patch('face_tracker.cv2.calcOpticalFlowPyrLK',return_value=(None,None,None)):
            self.assertEqual(tracker.select(self.frame,[]),[])

    def test_real_optical_flow_follows_motion_without_repeating_old_pixels(self):
        rng = np.random.default_rng(9)
        original = cv2.GaussianBlur(rng.integers(0,256,(120,140,3),dtype=np.uint8),(3,3),0)
        moved = cv2.warpAffine(original,np.float32([[1,0,3],[0,1,2]]),(140,120))
        tracker = FaceTracker('selected',[[1.,0.]])
        tracker.select(original,[person()])
        recovered = tracker.select(moved,[])
        self.assertEqual(len(recovered),1)
        np.testing.assert_allclose(recovered[0].kps,person().kps+[3,2],atol=.5)
        self.assertAlmostEqual(recovered[0].bbox[0],33,delta=.5)

    def test_dropout_across_parallel_batch_boundary_is_order_independent(self):
        class Capture:
            def __init__(self): self.index=0
            def read(self):
                if self.index==7: return False,None
                frame=np.full((120,140,3),100+self.index,np.uint8)
                self.index+=1
                return True,frame

        class TestEngine(FaceEngine):
            def faces(self,frame):
                return [] if int(frame[0,0,0]) in {103,104} else [person()]
            def swap_targets(self,frame,source,targets):
                return frame,len(targets)

        with patch.object(FaceTracker,'predict',return_value=person()):
            for threads in [1,4]:
                outputs=list(processed_frames(TestEngine(),None,Capture(),'selected',threading.Event(),[[1.,0.]],threads))
                self.assertEqual([count for _,count in outputs],[1]*7)
                self.assertEqual([int(frame[0,0,0]) for frame,_ in outputs],list(range(100,107)))

    def test_tracked_pipeline_keeps_inference_parallel(self):
        barrier=threading.Barrier(4)
        class Capture:
            def __init__(self): self.index=0
            def read(self):
                if self.index==4: return False,None
                frame=np.full((120,140,3),100+self.index,np.uint8)
                self.index+=1
                return True,frame
        class TestEngine(FaceEngine):
            def faces(self,frame):
                barrier.wait(timeout=5)
                return [person()]
            def swap_targets(self,frame,source,targets):
                barrier.wait(timeout=5)
                return frame,len(targets)
        outputs=list(processed_frames(TestEngine(),None,Capture(),'all',threading.Event(),None,4))
        self.assertEqual([int(frame[0,0,0]) for frame,_ in outputs],list(range(100,104)))
