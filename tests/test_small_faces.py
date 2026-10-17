import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from engine import FaceEngine, face_tiles, same_face_box


class SmallFaceTests(unittest.TestCase):
    def setUp(self):
        angles = patch('engine.DETECTION_ANGLES',(0,))
        angles.start()
        self.addCleanup(angles.stop)
        mirrors = patch('engine.DETECTION_MIRRORS',(False,))
        mirrors.start()
        self.addCleanup(mirrors.stop)

    def test_tiles_cover_edges_and_bound_work_for_large_and_portrait_images(self):
        for width, height in [(1920,1080),(1080,1920),(8000,5000),(3000,300)]:
            tiles = face_tiles(width,height)
            self.assertLessEqual(len(tiles),36)
            self.assertEqual(max(x2 for _,_,x2,_ in tiles),width)
            self.assertEqual(max(y2 for _,_,_,y2 in tiles),height)
            for x in range(0,width,97):
                for y in range(0,height,97):
                    self.assertTrue(any(x1<=x<x2 and y1<=y<y2 for x1,y1,x2,y2 in tiles))
        self.assertEqual(face_tiles(200,150),[])

    def test_merge_partial_duplicate_but_keep_nearby_person(self):
        face = np.array([100,100,140,150])
        self.assertTrue(same_face_box(face,np.array([102,101,139,148])))
        self.assertTrue(same_face_box(face,np.array([120,100,140,150])))
        self.assertFalse(same_face_box(face,np.array([141,100,181,150])))

    def test_small_faces_added_from_tiles_with_correct_landmarks_and_one_embedding(self):
        engine = FaceEngine()
        box = np.array([[10,20,30,50,.9]],np.float32)
        points = np.array([[[14,28],[26,28],[20,35],[15,44],[25,44]]],np.float32)
        detector = Mock()
        # 400x200 -> two horizontal windows, the second begins at x=144.
        detector.detect.side_effect = [(np.empty((0,5)),None),(box,points)]
        recognition = Mock()
        engine.analyzer = SimpleNamespace(get=Mock(return_value=[]), det_model=detector,
                                           models={'recognition':recognition})
        image = np.zeros((200,400,3),np.uint8)
        faces = engine.faces(image)
        self.assertEqual(len(faces),1)
        np.testing.assert_allclose(faces[0].bbox,[154,20,174,50])
        np.testing.assert_allclose(faces[0].kps,points[0]+[144,0])
        recognition.get.assert_called_once()
        self.assertIs(recognition.get.call_args.args[0],image)

    def test_tiles_do_not_duplicate_full_frame_face(self):
        engine = FaceEngine()
        box = np.array([[10,20,30,50,.9]],np.float32)
        points = np.ones((1,5,2),np.float32)*20
        original = SimpleNamespace(bbox=box[0,:4])
        recognition = Mock()
        detector = Mock()
        detector.detect.side_effect = [(box,points),(np.empty((0,5)),None)]
        engine.analyzer = SimpleNamespace(get=Mock(return_value=[original]),det_model=detector,
                                          models={'recognition':recognition})
        faces = engine.faces(np.zeros((200,400,3),np.uint8))
        self.assertEqual(faces,[original])
        recognition.get.assert_not_called()
