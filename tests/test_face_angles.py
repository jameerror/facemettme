import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from engine import FaceEngine, DETECTION_ANGLES, rotated_view, detection_views


class FaceAngleTests(unittest.TestCase):
    def test_mirrored_views_map_points_and_restore_eye_mouth_order(self):
        image = np.zeros((100,160,3),np.uint8)
        points = np.array([[40,30],[60,30],[50,40],[42,52],[58,52]],np.float32)
        for angle,mirrored,view,inverse in detection_views(image):
            with self.subTest(angle=angle,mirrored=mirrored):
                forward = cv2.invertAffineTransform(inverse)
                moved = cv2.transform(points[None],forward)[0]
                detector_points = moved[[1,0,2,4,3]] if mirrored else moved
                restored = detector_points[[1,0,2,4,3]] if mirrored else detector_points
                np.testing.assert_allclose(cv2.transform(restored[None],inverse)[0],points,atol=1e-4)

    def test_detection_found_only_in_mirror_gets_original_landmark_order(self):
        image = np.zeros((100,160,3),np.uint8)
        points = np.array([[40,30],[60,30],[50,40],[42,52],[58,52]],np.float32)
        reflected = points.copy()
        reflected[:,0] = 159-reflected[:,0]
        detector = Mock()
        detector.detect.return_value = (np.array([[89,20,129,60,.9]],np.float32),
                                        reflected[[1,0,2,4,3]][None])
        recognition = Mock()
        engine = FaceEngine()
        engine.analyzer = SimpleNamespace(get=Mock(return_value=[]),det_model=detector,
                                          models={'recognition':recognition})
        with patch('engine.DETECTION_ANGLES',(0,)):
            faces = engine.faces(image)
        self.assertEqual(len(faces),1)
        np.testing.assert_allclose(faces[0].bbox,[30,20,70,60])
        np.testing.assert_allclose(faces[0].kps,points)
        recognition.get.assert_called_once()

    def test_all_eight_angles_preserve_points_on_non_square_images(self):
        image = np.zeros((90,170,3),np.uint8)
        corners = np.array([[[0,0],[169,0],[169,89],[0,89],[80,40]]],np.float32)
        for angle in DETECTION_ANGLES:
            with self.subTest(angle=angle):
                view, inverse = rotated_view(image,angle)
                forward = cv2.invertAffineTransform(inverse)
                moved = cv2.transform(corners,forward)
                self.assertGreaterEqual(moved.min(),-.01)
                self.assertLessEqual(moved[0,:,0].max(),view.shape[1])
                self.assertLessEqual(moved[0,:,1].max(),view.shape[0])
                np.testing.assert_allclose(cv2.transform(moved,inverse),corners,atol=1e-4)

    def test_rotated_detection_maps_box_and_landmarks_and_merges_duplicates(self):
        image = np.zeros((100,160,3),np.uint8)
        points = np.array([[40,30],[60,30],[50,40],[42,52],[58,52]],np.float32)
        corners = np.array([[[30,20],[70,20],[70,60],[30,60]]],np.float32)
        outputs = []
        for angle in [90,270]:
            _, inverse = rotated_view(image,angle)
            forward = cv2.invertAffineTransform(inverse)
            box = cv2.transform(corners,forward)[0]
            outputs.append((np.array([np.r_[box.min(axis=0),box.max(axis=0),.9]]),
                            cv2.transform(points[None],forward)))
        detector = Mock()
        detector.detect.side_effect = outputs
        recognition = Mock()
        engine = FaceEngine()
        weak_original = SimpleNamespace(bbox=np.array([30,20,70,60],np.float32),
                                        kps=points[::-1],det_score=.5)
        engine.analyzer = SimpleNamespace(get=Mock(return_value=[weak_original]),det_model=detector,
                                          models={'recognition':recognition})
        with patch('engine.DETECTION_ANGLES',(90,270)), patch('engine.DETECTION_MIRRORS',(False,)):
            faces = engine.faces(image)
        self.assertEqual(len(faces),1)
        np.testing.assert_allclose(faces[0].bbox,[30,20,70,60],atol=1e-4)
        np.testing.assert_allclose(faces[0].kps,points,atol=1e-4)
        recognition.get.assert_called_once()
        self.assertIs(recognition.get.call_args.args[0],image)
