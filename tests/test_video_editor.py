import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import app as web
from engine import FaceEngine, process_video
from video_editor import extract_frame, video_metadata


def face(box, embedding):
    return SimpleNamespace(bbox=np.array(box), normed_embedding=np.array(embedding, dtype=np.float32))


class FakeEngine(FaceEngine):
    def __init__(self):
        super().__init__()
        self.people = [face([5,5,25,25], [1,0]), face([50,5,70,25], [0,1])]
        self.swapper = SimpleNamespace(get=self.paint)

    def load(self, *args): return 'CPU'
    def faces(self, image): return self.people
    def source(self, path): return self.people[0]
    def paint(self, image, target, source, paste_back):
        output = image.copy()
        x1,y1,x2,y2 = map(int, target.bbox)
        output[y1:y2,x1:x2] = (0,255,0)
        return output


class EditorApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=web.DATA)
        self.patches = [patch.object(web, 'DATA', Path(self.temp.name)),
                        patch.object(web, 'engine', FakeEngine()),
                        patch.dict(web.jobs, {}, clear=True), patch.dict(web.editor.videos, {}, clear=True),
                        patch('video_editor.video_metadata', return_value={'duration':10.,'fps':10.,'width':80,'height':80}),
                        patch('video_editor.extract_frame', return_value=np.zeros((80,80,3), np.uint8))]
        for item in self.patches: item.start()
        self.client = web.app.test_client()
        self.headers = {'X-App-Token':web.TOKEN}
        self.video_id = self.client.post('/api/videos', headers=self.headers,
            data={'target':(io.BytesIO(b'video'), 'target.mp4')}).json['id']

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def inspect(self, seconds=5):
        return self.client.post(f'/api/videos/{self.video_id}/frame', headers=self.headers,
                                json={'seconds':seconds, 'mode':'cpu', 'size':640})

    def source(self):
        _, data = cv2.imencode('.png', np.zeros((32,32,3), np.uint8))
        return (io.BytesIO(data.tobytes()), 'reference.png')

    def test_faces_have_images_and_private_embeddings(self):
        response = self.inspect()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['faces']), 2)
        self.assertNotIn('embedding', response.get_data(as_text=True))
        for person in response.json['faces']:
            image = self.client.get(person['thumbnail'])
            self.assertEqual(image.status_code, 200)
            self.assertIsNotNone(cv2.imdecode(np.frombuffer(image.data,np.uint8),cv2.IMREAD_COLOR))
            image.close()

    def test_preview_changes_only_selected_person(self):
        frame_id = self.inspect().json['frame_id']
        response = self.client.post(f'/api/videos/{self.video_id}/preview', headers=self.headers, data={
            'source':self.source(), 'frame_id':frame_id, 'face_ids':'[1]', 'selection':'selected','mode':'cpu'})
        self.assertEqual(response.status_code, 200)
        image_response = self.client.get(response.json['result'])
        image = cv2.imdecode(np.frombuffer(image_response.data,np.uint8),cv2.IMREAD_COLOR)
        image_response.close()
        self.assertTrue(np.all(image[5:25,5:25] == 0))
        self.assertTrue(np.all(image[5:25,50:70] == (0,255,0)))
        folder = web.editor.videos[self.video_id]['folder']
        self.assertFalse(list(folder.glob('reference*')))

    def test_stale_frame_and_invalid_faces_rejected(self):
        old = self.inspect().json['frame_id']
        current = self.inspect(7).json['frame_id']
        for frame_id, ids in [(old,'[1]'), (current,'[99]'), (current,'[]'), (current,'[true]')]:
            response = self.client.post('/api/jobs', headers=self.headers, data={
                'source':self.source(), 'video_id':self.video_id, 'frame_id':frame_id,
                'face_ids':ids, 'selection':'selected'})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(web.editor.videos[self.video_id]['claimed'])

    def test_job_reuses_video_and_receives_selected_identity(self):
        frame_id = self.inspect().json['frame_id']
        target = web.editor.videos[self.video_id]['path']
        with patch.object(web.executor, 'submit') as submit:
            response = self.client.post('/api/jobs', headers=self.headers, data={
                'source':self.source(), 'video_id':self.video_id, 'frame_id':frame_id,
                'face_ids':'[1]', 'selection':'selected'})
        self.assertEqual(response.status_code, 202)
        args = submit.call_args.args
        self.assertEqual(args[3], target)
        np.testing.assert_array_equal(args[-2], [[0,1]])
        self.assertEqual(args[-1], self.video_id)
        self.assertEqual(self.client.delete('/api/videos/'+self.video_id, headers=self.headers).status_code, 409)

    def test_extract_invalid_time_and_no_faces(self):
        for seconds in [-1, 10, 'nan', 'bad']:
            self.assertEqual(self.inspect(seconds).status_code, 400)
        with patch.object(web.engine, 'faces', return_value=[]):
            self.assertEqual(self.inspect().json['faces'], [])

    def test_delete_video_removes_assets(self):
        folder = web.editor.videos[self.video_id]['folder']
        self.inspect()
        self.assertEqual(self.client.delete('/api/videos/'+self.video_id, headers=self.headers).status_code, 200)
        self.assertFalse(folder.exists())
        self.assertEqual(self.inspect().status_code, 404)

    def test_preview_and_video_cannot_use_engine_concurrently(self):
        web.inference_lock.acquire()
        try:
            self.assertEqual(self.inspect().status_code, 409)
        finally:
            web.inference_lock.release()


class SelectedIdentityTests(unittest.TestCase):
    def test_late_person_skips_other_people_and_keeps_identity(self):
        engine = FakeEngine()
        selected = np.array([[0.,1.]])
        image = np.zeros((80,80,3),np.uint8)
        with patch.object(engine, 'faces', return_value=engine.people[:1]):
            output, count, identity = engine.swap(image, None, 'selected', selected)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(output, image)
        np.testing.assert_array_equal(identity, selected)
        output, count, identity = engine.swap(image, None, 'selected', identity)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(output[5:25,5:25] == 0))

    def test_multiple_selected_people_match_once_each(self):
        engine = FakeEngine()
        result, count, _ = engine.swap(np.zeros((80,80,3),np.uint8), None, 'selected', np.eye(2))
        self.assertEqual(count, 2)
        self.assertTrue(np.all(result[5:25,5:25] == (0,255,0)))

    def test_real_seek_and_video_person_appearing_later(self):
        engine = FakeEngine()
        # First half contains another person; the selected identity enters only in the second half.
        engine.faces = lambda image: [engine.people[0] if image.mean() < 100 else engine.people[1]]
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory:
            target, output = Path(directory)/'target.mp4', Path(directory)/'result.mp4'
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80,80))
            self.assertTrue(writer.isOpened())
            for i in range(10): writer.write(np.full((80,80,3), 0 if i < 5 else 255,np.uint8))
            writer.release()
            self.assertLess(extract_frame(target, .1).mean(), 10)
            self.assertGreater(extract_frame(target, .7).mean(), 240)
            self.assertGreater(extract_frame(target, .999).mean(), 240)
            self.assertAlmostEqual(video_metadata(target)['duration'], 1.)
            details = process_video(engine, None, target, output, 'selected', lambda *args:None,
                                    threading.Event(), np.array([[0.,1.]]))
            self.assertEqual(details['frames'], 10)
            self.assertEqual(details['swapped_frames'], 5)


if __name__ == '__main__':
    unittest.main()
