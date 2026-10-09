import io
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import app as web
from engine import FaceEngine, Cancelled, providers_for, process_video
from settings import load_settings


class FaceSelectionTests(unittest.TestCase):
    def test_cpu_and_auto(self):
        self.assertEqual(providers_for('auto', ['CPUExecutionProvider']), ['CPUExecutionProvider'])
        self.assertEqual(providers_for('cpu', ['CUDAExecutionProvider', 'CPUExecutionProvider']), ['CPUExecutionProvider'])
        self.assertEqual(providers_for('auto', ['CUDAExecutionProvider', 'CPUExecutionProvider'])[0], 'CUDAExecutionProvider')

    def test_unavailable_gpu_fails(self):
        with self.assertRaises(ValueError):
            providers_for('cuda', ['CPUExecutionProvider'])

    def test_tracking_keeps_identity_when_face_order_changes(self):
        a = SimpleNamespace(bbox=np.array([0, 0, 20, 20]), normed_embedding=np.array([1., 0.]))
        b = SimpleNamespace(bbox=np.array([30, 0, 90, 60]), normed_embedding=np.array([0., 1.]))
        engine = FaceEngine()
        engine.faces = lambda _: [a, b]
        selected = []
        engine.swapper = SimpleNamespace(get=lambda image, face, source, paste_back: (selected.append(face) or image))
        image = np.zeros((100, 100, 3), np.uint8)
        _, count, identity = engine.swap(image, a, 'largest')
        self.assertEqual(count, 1)
        self.assertIs(selected[-1], b)
        engine.faces = lambda _: [b, a]
        engine.swap(image, a, 'largest', identity)
        self.assertIs(selected[-1], b)
        engine.faces = lambda _: [a]
        _, count, _ = engine.swap(image, a, 'largest', identity)
        self.assertEqual(count, 0)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.original_settings = web.app.config['SERVER_SETTINGS']
        with patch.dict(os.environ, {}, clear=True):
            web.configure(load_settings())
        self.temp = tempfile.TemporaryDirectory(dir=web.DATA)
        self.data_patch = patch.object(web, 'DATA', Path(self.temp.name))
        self.data_patch.start()
        self.client = web.app.test_client()
        self.headers = {'X-App-Token': web.TOKEN}
        web.jobs.clear()

    def tearDown(self):
        web.configure(self.original_settings)
        web.jobs.clear()
        self.data_patch.stop()
        self.temp.cleanup()

    def test_home_and_status(self):
        self.assertIn('Face Swap Me TT Me'.encode(), self.client.get('/').data)
        self.assertEqual(self.client.get('/api/status').status_code, 200)

    def test_token_and_host(self):
        self.assertEqual(self.client.post('/api/jobs').status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host': 'evil.example'}).status_code, 400)

    def test_upload_validation(self):
        self.assertEqual(self.client.post('/api/jobs', headers=self.headers).status_code, 400)
        response = self.client.post('/api/jobs', headers=self.headers, data={
            'source': (io.BytesIO(b'bad'), 'a.exe'), 'target': (io.BytesIO(b'bad'), 'b.png')})
        self.assertEqual(response.status_code, 400)

    def test_job_cancel_and_delete(self):
        with patch.object(web.executor, 'submit'):
            response = self.client.post('/api/jobs', headers=self.headers, data={
                'source': (io.BytesIO(b'image'), 'a.png'), 'target': (io.BytesIO(b'image'), 'b.png')})
        self.assertEqual(response.status_code, 202)
        job_id = response.json['id']
        self.assertNotIn('folder', self.client.get('/api/jobs/'+job_id).json)
        self.assertEqual(self.client.post('/api/jobs/'+job_id+'/cancel', headers=self.headers).status_code, 200)
        self.assertTrue(web.jobs[job_id]['cancel'].is_set())
        self.assertEqual(self.client.delete('/api/jobs/'+job_id, headers=self.headers).status_code, 409)
        web.jobs[job_id]['state'] = 'cancelled'
        self.assertEqual(self.client.delete('/api/jobs/'+job_id, headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get('/api/jobs/'+job_id).status_code, 404)

    def test_worker_image_and_download(self):
        class FakeEngine:
            def load(self, *args): return 'CPU'
            def source(self, path): return None
            def swap(self, image, *args): return image, 1, None
        folder = Path(self.temp.name)/'job'
        folder.mkdir()
        src, dst = folder/'source.png', folder/'target.png'
        cv2.imwrite(str(src), np.zeros((32,32,3), np.uint8))
        cv2.imwrite(str(dst), np.zeros((32,32,3), np.uint8))
        web.jobs['job'] = dict(state='queued', cancel=threading.Event(), folder=folder)
        with patch.object(web, 'engine', FakeEngine()):
            web.worker('job', src, dst, 'cpu', 640, 'all', 'image')
        self.assertEqual(web.jobs['job']['state'], 'done')
        self.assertFalse(src.exists())
        self.assertFalse(dst.exists())
        response = self.client.get('/api/jobs/job/result?download=1')
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['Content-Disposition'])
        response.close()


class VideoTests(unittest.TestCase):
    def test_real_video_pipeline_audio_and_frame_count(self):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        class FakeEngine:
            def swap(self, frame, *args): return frame, 1, None
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory:
            folder = Path(directory)
            target, output = folder/'test.mp4', folder/'result.mp4'
            subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=10:d=1',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'libx264',
                            '-c:a', 'aac', '-shortest', str(target)], check=True, capture_output=True)
            details = process_video(FakeEngine(), None, target, output, 'all', lambda *args: None, threading.Event())
            self.assertEqual(details['frames'], 10)
            probe = subprocess.run([ffmpeg, '-i', str(output), '-f', 'null', '-'], capture_output=True)
            self.assertEqual(probe.returncode, 0)
            self.assertIn(b'Audio: aac', probe.stderr)
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            cap.release()
            self.assertFalse((folder/'silent.mp4').exists())

    def test_cancel_before_video_work(self):
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory:
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(Cancelled):
                process_video(None, None, Path(directory)/'missing.mp4', Path(directory)/'result.mp4',
                              'all', lambda *args: None, cancel)


if __name__ == '__main__':
    unittest.main()
