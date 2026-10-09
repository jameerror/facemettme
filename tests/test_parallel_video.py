import threading
import unittest
from types import SimpleNamespace

import numpy as np

from engine import Cancelled, processed_frames
import app as web


class Capture:
    def __init__(self, count):
        self.count, self.reads = count, 0

    def read(self):
        if self.reads == self.count:
            return False, None
        frame = np.full((2, 2, 3), self.reads, np.uint8)
        self.reads += 1
        return True, frame


class ParallelVideoTests(unittest.TestCase):
    def test_concurrent_out_of_order_completion_writes_in_order_and_bounds_queue(self):
        capture = Capture(12)
        barrier = threading.Barrier(4)
        other_done = threading.Event()
        lock = threading.Lock()
        active = peak = done = 0
        completed = []

        def swap(frame, *args):
            nonlocal active, peak, done
            index = int(frame[0, 0, 0])
            with lock:
                active += 1
                peak = max(peak, active)
            if index < 4:
                barrier.wait(timeout=5)
                if index == 0:
                    self.assertTrue(other_done.wait(5))
                else:
                    with lock:
                        done += 1
                        if done == 3:
                            other_done.set()
            with lock:
                completed.append(index)
                active -= 1
            return frame, 1, None

        frames = processed_frames(SimpleNamespace(swap=swap), None, capture, 'all', threading.Event(), None, 4)
        output = []
        for result, _ in frames:
            self.assertLessEqual(capture.reads, len(output) + 4)
            output.append(int(result[0, 0, 0]))
        self.assertEqual(output, list(range(12)))
        self.assertEqual(peak, 4)
        self.assertNotEqual(completed[0], 0)
        self.assertEqual(active, 0)

    def test_principal_identity_is_established_before_parallel_work(self):
        identity = np.array([1., 0.])
        seen = []
        lock = threading.Lock()

        def swap(frame, source, selection, selected):
            index = int(frame[0, 0, 0])
            with lock:
                seen.append((index, selected))
            if index == 0:
                self.assertIsNone(selected)
                return frame, 0, None
            if index == 1:
                self.assertIsNone(selected)
                return frame, 1, identity
            np.testing.assert_array_equal(selected, identity)
            return frame, 1, selected

        result = list(processed_frames(SimpleNamespace(swap=swap), None, Capture(8),
                                       'largest', threading.Event(), None, 4))
        self.assertEqual([int(frame[0, 0, 0]) for frame, _ in result], list(range(8)))
        self.assertEqual([index for index, _ in seen[:2]], [0, 1])

    def test_cancellation_drains_running_workers(self):
        cancel, started, release = threading.Event(), threading.Event(), threading.Event()
        active = 0
        lock = threading.Lock()
        errors = []

        def swap(frame, *args):
            nonlocal active
            with lock:
                active += 1
            started.set()
            release.wait(5)
            with lock:
                active -= 1
            return frame, 1, None

        def consume():
            try:
                list(processed_frames(SimpleNamespace(swap=swap), None, Capture(40), 'all', cancel, None, 4))
            except Exception as exc:
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        try:
            self.assertTrue(started.wait(5))
            cancel.set()
        finally:
            release.set()
            consumer.join(5)
        self.assertFalse(consumer.is_alive())
        self.assertEqual(active, 0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], Cancelled)

    def test_worker_error_is_propagated(self):
        def swap(*args):
            raise ValueError('test inference failure')
        with self.assertRaisesRegex(ValueError, 'test inference failure'):
            list(processed_frames(SimpleNamespace(swap=swap), None, Capture(10),
                                  'all', threading.Event(), None, 4))

    def test_api_rejects_invalid_thread_counts_and_forwards_valid_count(self):
        import io
        from unittest.mock import patch
        client = web.app.test_client()
        for value in ['0', '33', '1.5', 'invalid']:
            response = client.post('/api/jobs', headers={'X-App-Token': web.TOKEN}, data={
                'source': (io.BytesIO(b'x'), 'source.png'), 'target': (io.BytesIO(b'x'), 'target.mp4'),
                'execution_thread_count': value})
            self.assertEqual(response.status_code, 400)
        with patch.object(web.executor, 'submit') as submit:
            response = client.post('/api/jobs', headers={'X-App-Token': web.TOKEN}, data={
                'source': (io.BytesIO(b'x'), 'source.png'), 'target': (io.BytesIO(b'x'), 'target.mp4'),
                'execution_thread_count': '16'})
        try:
            self.assertEqual(response.status_code, 202)
            self.assertEqual(submit.call_args.args[-1], 16)
        finally:
            if response.status_code == 202:
                job_id = response.json['id']
                web.jobs[job_id]['state'] = 'cancelled'
                client.delete('/api/jobs/' + job_id, headers={'X-App-Token': web.TOKEN})
