from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import app as web
from settings import load_settings


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.original = web.app.config['SERVER_SETTINGS']
        with patch.dict(os.environ, {}, clear=True):
            self.local = load_settings()
        self.cloud = replace(self.local, host='0.0.0.0', cloud=True,
                             trusted_hosts=('cloud.example', '127.0.0.1', 'localhost'), max_upload_mb=2)
        web.configure(self.cloud)
        self.client = web.app.test_client()
        self.headers = {'Host': 'cloud.example'}

    def tearDown(self):
        web.configure(self.original)

    def test_cloud_opens_pages_api_and_static_without_login(self):
        for path in ['/', '/api/status', '/static/app.js']:
            response = self.client.get(path, headers={'Host': 'cloud.example'})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn('WWW-Authenticate', response.headers)
            response.close()

    def test_direct_access_and_cloud_copy(self):
        response = self.client.get('/', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn('ประมวลผลบนเซิร์ฟเวอร์ของคุณ'.encode(), response.data)
        self.assertEqual(self.client.get('/api/status', headers=self.headers).json['max_upload_mb'], 2)

    def test_results_download_without_login(self):
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory:
            folder = Path(directory)
            (folder/'result.png').write_bytes(b'test-result')
            with patch.dict(web.jobs, {'cloud-test': {'state':'done', 'kind':'image', 'folder':folder}}):
                response = self.client.get('/api/jobs/cloud-test/result?download=1', headers=self.headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b'test-result')
                self.assertNotIn('WWW-Authenticate', response.headers)
                response.close()

    def test_untrusted_cloud_host_rejected(self):
        self.assertEqual(self.client.get('/', headers=dict(self.headers, Host='evil.example')).status_code, 400)

    def test_health_no_auth_no_sensitive_details(self):
        response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'ok'})

    def test_cloud_mutations_still_require_token(self):
        self.assertEqual(self.client.post('/api/jobs', headers=self.headers).status_code, 403)
        headers = dict(self.headers, **{'X-App-Token': web.TOKEN})
        self.assertEqual(self.client.post('/api/jobs', headers=headers).status_code, 400)

    def test_upload_limit(self):
        headers = dict(self.headers, **{'X-App-Token': web.TOKEN})
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory, \
             patch.object(web, 'DATA', Path(directory)), patch.dict(web.jobs, {}, clear=True), \
             patch.object(web.executor, 'submit') as submit:
            response = self.client.post('/api/jobs', data={
                'source': (io.BytesIO(b'reference'), 'source.png'),
                'target': (io.BytesIO(b'x'*(2*1024**2+1)), 'target.mp4')}, headers=headers)
            self.assertEqual(response.status_code, 413)
            submit.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(web.jobs, {})
            response.close()
            response.request.environ['wsgi.input'].close()

    def test_exact_target_limit_plus_reference_is_accepted(self):
        headers = dict(self.headers, **{'X-App-Token': web.TOKEN})
        with tempfile.TemporaryDirectory(dir=web.DATA) as directory, \
             patch.object(web, 'DATA', Path(directory)), patch.dict(web.jobs, {}, clear=True), \
             patch.object(web.executor, 'submit') as submit:
            response = self.client.post('/api/jobs', data={
                'source': (io.BytesIO(b'x'*1024), 'source.png'),
                'target': (io.BytesIO(b'x'*(2*1024**2)), 'target.mp4')}, headers=headers)
            self.assertEqual(response.status_code, 202)
            submit.assert_called_once()
            response.close()
            response.request.environ['wsgi.input'].close()

    def test_default_five_gb_and_request_overhead(self):
        from settings import REFERENCE_MAX_MB, MULTIPART_OVERHEAD_MB
        from waitress.adjustments import Adjustments
        with patch.dict(os.environ, {}, clear=True):
            config = load_settings()
        self.assertEqual(config.max_upload_mb, 5120)
        self.assertEqual(config.max_request_bytes, 5*1024**3 + (REFERENCE_MAX_MB+MULTIPART_OVERHEAD_MB)*1024**2)
        self.assertGreater(config.max_request_bytes, 2**32)
        self.assertEqual(Adjustments(max_request_body_size=config.max_request_bytes).max_request_body_size,
                         config.max_request_bytes)
        web.configure(config)
        self.assertEqual(web.app.config['MAX_CONTENT_LENGTH'], config.max_request_bytes)
        self.assertIn('ไฟล์เป้าหมายสูงสุด 5 GB'.encode(), self.client.get('/').data)

    def test_request_exceeding_total_limit_is_rejected_before_read(self):
        headers = dict(self.headers, **{'X-App-Token': web.TOKEN})
        response = self.client.post('/api/jobs', headers=headers, environ_overrides={
            'CONTENT_LENGTH': str(self.cloud.max_request_bytes+1),
            'CONTENT_TYPE': 'multipart/form-data; boundary=example', 'wsgi.input': io.BytesIO()})
        self.assertEqual(response.status_code, 413)

    def test_local_mode_remains_without_login(self):
        web.configure(self.local)
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('ประมวลผลในเครื่องคุณ'.encode(), response.data)

    def test_nonloopback_starts_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(load_settings(host='0.0.0.0').cloud)
        with patch.dict(os.environ, {'FACE_SWAP_PASSWORD': 'short'}, clear=True):
            self.assertTrue(load_settings(host='0.0.0.0').cloud)

    def test_environment_port_host(self):
        with patch.dict(os.environ, {'FACE_SWAP_HOST':'0.0.0.0', 'PORT':'8080',
                                     'FACE_SWAP_TRUSTED_HOSTS':'cloud.example, localhost'}, clear=True):
            config = load_settings()
        self.assertTrue(config.cloud)
        self.assertEqual(config.port, 8080)
        self.assertEqual(config.trusted_hosts, ('cloud.example', 'localhost'))


if __name__ == '__main__':
    unittest.main()
