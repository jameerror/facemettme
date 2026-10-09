"""Optional HTTP integration check with real models and direct cloud access."""
from pathlib import Path
import os
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import DATA_ROOT

folder = DATA_ROOT / 'smoke'
required = [folder/'reference.png', folder/'target.png', folder/'target.mp4']
if not all(path.is_file() for path in required):
    raise SystemExit('Run tests/smoke_models.py first to create demo inputs.')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
env = dict(os.environ, FACE_SWAP_CLOUD='1', FACE_SWAP_HOST='127.0.0.1', PORT=str(port),
           FACE_SWAP_TRUSTED_HOSTS='127.0.0.1,localhost')
url = f'http://127.0.0.1:{port}'
session = requests.Session()
with (folder/'cloud-server.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT/'app.py'), '--no-browser'],
                               cwd=ROOT, env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+60
        while True:
            try:
                if requests.get(url+'/healthz', timeout=2).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Cloud server failed to start; see cloud-server.log')
            time.sleep(.2)
        assert requests.get(url, timeout=5).status_code == 200
        assert requests.get(url+'/static/app.js', timeout=5).status_code == 200
        page = session.get(url, timeout=5)
        assert page.status_code == 200
        assert 'ประมวลผลบนเซิร์ฟเวอร์ของคุณ' in page.text
        import re
        token = re.search(r'name="app-token" content="([^"]+)"', page.text).group(1)
        session.headers['X-App-Token'] = token
        for target in required[1:]:
            with required[0].open('rb') as source_file, target.open('rb') as target_file:
                response = session.post(url+'/api/jobs', files={'source':(required[0].name, source_file),
                                                              'target':(target.name, target_file)},
                                        data={'mode':'cpu', 'selection':'largest', 'size':'640'}, timeout=30)
            assert response.status_code == 202, response.text
            job_id = response.json()['id']
            deadline = time.monotonic()+240
            while True:
                job = session.get(url+'/api/jobs/'+job_id, timeout=5).json()
                if job['state'] in {'done','error','cancelled'}:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('Model job timed out')
                time.sleep(.5)
            assert job['state'] == 'done', job
            output_url = url+job['result']
            assert requests.get(output_url, timeout=5).status_code == 200
            download = session.get(output_url+'?download=1', timeout=10)
            assert download.status_code == 200 and len(download.content) > 1000
            assert 'attachment' in download.headers['Content-Disposition']
            assert session.delete(url+'/api/jobs/'+job_id, timeout=5).status_code == 200
            print('CLOUD HTTP PASSED:', target.suffix, job['backend'], job['details'], flush=True)
        print('Direct access, upload, real inference, download and deletion passed.', flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
