"""Real-model HTTP editor check against a running app, using bundled demo media."""
import argparse
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import DATA_ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:7860')
args = parser.parse_args()
session = requests.Session()
token = re.search(r'name="app-token" content="([^"]+)"', session.get(args.url,timeout=10).text).group(1)
session.headers['X-App-Token'] = token
folder = DATA_ROOT/'smoke'
video_id = job_id = None


def check(response, code=200):
    assert response.status_code == code, response.text
    return response.json()


try:
    with (folder/'late-person.mp4').open('rb') as file:
        video_id = check(session.post(args.url+'/api/videos', files={'target':('late-person.mp4',file)},timeout=30),201)['id']
    first = check(session.post(f'{args.url}/api/videos/{video_id}/frame',json={'seconds':0,'mode':'cpu'},timeout=120))
    later = check(session.post(f'{args.url}/api/videos/{video_id}/frame',json={'seconds':.7,'mode':'cpu'},timeout=120))
    assert len(later['faces']) == len(first['faces'])+1
    print('REAL EDITOR: first frame',len(first['faces']),'faces; chosen later frame',len(later['faces']),'faces',flush=True)
    selected = later['faces'][1]['id']
    values = dict(frame_id=later['frame_id'],face_ids=str([selected]),selection='selected',mode='cpu')
    with (folder/'reference.png').open('rb') as file:
        preview = check(session.post(f'{args.url}/api/videos/{video_id}/preview',data=values,
                                    files={'source':('reference.png',file)},timeout=120))
    original_bytes = session.get(args.url+later['original'].replace('detected.jpg','original.png'),timeout=10).content
    result_bytes = session.get(args.url+preview['result'],timeout=10).content
    original = cv2.imdecode(np.frombuffer(original_bytes,np.uint8),cv2.IMREAD_COLOR)
    result = cv2.imdecode(np.frombuffer(result_bytes,np.uint8),cv2.IMREAD_COLOR)
    assert preview['faces'] == 1 and cv2.absdiff(original,result).sum() > 0
    (folder/'editor-preview.png').write_bytes(result_bytes)
    print('REAL EDITOR: selected-person preview passed',flush=True)
    with (folder/'reference.png').open('rb') as file:
        job_id = check(session.post(args.url+'/api/jobs',data=dict(values,video_id=video_id),
                                   files={'source':('reference.png',file)},timeout=30),202)['id']
    video_id = None
    deadline = time.monotonic()+240
    while True:
        job = check(session.get(f'{args.url}/api/jobs/{job_id}',timeout=10))
        if job['state'] in {'done','error','cancelled'}: break
        if time.monotonic() > deadline: raise TimeoutError('Video job exceeded test timeout')
        time.sleep(.5)
    assert job['state'] == 'done', job
    assert job['details']['frames'] == 3 and job['details']['swapped_frames'] == 1, job
    result = session.get(args.url+job['result'],timeout=30)
    assert result.status_code == 200
    (folder/'editor-result.mp4').write_bytes(result.content)
    print('REAL EDITOR PASSED: selected later person swaps only 1/3 frames; video uploaded once',flush=True)
finally:
    if video_id: session.delete(f'{args.url}/api/videos/{video_id}',timeout=10)
    if job_id: session.delete(f'{args.url}/api/jobs/{job_id}',timeout=10)
