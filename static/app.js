const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
let currentJob = null;
let timer = null;
let busy = false;
let videoSession = null;
let frameId = null;
let frameSeconds = null;
let duration = 0;
const isVideoTarget = () => /\.(mp4|mov|mkv|avi|webm)$/i.test($('target').files[0]?.name || '');
const selectedFaces = () => [...$('face-picker').querySelectorAll('input:checked')].map(el => Number(el.value));
const urls = {};
let maxUploadMB = 5120;
let maxReferenceMB = 50;
const formatLimit = mb => mb >= 1024 ? `${mb / 1024} GB` : `${mb} MB`;

async function api(url, options = {}) {
  options.headers = {...options.headers, 'X-App-Token': token};
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `คำขอไม่สำเร็จ (${response.status})`);
  return data;
}
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function setBusy(value, processingJob = true) {
  busy = value;
  document.querySelectorAll('#swap-form input,#swap-form select,#swap-form button').forEach(el => el.disabled = value);
  $('cancel').hidden = !value || !processingJob;
  updateEditorButtons();
}
function preview(name) {
  const file = $(name).files[0];
  if (!file) return;
  if (name === 'source') clearSwappedPreview();
  if (name === 'target') {
    clearVideoSession();
    duration = 0; setFrameTime(0);
    $('video-duration').textContent = 'ยังไม่ทราบความยาว';
    $('video-editor').hidden = !isVideoTarget();
    const selectedOption = $('selection').querySelector('[value="selected"]');
    const largestOption = $('selection').querySelector('[value="largest"]');
    selectedOption.hidden = selectedOption.disabled = !isVideoTarget();
    largestOption.hidden = largestOption.disabled = isVideoTarget();
    $('selection').value = isVideoTarget() ? 'selected' : 'largest';
    $('selection-hint').textContent = isVideoTarget()
      ? 'เลือกคนจากรูปในเฟรมตัวอย่างเพื่อให้ติดตามคนเดิมตลอดคลิป · ทุกใบหน้า = รวมคนที่ปรากฏในเฟรมอื่นด้วย'
      : 'คนหลัก = ใบหน้าที่ใหญ่ที่สุดในภาพ · ความละเอียดตรวจจับไม่ได้เพิ่มความละเอียดใบหน้าของโมเดล';
  }
  if (urls[name]) URL.revokeObjectURL(urls[name]);
  urls[name] = URL.createObjectURL(file);
  const image = $(name + '-preview');
  const video = name === 'target' ? $('target-video') : null;
  image.hidden = true;
  if (video) { video.pause(); video.removeAttribute('src'); video.hidden = true; }
  const isVideo = /\.(mp4|mov|mkv|avi|webm)$/i.test(file.name);
  const media = isVideo && video ? video : image;
  media.src = urls[name]; media.hidden = false;
  $(name + '-drop').querySelector('.empty').hidden = true;
  $(name + '-name').textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(1) + ' MB';
  media.onerror = () => { media.hidden = true; $(name + '-name').textContent = file.name + ' · เบราว์เซอร์แสดงตัวอย่างไฟล์นี้ไม่ได้ แต่ยังประมวลผลได้'; };
}
['source', 'target'].forEach(name => {
  $(name).addEventListener('change', () => preview(name));
  const drop = $(name + '-drop');
  drop.addEventListener('dragover', e => { e.preventDefault(); if (!busy) drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag');
    if (busy || !e.dataTransfer.files.length) return;
    const dt = new DataTransfer(); dt.items.add(e.dataTransfer.files[0]);
    $(name).files = dt.files; preview(name);
  });
});
api('/api/status').then(data => {
  maxUploadMB = data.max_upload_mb;
  maxReferenceMB = data.max_reference_mb;
  const cuda = data.providers.includes('CUDAExecutionProvider');
  $('mode').querySelector('[value="cuda"]').disabled = !cuda;
  $('setup').classList.toggle('warn', !data.model_ready);
  $('setup').textContent = data.model_ready
    ? `โมเดลพร้อมใช้งาน · ${cuda ? 'พบ CUDA runtime · ระบบจะตรวจ GPU เมื่อเริ่มงาน' : 'พร้อมใช้ CPU'}`
    : 'ยังไม่มีโมเดล · รัน setup_models.py ก่อนเริ่มใช้งาน (ดูวิธีใน README.md)';
}).catch(error => showError(error.message));

async function poll() {
  try {
    const job = await api('/api/jobs/' + currentJob);
    $('bar').style.width = job.progress + '%'; $('percent').textContent = job.progress + '%';
    $('message').textContent = job.message; $('backend').textContent = job.backend;
    sessionStorage.setItem('face-swap-job', currentJob);
    if (['done', 'error', 'cancelled'].includes(job.state)) {
      setBusy(false);
      if (isVideoTarget()) $('editor-message').textContent = 'งานเสร็จแล้ว หากต้องการทำใหม่ ให้โหลดเฟรมเพื่ออัปโหลดวิดีโออีกครั้ง';
      $('delete').hidden = false;
      $('job-title').textContent = job.state === 'done' ? 'ผลลัพธ์ของคุณ' : job.state === 'error' ? 'ประมวลผลไม่สำเร็จ' : 'ยกเลิกแล้ว';
      if (job.state === 'done') {
        const media = $(job.kind === 'video' ? 'result-video' : 'result-image');
        media.src = job.result; media.hidden = false;
        $('download').href = job.result + '?download=1'; $('download').hidden = false;
        $('details').textContent = job.kind === 'video'
          ? `สลับใบหน้า ${job.details.swapped_frames.toLocaleString()} / ${job.details.frames.toLocaleString()} เฟรม · ${job.details.fps} FPS`
          : `สลับใบหน้า ${job.details.faces} ใบหน้า · PNG`;
      }
      return;
    }
    timer = setTimeout(poll, 1000);
  } catch (error) {
    showError(error.message);
    if (currentJob) timer = setTimeout(poll, 3000);
  }
}
function resetResult() {
  clearTimeout(timer);
  ['result-image', 'result-video', 'download', 'delete'].forEach(id => $(id).hidden = true);
  $('result-video').pause(); $('result-video').removeAttribute('src');
  $('result-image').removeAttribute('src'); $('details').textContent = '';
  $('job-title').textContent = 'กำลังสร้างผลลัพธ์'; $('backend').textContent = '';
  $('bar').style.width = '0%'; $('percent').textContent = '0%';
}
$('swap-form').addEventListener('submit', async e => {
  e.preventDefault(); $('error').hidden = true;
  const form = new FormData($('swap-form'));
  if (form.get('target').size > maxUploadMB * 1024 ** 2) return showError(`ไฟล์เป้าหมายต้องไม่เกิน ${formatLimit(maxUploadMB)}`);
  if (form.get('source').size > maxReferenceMB * 1024 ** 2) return showError(`รูปอ้างอิงต้องไม่เกิน ${formatLimit(maxReferenceMB)}`);
  if (isVideoTarget()) {
    if (!videoSession) return showError('กรุณาโหลดเฟรมตัวอย่างของวิดีโอก่อน');
    if ($('selection').value === 'selected' && (!frameId || !selectedFaces().length))
      return showError('กรุณาเลือกใบหน้าจากรูปในเฟรมตัวอย่างอย่างน้อยหนึ่งคน');
    form.delete('target');
    form.set('video_id', videoSession);
    if (frameId) form.set('frame_id', frameId);
    form.set('face_ids', JSON.stringify(selectedFaces()));
  }
  setBusy(true);
  try {
    const data = await api('/api/jobs', {method: 'POST', body: form});
    currentJob = data.id;
    if (isVideoTarget()) {
      videoSession = null;
      invalidateFrame();
    }
    resetResult(); $('job-panel').hidden = false;
    $('message').textContent = 'รอประมวลผล';
    poll(); $('job-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { setBusy(false); showError(error.message); }
});
$('cancel').addEventListener('click', async () => {
  try { await api('/api/jobs/' + currentJob + '/cancel', {method: 'POST'}); $('message').textContent = 'กำลังยกเลิก…'; }
  catch (error) { showError(error.message); }
});
$('delete').addEventListener('click', async () => {
  try {
    $('result-video').pause(); $('result-video').removeAttribute('src');
    await api('/api/jobs/' + currentJob, {method: 'DELETE'});
    currentJob = null; sessionStorage.removeItem('face-swap-job'); $('job-panel').hidden = true;
  } catch (error) { showError(error.message); }
});
const saved = sessionStorage.getItem('face-swap-job');
if (saved) api('/api/jobs/' + saved).then(() => {
  currentJob = saved; $('job-panel').hidden = false; setBusy(true); poll();
}).catch(() => sessionStorage.removeItem('face-swap-job'));

function updateEditorButtons() {
  $('swap-preview').disabled = busy || !frameId || !$('source').files.length ||
    ($('selection').value === 'selected' && !selectedFaces().length);
}
function clearSwappedPreview() {
  $('frame-swapped').hidden = true;
  $('frame-swapped').removeAttribute('src');
  $('preview-placeholder').hidden = false;
  updateEditorButtons();
}
function invalidateFrame() {
  frameId = null; frameSeconds = null;
  $('face-picker').replaceChildren();
  $('frame-comparison').hidden = true;
  $('frame-original').removeAttribute('src');
  clearSwappedPreview();
}
function clearVideoSession() {
  const previous = videoSession;
  videoSession = null; invalidateFrame();
  $('editor-message').textContent = 'เลือกเวลาแล้วโหลดเฟรม วิดีโอจะอัปโหลดเพียงครั้งเดียว';
  if (previous) api('/api/videos/' + previous, {method:'DELETE'}).catch(error => showError(error.message));
}
function setFrameTime(seconds, seek = false) {
  let value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) value = 0;
  if (duration > 0) value = Math.min(value, Math.max(0, duration - .05));
  $('frame-time').value = value.toFixed(2);
  $('frame-slider').value = value;
  if (frameId && Math.abs(value - frameSeconds) > .005) {
    invalidateFrame();
    $('editor-message').textContent = 'เวลาเปลี่ยนแล้ว กรุณาโหลดเฟรมใหม่และเลือกใบหน้า';
  }
  if (seek && Number.isFinite($('target-video').duration)) {
    $('target-video').pause(); $('target-video').currentTime = value;
  }
}
function setDuration(seconds) {
  duration = seconds;
  $('frame-time').max = $('frame-slider').max = Math.max(0, seconds - .05);
  $('video-duration').textContent = `ความยาว ${seconds.toFixed(2)} วินาที`;
  setFrameTime($('frame-time').value);
}
$('target-video').addEventListener('loadedmetadata', () => {
  if (Number.isFinite($('target-video').duration)) setDuration($('target-video').duration);
});
$('target-video').addEventListener('timeupdate', () => {
  if (!busy && isVideoTarget()) setFrameTime($('target-video').currentTime);
});
$('frame-time').addEventListener('change', () => setFrameTime($('frame-time').value, true));
$('frame-slider').addEventListener('input', () => setFrameTime($('frame-slider').value, true));
$('selection').addEventListener('change', () => {
  if ($('selection').value === 'all') $('face-picker').querySelectorAll('input').forEach(el => el.checked = true);
  clearSwappedPreview();
});
['mode', 'size'].forEach(id => $(id).addEventListener('change', clearSwappedPreview));
$('clear-video').addEventListener('click', clearVideoSession);

async function uploadVideo() {
  if (videoSession) return;
  const file = $('target').files[0];
  if (!file || !isVideoTarget()) throw new Error('กรุณาเลือกวิดีโอ');
  if (file.size > maxUploadMB * 1024**2) throw new Error(`วิดีโอต้องไม่เกิน ${formatLimit(maxUploadMB)}`);
  const form = new FormData(); form.append('target', file);
  const data = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/videos'); xhr.setRequestHeader('X-App-Token', token);
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) $('editor-message').textContent = `กำลังอัปโหลดวิดีโอ ${Math.round(event.loaded/event.total*100)}% · อัปโหลดครั้งเดียวเพื่อเลือกเฟรมต่อได้`;
    };
    xhr.onload = () => {
      let response;
      try { response = JSON.parse(xhr.responseText); } catch { response = {}; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(response);
      else reject(new Error(response.error || `อัปโหลดไม่สำเร็จ (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('การเชื่อมต่ออัปโหลดขัดข้อง กรุณาลองใหม่'));
    xhr.send(form);
  });
  videoSession = data.id; setDuration(data.duration);
}
$('inspect-frame').addEventListener('click', async () => {
  $('error').hidden = true; $('target-video').pause();
  setBusy(true, false);
  try {
    await uploadVideo();
    $('editor-message').textContent = 'กำลังดึงเฟรมและตรวจหาใบหน้า…';
    const data = await api(`/api/videos/${videoSession}/frame`, {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({
        seconds:Number($('frame-time').value), mode:$('mode').value, size:Number($('size').value)})});
    invalidateFrame(); frameId = data.frame_id; frameSeconds = data.seconds;
    $('frame-original').src = data.original;
    $('frame-label').textContent = `${data.frame_time.toFixed(2)} วินาที`;
    $('frame-backend').textContent = data.backend;
    $('frame-comparison').hidden = false;
    for (const face of data.faces) {
      const card = document.createElement('label'); card.className = 'face-card';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.value = face.id;
      checkbox.checked = $('selection').value === 'all';
      const image = document.createElement('img'); image.src = face.thumbnail; image.alt = `ใบหน้าคนที่ ${face.id+1}`;
      const text = document.createElement('span'); text.textContent = `คนที่ ${face.id+1}`;
      card.append(checkbox, image, text); $('face-picker').append(card);
      checkbox.addEventListener('change', () => { $('selection').value = 'selected'; clearSwappedPreview(); });
    }
    $('editor-message').textContent = data.faces.length
      ? `พบ ${data.faces.length} ใบหน้า · คลิกรูปคนที่ต้องการเปลี่ยน เลือกได้มากกว่าหนึ่งคน`
      : 'ไม่พบใบหน้าในเฟรมนี้ กรุณาเลื่อนไปยังช่วงที่เห็นคนที่ต้องการแล้วโหลดเฟรมใหม่';
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
});
$('swap-preview').addEventListener('click', async () => {
  const source = $('source').files[0];
  if (!source || !frameId) return showError('กรุณาเลือกรูปอ้างอิงและโหลดเฟรมก่อน');
  if (source.size > maxReferenceMB*1024**2) return showError(`รูปอ้างอิงต้องไม่เกิน ${formatLimit(maxReferenceMB)}`);
  const form = new FormData(); form.append('source', source); form.set('frame_id', frameId);
  form.set('face_ids', JSON.stringify(selectedFaces())); form.set('selection', $('selection').value);
  form.set('mode', $('mode').value); form.set('size', $('size').value);
  $('error').hidden = true; setBusy(true, false);
  $('editor-message').textContent = 'กำลังสร้างตัวอย่างหลังสลับหน้า…';
  try {
    const data = await api(`/api/videos/${videoSession}/preview`, {method:'POST', body:form});
    $('frame-swapped').src = data.result + '?v=' + Date.now(); $('frame-swapped').hidden = false;
    $('preview-placeholder').hidden = true;
    $('editor-message').textContent = `ตัวอย่างสลับแล้ว ${data.faces} ใบหน้า · หากพอใจ กดเริ่มสลับใบหน้าเพื่อทำทั้งคลิป`;
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
});
