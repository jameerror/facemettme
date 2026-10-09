const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
let currentJob = null;
let timer = null;
let busy = false;
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
function setBusy(value) {
  busy = value;
  document.querySelectorAll('#swap-form input,#swap-form select,#start').forEach(el => el.disabled = value);
  $('cancel').hidden = !value;
}
function preview(name) {
  const file = $(name).files[0];
  if (!file) return;
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
  setBusy(true);
  try {
    const data = await api('/api/jobs', {method: 'POST', body: form});
    currentJob = data.id;
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
