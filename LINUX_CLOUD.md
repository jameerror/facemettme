# Face Swap Me TT Me — Linux cloud

รองรับ Linux x86_64 เช่น Ubuntu 22.04/24.04, Python 3.12 และ Docker สำหรับ CPU หรือ NVIDIA GPU เปิดหน้าเว็บแล้วใช้งานได้ทันที ไม่ต้องล็อกอินหรือใส่รหัสผ่าน แอปประมวลผลทีละงานและใช้พื้นที่งานร่วมกัน เตรียมพื้นที่ดิสก์สำหรับโมเดลประมาณ 1 GB รวมทั้งไฟล์วิดีโอและไฟล์ชั่วคราวหลายเท่าของต้นฉบับ

## Docker CPU

ติดตั้ง Docker Engine และ Docker Compose plugin ตาม [คู่มือ Docker](https://docs.docker.com/engine/install/ubuntu/) แล้วคัดลอกโปรเจกต์ไปเซิร์ฟเวอร์ ไม่ต้องคัดลอก `.venv`, `data` หรือ `.env`

```bash
cp .env.example .env
```

ใช้ค่าเริ่มต้นใน `.env` ได้เลย หากใช้โดเมนให้ตั้งค่าตามหัวข้อ HTTPS ด้านล่าง

```bash
chmod 600 .env
docker compose build
docker compose run --rm app python setup_models.py
docker compose up -d app
docker compose logs -f app
```

โมเดลและผลลัพธ์เก็บใน named volumes คงอยู่เมื่อสร้างคอนเทนเนอร์ใหม่ โมเดลดาวน์โหลดตอน setup ไม่รวมเข้า Docker image หากดาวน์โหลดไม่ได้ ดูวิธีใช้ไฟล์โมเดลใน README

## เปิดจากเครื่องของคุณผ่าน SSH

Docker เปิดพอร์ตเฉพาะ loopback ของเซิร์ฟเวอร์ ใช้คำสั่งนี้บนคอมพิวเตอร์ของคุณ แทน `USER` และ `SERVER_IP` ด้วยค่าจริง:

```bash
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_IP
```

เปิด http://127.0.0.1:7860 แล้วใช้งานได้เลย วิธีนี้ไม่ต้องมีโดเมนหรือเปิดพอร์ตแอปให้สาธารณะ

## โดเมน HTTPS

ตั้ง DNS A record ให้ชี้ไปเซิร์ฟเวอร์ เปิด TCP 80/443 ที่ firewall และ cloud security group แล้วปรับ `.env`:

```dotenv
FACE_SWAP_DOMAIN=swap.your-domain.com
FACE_SWAP_TRUSTED_HOSTS=swap.your-domain.com,127.0.0.1,localhost
```

```bash
docker compose --profile https up -d
```

เปิด `https://swap.your-domain.com` แล้วใช้งานได้ทันที Caddy จัดการ HTTPS และส่งต่อให้แอปภายใน ไม่ต้องเปิดพอร์ต 7860 ต่อสาธารณะ หากผู้ให้บริการมี reverse proxy/HTTPS อยู่แล้ว ใช้ proxy นั้นแทน Caddy และเพิ่ม hostname จริงใน `FACE_SWAP_TRUSTED_HOSTS` โดยไม่ใส่ scheme หรือพอร์ต

## NVIDIA GPU

ติดตั้งไดรเวอร์ NVIDIA และ [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) บน host ให้ `nvidia-smi` ใช้ได้ แล้วใช้ไฟล์ GPU เพิ่มเติมทุกครั้ง:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app python setup_models.py
docker compose -f compose.yaml -f compose.gpu.yaml up -d app
# หากใช้ HTTPS เพิ่ม --profile https:
docker compose -f compose.yaml -f compose.gpu.yaml --profile https up -d
```

GPU image ใช้ `onnxruntime-gpu[cuda,cudnn]` เพื่อมีไลบรารี CUDA/cuDNN สำหรับ runtime มี CPU fallback ในโหมด Auto ดู [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) และ [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)

## Linux ที่ไม่ใช้ Docker

บน Ubuntu 24.04 ติดตั้งไลบรารีระบบแล้วรัน:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv libgl1 libglib2.0-0 libgomp1
bash install.sh cpu
# หรือ bash install.sh gpu สำหรับ NVIDIA
```

เปิดแอปผ่าน SSH tunnel แบบเดียวกับ Docker:

```bash
export FACE_SWAP_CLOUD=1
bash start.sh
```

หาก proxy ของคลาวด์ต้องการฟังทุก interface ให้ตั้งเพิ่มก่อนรัน:

```bash
export FACE_SWAP_HOST=0.0.0.0
export FACE_SWAP_TRUSTED_HOSTS=swap.your-domain.com,127.0.0.1,localhost
export PORT=7860
bash start.sh
```

ใช้ HTTPS proxy ก่อนเปิดให้อินเทอร์เน็ต ไม่เปิดเว็บเบราว์เซอร์บนเซิร์ฟเวอร์

## ตัวแปรตั้งค่า

| ตัวแปร | ค่าเริ่มต้น | ใช้ทำอะไร |
|---|---|---|
| `FACE_SWAP_HOST` | `127.0.0.1` | ที่อยู่รับการเชื่อมต่อ; Docker ตั้ง `0.0.0.0` |
| `PORT` | `7860` | พอร์ต; override ด้วย `--port` ได้ |
| `FACE_SWAP_CLOUD` | `0` | `1` แสดงข้อความว่าประมวลผลบนเซิร์ฟเวอร์ |
| `FACE_SWAP_TRUSTED_HOSTS` | loopback names | domain/IP คั่นด้วย comma |
| `FACE_SWAP_MODEL_DIR` | `models/` | โฟลเดอร์โมเดล; setup และแอปใช้ร่วมกัน |
| `FACE_SWAP_DATA_DIR` | `data/` | โฟลเดอร์ไฟล์งาน/ผลลัพธ์ |
| `FACE_SWAP_MAX_UPLOAD_MB` | `5120` | ขนาดไฟล์เป้าหมายสูงสุด; กำหนดได้ 1–10240 MiB |

เมื่อใช้ Compose ปรับขนาด upload ใน `.env` ตรวจขนาด upload และ timeout ของ proxy ด้วย การประมวลผลเป็นงานเบื้องหลัง ไม่ค้าง HTTP request จนวิดีโอเสร็จ

ค่าเริ่มต้นรองรับวิดีโอ 5 GB (5120 MiB) แยกจากรูปอ้างอิงสูงสุด 50 MB โดยเผื่อขนาด multipart อีก 1 MB ที่ Flask และ Waitress ชุด Caddy ที่ให้มารองรับการส่งต่อ request นี้ หากใช้ proxy ของผู้ให้บริการ ต้องให้รองรับ request อย่างน้อย 5171 MiB ด้วย หากเคยตั้ง `.env` เป็น `1024` ให้เปลี่ยน `FACE_SWAP_MAX_UPLOAD_MB=5120` แล้วสร้างคอนเทนเนอร์ใหม่ด้วย `docker compose up -d --build app`

ไฟล์อัปโหลดขนาดใหญ่ถูกพักบนดิสก์โดย Waitress/Werkzeug ไม่โหลดทั้งไฟล์ไว้ใน RAM สำหรับวิดีโอ 5 GB ควรมีพื้นที่ว่างอย่างน้อย 30 GB ทั้งพื้นที่ชั่วคราวและพื้นที่ไฟล์งาน เพราะมีต้นฉบับ วิดีโอที่แปลงแล้ว และไฟล์ผลลัพธ์ระหว่างประมวลผล

โหมดวิดีโออัปโหลดต้นฉบับครั้งเดียว จากนั้นเลือกเวลาของเฟรม ดูรูปใบหน้าที่ตรวจพบ และดูตัวอย่างก่อน–หลังได้บนเซิร์ฟเวอร์ เมื่อเริ่มงานทั้งคลิปจะใช้วิดีโอที่พักไว้และคนที่เลือกจากเฟรมนั้น ไม่ต้องส่งวิดีโอ 5 GB ซ้ำ หากไม่ทำต่อให้กด **ล้างวิดีโอที่อัปโหลด** เพื่อคืนพื้นที่ วิดีโอและเฟรมใน `data/editor/` อาจค้างเมื่อปิดหน้าเว็บหรือรีสตาร์ตแอปก่อนจบงาน

`GET /healthz` รายงานเฉพาะเซิร์ฟเวอร์ทำงาน ไม่มีรายละเอียดไฟล์หรือโมเดล หน้าเว็บ, static files, API และผลลัพธ์เข้าถึงได้โดยไม่ต้องล็อกอิน การอัปโหลด ยกเลิก และลบงานใช้ token ที่หน้าแอปจัดการอัตโนมัติ ไม่ต้องกรอกเอง

ไฟล์อัปโหลดถูกส่งไปยังเซิร์ฟเวอร์ที่คุณติดตั้ง ต้นฉบับถูกลบหลังงานจบ ผลลัพธ์อยู่ใน volume จนกดลบหรือจัดการเอง งานในหน่วยความจำหายหลังรีสตาร์ตแม้ไฟล์ผลลัพธ์ยังอยู่ รันแอปเพียง **หนึ่ง process/หนึ่ง replica** ไม่ใช้หลาย Gunicorn workers หรือหลาย replicas เพื่อให้คิวงานสอดคล้องกัน

## ตรวจการทำงาน

```bash
docker compose ps
docker compose exec app python -m unittest discover -s tests -v
docker compose exec app python -X utf8 tests/smoke_models.py
docker compose exec app python -X utf8 tests/smoke_cloud.py
```

`smoke_models.py` สร้างไฟล์ตัวอย่างและตรวจโมเดลจริง ส่วน `smoke_cloud.py` ใช้ไฟล์ตัวอย่างนั้นตรวจ HTTP server โดยไม่ล็อกอิน ตั้งแต่เปิดหน้าเว็บ อัปโหลดภาพ/วิดีโอ ประมวลผล ดาวน์โหลด และลบงานทดสอบ โมเดลมี license แยกจากโค้ด ดู README ก่อนใช้เชิงพาณิชย์
