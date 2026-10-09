"""Deployment settings shared by local, Linux and container launches."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get('FACE_SWAP_MODEL_DIR', ROOT / 'models')).expanduser().resolve()
DATA_ROOT = Path(os.environ.get('FACE_SWAP_DATA_DIR', ROOT / 'data')).expanduser().resolve()
REFERENCE_MAX_MB = 50
MULTIPART_OVERHEAD_MB = 1


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    cloud: bool
    trusted_hosts: tuple
    max_upload_mb: int

    @property
    def max_request_bytes(self):
        # A full-size target must still fit alongside its reference and form headers.
        return (self.max_upload_mb + REFERENCE_MAX_MB + MULTIPART_OVERHEAD_MB) * 1024**2


def load_settings(host=None, port=None):
    host = host or os.environ.get('FACE_SWAP_HOST', '127.0.0.1')
    port = int(port if port is not None else os.environ.get('PORT', '7860'))
    cloud = os.environ.get('FACE_SWAP_CLOUD', '0') == '1' or host not in {'127.0.0.1', 'localhost', '::1'}
    trusted_hosts = tuple(item.strip() for item in os.environ.get('FACE_SWAP_TRUSTED_HOSTS', '127.0.0.1,localhost,[::1]').split(',') if item.strip())
    max_upload_mb = int(os.environ.get('FACE_SWAP_MAX_UPLOAD_MB', '5120'))
    if not 1 <= port <= 65535 or not 1 <= max_upload_mb <= 10240:
        raise ValueError('Invalid PORT or FACE_SWAP_MAX_UPLOAD_MB (1–10240)')
    if not trusted_hosts or '*' in trusted_hosts:
        raise ValueError('Set FACE_SWAP_TRUSTED_HOSTS to your domain/IP without scheme or port')
    return ServerSettings(host, port, cloud, trusted_hosts, max_upload_mb)
