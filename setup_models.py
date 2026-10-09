"""Download models from the model author's official release, without bundling weights."""
import argparse
from pathlib import Path
import urllib.request
import os
from settings import MODEL_ROOT

ROOT = MODEL_ROOT
SWAP_URL = 'https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx'


def main():
    parser = argparse.ArgumentParser(description='Prepare InsightFace models locally')
    parser.add_argument('--swap-file', type=Path, help='Use an existing, licensed inswapper_128.onnx')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / 'inswapper_128.onnx'
    if args.swap_file:
        import shutil
        if not args.swap_file.is_file():
            parser.error('The supplied model file does not exist')
        if args.swap_file.resolve() != destination.resolve():
            shutil.copyfile(args.swap_file, destination)
    elif not destination.exists():
        partial = destination.with_suffix('.part')
        print('Downloading INSwapper from the official InsightFace release...', flush=True)
        try:
            with urllib.request.urlopen(SWAP_URL, timeout=60) as response, partial.open('wb') as output:
                import shutil
                shutil.copyfileobj(response, output)
            if partial.stat().st_size < 1_000_000:
                raise ValueError('Downloaded model is unexpectedly small')
            os.replace(partial, destination)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise SystemExit(f'Download unavailable: {exc}\nObtain a model from its author and run: python setup_models.py --swap-file PATH_TO_MODEL')
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    print('Downloading / validating buffalo_l...', flush=True)
    analyzer = FaceAnalysis(name='buffalo_l', root=str(ROOT),
                            allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = get_model(str(destination), providers=['CPUExecutionProvider'])
    if swapper is None:
        raise SystemExit('Invalid face swap model')
    print('Models are ready. Available providers: ' + ', '.join(ort.get_available_providers()))
    print('Pretrained InsightFace weights are for non-commercial research; consult their author for other licensing.')


if __name__ == '__main__':
    main()
