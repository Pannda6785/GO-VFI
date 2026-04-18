import torch
import argparse
import time

from core.pipeline import Pipeline
from core.device import resolve_device


def test_runtime(model_size="base", device_arg="auto"):
    device = resolve_device(device_arg)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    if model_size not in ("base", "large", "LARGE"):
        raise ValueError("model_size must be one of ('base', 'large', 'LARGE')")
    else:
        model_cfg_dict = dict(
                pyr_level=3,
                load_pretrain=False,
                device=device_arg,
                )

    ppl = Pipeline(model_cfg_dict)
    ppl.device()
    ppl.eval()

    img0 = torch.randn(1, 3, 480, 640)
    img0 = img0.to(device)
    img1 = torch.randn(1, 3, 480, 640)
    img1 = img1.to(device)

    with torch.no_grad():
        for i in range(100):
            _, _ = ppl.inference(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(100):
            _, _ = ppl.inference(img0, img1, 0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("average runtime: %4f seconds" % \
                ((time.time() - time_stamp) / 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="base")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    test_runtime(args.model_size, args.device)
