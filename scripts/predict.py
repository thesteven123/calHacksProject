"""Run the tiered funnel on images/videos and print JSON verdicts.

    python scripts/predict.py --checkpoint runs/smoke/best.pt path/to/img.jpg path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import json

from dfd.env import load_dotenv
from dfd.inference.pipeline import DetectionPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--device", default="auto")
    # Default stays "center": on Tiny-GenImage under platform compression, multi-crop did not improve
    # TPR@1%FPR and costs ~7-9x compute (docs/implementation-architecture.md §11). Use "multi" for
    # large media where off-centre/localized edits matter, with calibration_multicrop.json present.
    ap.add_argument("--crop-mode", choices=["center", "multi"], default="center")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    load_dotenv()
    pipe = DetectionPipeline(args.checkpoint, args.calibration, args.device, crop_mode=args.crop_mode)
    for p in args.paths:
        print(json.dumps(pipe.predict(p).to_dict()))


if __name__ == "__main__":
    main()
