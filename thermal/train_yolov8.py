import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project = ROOT / "runs" / "thermal"
    data = ROOT / "data" / "thermal" / "yolo" / "data.yaml"

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Use your RTX 3060 Ti "
            "desktop or specify --device cpu."
        )

    if args.resume:
        checkpoint = project / "yolov8n_full" / "weights" / "last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))
        model.train(resume=True)
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer="AdamW",
            lr0=0.001,
            patience=15,
            amp=True,
            save=True,
            save_period=1,
            project=str(project),
            name="yolov8n_full",
            exist_ok=True,
            seed=42,
            plots=True,
        )

    print("PASS: Thermal training run completed")

if __name__ == "__main__":
    main()
