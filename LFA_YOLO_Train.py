import os
import shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch
import gc

# --------------------------
# CONFIG
# --------------------------
DATA_YAML = "C:/Users/Dhaksha Aniesh/Desktop/YOLO/Dataset Final/data.yaml"
EPOCHS = 100
IMG_SIZE = 1024
OUTPUT_DIR = "C:/Users/Dhaksha Aniesh/Desktop/YOLO/Output/detect3"

HYP_YAML = "C:/Users/Dhaksha Aniesh/Desktop/YOLO/Dataset Final/hyp.yaml"
USE_HYP = os.path.exists(HYP_YAML)


# --------------------------
# MEMORY + CACHE CLEANUP
# --------------------------
def cleanup_memory():
    """Free GPU VRAM, clear Python garbage, Ultralytics cache, and Windows standby list if possible."""
    print("\n🧹 Cleaning memory and caches...")

    # Clear Python garbage
    gc.collect()

    # Clear CUDA VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("  ✅ CUDA cache cleared")

    # Clear Ultralytics cache
    ucache = Path.home() / "AppData" / "Local" / "Ultralytics"
    if ucache.exists():
        try:
            shutil.rmtree(ucache)
            print(f"  ✅ Deleted YOLO cache at {ucache}")
        except Exception as e:
            print(f"  ⚠️ Could not delete YOLO cache: {e}")

    # Clear __pycache__ folders
    for root, dirs, _ in os.walk(Path.cwd()):
        for d in dirs:
            if d == "__pycache__":
                try:
                    shutil.rmtree(Path(root) / d)
                    print(f"  ✅ Removed {Path(root) / d}")
                except Exception:
                    pass

    # Clear Windows standby memory using RAMMap if available
    rammap_paths = [
        Path("C:/Windows/System32/RAMMap.exe"),
        Path.cwd() / "RAMMap.exe"
    ]
    rammap = next((p for p in rammap_paths if p.exists()), None)

    if rammap:
        try:
            subprocess.run([str(rammap), "-E"], check=True)
            print("  ✅ Flushed Windows standby memory via RAMMap")
        except Exception as e:
            print(f"  ⚠️ Failed to flush standby memory: {e}")
    else:
        print("  ⚠️ RAMMap.exe not found (standby list not cleared)")


# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_models():
    variants = ["yolov8n.pt"]  # can add yolov8s.pt, yolov8m.pt if needed
    trained_models = []

    for variant in variants:
        print(f"\n🚀 Training {variant} ...")

        cleanup_memory()  # free RAM + VRAM before training

        model = YOLO(variant)

        train_args = dict(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            project=OUTPUT_DIR,
            name=f"exp_{variant.replace('.pt','')}",
            exist_ok=True,
            batch=2,           # ✅ safe for 4GB VRAM
            workers=2,         # ✅ reduces CPU RAM usage
            device=0,          # Force GPU
            verbose=False,     # ✅ prevents log spam
            mosaic=0.5,
            close_mosaic=15,
            mixup=0.0,
            degrees=0.0,
            shear=0.0,
            perspective=0.0,
            scale=0.2,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.3,
        )

        if USE_HYP:
            train_args["hyp"] = HYP_YAML

        # Train the model
        model.train(**train_args)

        best_model = Path(OUTPUT_DIR) / f"exp_{variant.replace('.pt','')}" / "weights" / "best.pt"
        trained_models.append(str(best_model))

    return trained_models

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LFA YOLO Training with Cleanup")
    parser.add_argument("--mode", type=str, required=True, choices=["train"])
    args = parser.parse_args()

    if args.mode == "train":
        trained = train_models()
        print("\n✅ Training complete. Trained models:", trained)
