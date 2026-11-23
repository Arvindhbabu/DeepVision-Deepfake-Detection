import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from facenet_pytorch import MTCNN

# ✅ IMPORTANT: correct absolute import
from src.config import FRAMES_DIR, ALIGNED_DIR, INPUT_SIZE, NUM_WORKERS


# ------------------------------
# Initialize MTCNN
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=INPUT_SIZE,
    margin=0,
    keep_all=False,
    device=device,
    post_process=False,  # Return pixel intensities rather than standardized embeddings.
)

print(f"[INFO] Using device: {device}")


# ------------------------------
# Align faces for one video
# ------------------------------
def align_video_frames(video_frames_dir: Path, output_dir: Path):
    """
    Detect and align faces from all frames of a single video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(video_frames_dir.glob("*.jpg"))

    if len(frame_files) == 0:
        return

    for frame_path in frame_files:
        out_path = output_dir / frame_path.name

        # ✅ Skip already processed frames
        if out_path.exists():
            continue

        try:
            img = Image.open(frame_path).convert("RGB")
            face = mtcnn(img)

            if face is None:
                continue

            # Convert tensor to PIL image
            face = face.permute(1, 2, 0).cpu().numpy()
            # MTCNN(post_process=False) returns pixel intensities in [0, 255].
            face_rgb = face.clip(0, 255).round().astype("uint8")
            face_img = Image.fromarray(face_rgb, mode="RGB")

            face_img.save(out_path)

        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[WARNING] Could not align {frame_path}: {exc}")
            continue


# ------------------------------
# Process entire dataset (FF++ / Celeb-DF)
# ------------------------------
def process_dataset(dataset_name: str):
    input_root = Path(FRAMES_DIR) / dataset_name
    output_root = Path(ALIGNED_DIR) / dataset_name

    if not input_root.exists():
        print(f"[WARNING] Frames folder not found: {input_root}")
        return

    video_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    print(f"[INFO] Processing {len(video_dirs)} videos for dataset: {dataset_name}")

    for video_dir in tqdm(video_dirs, desc=f"Aligning faces ({dataset_name})"):
        out_dir = output_root / video_dir.name

        # Resume per frame: do not skip a partly processed video.
        align_video_frames(video_dir, out_dir)


# ------------------------------
# Main entry point
# ------------------------------
if __name__ == "__main__":
    print("[INFO] Starting face alignment using MTCNN")

    process_dataset("ffpp")
    process_dataset("celebdf")

    print("[INFO] Face alignment completed successfully")

