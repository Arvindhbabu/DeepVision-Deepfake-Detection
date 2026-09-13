"""
DeepVision AI
Dataset Builder

Scans FaceForensics++ and Celeb-DF datasets
and creates a unified metadata index.
"""

from pathlib import Path
from typing import List, Dict


class DatasetIndexer:
    """
    Creates a unified dataset index.

    Output Example:
    {
        "dataset": "FFPP",
        "label": 1,
        "category": "Deepfakes",
        "video_name": "000_003",
        "video_path": "data/raw/ffpp_c23/Deepfakes/000_003.mp4"
    }
    """

    VIDEO_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    def __init__(self, root_dir: str):

        self.root_dir = Path(root_dir)
        self.samples: List[Dict] = []

    def scan(self) -> List[Dict]:
        """
        Scan all datasets and create metadata list.
        """

        self.samples = []

        datasets = [

            (
                "FFPP",
                self.root_dir / "ffpp_c23",
                {
                    "original": 0,
                    "Deepfakes": 1,
                    "FaceSwap": 1,
                },
            ),

            (
                "CelebDF",
                self.root_dir / "celebdf_v2_sampled",
                {
                    "Celeb-real": 0,
                    "Celeb-synthesis": 1,
                },
            ),

        ]

        for dataset_name, dataset_path, mapping in datasets:

            if not dataset_path.exists():
                print(f"[WARNING] Dataset not found: {dataset_path}")
                continue

            print(f"\nScanning {dataset_name}...")

            for category, label in mapping.items():

                category_path = dataset_path / category

                if not category_path.exists():
                    print(f"[WARNING] Missing folder: {category_path}")
                    continue

                video_count = 0

                for video_file in category_path.iterdir():

                    if (
                        video_file.is_file()
                        and video_file.suffix.lower()
                        in self.VIDEO_EXTENSIONS
                    ):

                        self.samples.append(
                            {
                                "dataset": dataset_name,
                                "category": category,
                                "label": label,
                                "video_name": video_file.stem,
                                "video_path": str(video_file.resolve()),
                            }
                        )

                        video_count += 1

                print(
                    f"  {category:<18} : {video_count:5d} videos"
                )

        return self.samples

    def summary(self):
        """
        Print dataset statistics.
        """

        total_real = sum(
            sample["label"] == 0
            for sample in self.samples
        )

        total_fake = sum(
            sample["label"] == 1
            for sample in self.samples
        )

        ffpp = sum(
            sample["dataset"] == "FFPP"
            for sample in self.samples
        )

        celebdf = sum(
            sample["dataset"] == "CelebDF"
            for sample in self.samples
        )

        print("\n" + "=" * 60)
        print("DeepVision AI Dataset Summary")
        print("=" * 60)

        print(f"Total Samples : {len(self.samples)}")
        print(f"Real Videos   : {total_real}")
        print(f"Fake Videos   : {total_fake}")
        print(f"FFPP Videos   : {ffpp}")
        print(f"CelebDF Videos: {celebdf}")

        print("=" * 60)

    def save_index(self, output_path="outputs/dataset_index.csv"):
        """
        Save dataset metadata to CSV.
        """

        import pandas as pd

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.samples)

        df.to_csv(output_path, index=False)

        print(f"\nDataset index saved to: {output_path}")


if __name__ == "__main__":

    builder = DatasetIndexer("data/raw")

    builder.scan()

    builder.summary()