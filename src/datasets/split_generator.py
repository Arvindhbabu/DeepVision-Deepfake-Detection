"""Safe entry point for generating DeepVision dataset splits.

Actor and source-video identities must never span dataset partitions.
Use the validated manifest pipeline in scripts.validate_dataset instead of
ordinary row-level stratified splitting.
"""

from pathlib import Path


class DatasetSplitGenerator:
    def __init__(
        self,
        input_csv="outputs/dataset_index.csv",
        output_dir="outputs/splits",
        random_state=42,
    ):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.random_state = random_state

    def generate(self):
        raise RuntimeError(
            "Unsafe row-level splitting is disabled. Run "
            "'python scripts/validate_dataset.py --help' and use the "
            "validated actor-disjoint dataset splitting pipeline instead. "
            "The input must contain verified .npy sequences and actor metadata."
        )


def generate_splits(input_csv="outputs/dataset_index.csv", output_dir="outputs/splits", random_state=42):
    return DatasetSplitGenerator(input_csv, output_dir, random_state).generate()
