"""
DeepVision AI

Dataset Split Generator

Creates reproducible train/validation/test CSV files.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitGenerator:

    def __init__(
        self,
        input_csv="outputs/dataset_index.csv",
        output_dir="outputs/splits",
        random_state=42,
    ):

        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.random_state = random_state

    def generate(self):

        df = pd.read_csv(self.input_csv)

        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=self.random_state,
            stratify=df["label"],
            shuffle=True,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=self.random_state,
            stratify=temp_df["label"],
            shuffle=True,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(
            self.output_dir / "train.csv",
            index=False,
        )

        val_df.to_csv(
            self.output_dir / "val.csv",
            index=False,
        )

        test_df.to_csv(
            self.output_dir / "test.csv",
            index=False,
        )

        print("\nDeepVision AI Dataset Splits")
        print("=" * 40)

        print(f"Train : {len(train_df)}")
        print(f"Validation : {len(val_df)}")
        print("\nSaved to")

        print(self.output_dir)


def generate_splits(input_csv="outputs/dataset_index.csv", output_dir="outputs/splits", random_state=42):
    """
    Function helper to generate dataset splits.
    """
    generator = DatasetSplitGenerator(input_csv=input_csv, output_dir=output_dir, random_state=random_state)
    return generator.generate()