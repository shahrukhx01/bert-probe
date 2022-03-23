from textattack.datasets import Dataset
import pandas as pd
import random


class GermanDataset:
    """
    Dataset wrapper for wrapping the dataset which will be perturbed during the
    attacks
    """

    def __init__(self, filepath, do_sampling=True, sample_size=10):
        self.filepath = filepath
        self.sample_size = sample_size
        self.do_sampling = do_sampling

    def load_dataset(self):
        import pathlib

        base_path = pathlib.Path(__file__).parent.parent.parent.resolve()
        data = pd.read_csv(f"{base_path}/{self.filepath}")
        dataset = []

        for _, row in data.iterrows():
            dataset.append((row.text, row.label))

        if self.do_sampling:
            dataset = random.sample(dataset, self.sample_size)

        custom_dataset = Dataset(dataset)

        return custom_dataset
