from textattack.datasets import Dataset
import pandas as pd
import random


class GermanDataset:
    def __init__(self, filepath, do_sampling=True, sample_size=10):
        self.filepath = filepath
        self.sample_size = sample_size
        self.do_sampling = do_sampling

    def load_dataset(self):
        data = pd.read_csv(self.filepath)
        dataset = []

        for _, row in data.iterrows():
            dataset.append((row.text, row.label))

        if self.do_sampling:
            dataset = random.sample(dataset, self.sample_size)

        custom_dataset = Dataset(dataset)

        return custom_dataset
