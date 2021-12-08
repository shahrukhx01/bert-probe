from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data import GermanData


class GermanDataLoader:
    def __init__(self, data_paths, batch_size=8):
        self.german_data = GermanData(data_paths)
        self.batch_size = batch_size
        self.create_loaders()

    def create_loaders(self):
        """
        Create Torch dataloaders for data splits
        """
        self.german_data.text_to_tensors()
        print("creating dataloaders")
        train_data = TensorDataset(
            self.german_data.train_inputs,
            self.german_data.train_masks,
            self.german_data.train_labels,
        )
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size
        )

        validation_data = TensorDataset(
            self.german_data.validation_inputs,
            self.german_data.validation_masks,
            self.german_data.validation_labels,
        )
        validation_sampler = SequentialSampler(validation_data)
        self.validation_dataloader = DataLoader(
            validation_data, sampler=validation_sampler, batch_size=self.batch_size
        )

        test_data = TensorDataset(
            self.german_data.test_inputs,
            self.german_data.test_masks,
            self.german_data.test_labels,
        )
        test_sampler = SequentialSampler(test_data)
        self.test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=self.batch_size
        )
        print("finished creating dataloaders")


"""
** FOR DEBUGGING **

if __name__ == "__main__":
    ## define data paths
    germeval_data_paths = {
        "train": "./datasets/hasoc_dataset/hasoc_german_train.csv",
        "dev": "./datasets/hasoc_dataset/hasoc_german_validation.csv",
        "test": "./datasets/hasoc_dataset/hasoc_german_test.csv",
    }

    hasoc_german_data_paths = {
        "train": "./datasets/hasoc_dataset/hasoc_german_train.csv",
        "dev": "./datasets/hasoc_dataset/hasoc_german_validation.csv",
        "test": "./datasets/hasoc_dataset/hasoc_german_test.csv",
    }

    ## create dataloaders

    print("creating germeval dataloaders...")
    germ_eval_dataloader = GermanDataLoader(germeval_data_paths)

    print("creating hasoc dataloaders...")
    hasoc_german_dataloader = GermanDataLoader(hasoc_german_data_paths)

"""
