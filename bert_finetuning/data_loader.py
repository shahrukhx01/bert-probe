from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data import QuestionsData


class QuestionsDataLoader:

    def __init__(self, data_file, batch_size=8):
        self.spam_data = QuestionsData(data_file)
        self.batch_size = batch_size
        self.create_loaders()
    
    def create_loaders(self):
        """
        Create Torch dataloaders for data splits
        """
        self.spam_data.text_to_tensors()
        print('creating dataloaders')
        train_data = TensorDataset(self.spam_data.train_inputs, 
                                    self.spam_data.train_masks, 
                                    self.spam_data.train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, 
                                            sampler=train_sampler, 
                                            batch_size=self.batch_size)

        validation_data = TensorDataset(self.spam_data.validation_inputs, 
                                        self.spam_data.validation_masks, 
                                        self.spam_data.validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        self.validation_dataloader = DataLoader(validation_data, 
                                                sampler=validation_sampler, 
                                                batch_size=self.batch_size)
        
        test_data = TensorDataset(self.spam_data.test_inputs, 
                                        self.spam_data.test_masks, 
                                        self.spam_data.test_labels)
        test_sampler = SequentialSampler(test_data)
        self.test_dataloader = DataLoader(test_data, 
                                                sampler=test_sampler, 
                                                batch_size=self.batch_size)
        print('finished creating dataloaders')
    
if __name__=='__main__':
    spam_loader = SpamDataLoader('spam.csv')
    spam_loader.create_loaders()