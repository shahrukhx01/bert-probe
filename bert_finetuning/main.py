import torch
from data_loader import GermanDataLoader
from model import BERTClassifier
from config import BertOptimConfig
from train import train_model
from eval import eval_model


if __name__ == "__main__":
    epochs = 10
    num_labels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = {
        "train": "./datasets/hasoc_dataset/hasoc_german_train.csv",
        "dev": "./datasets/hasoc_dataset/hasoc_german_validation.csv",
        "test": "./datasets/hasoc_dataset/hasoc_german_test.csv",
    }
    model_name = "deepset/gbert-base"
    data_loaders = GermanDataLoader(data_path, model_name, batch_size=8)
    model = BERTClassifier(num_labels=num_labels).get_model()
    optim_config = BertOptimConfig(
        model=model, train_dataloader=data_loaders.train_dataloader, epochs=epochs
    )

    ## execute the training routine
    model = train_model(
        model=model,
        optimizer=optim_config.optimizer,
        scheduler=optim_config.scheduler,
        train_dataloader=data_loaders.train_dataloader,
        validation_dataloader=data_loaders.validation_dataloader,
        epochs=epochs,
        device=device,
    )

    ## test model performance on unseen test set
    eval_model(model=model, test_dataloader=data_loaders.test_dataloader, device=device)
