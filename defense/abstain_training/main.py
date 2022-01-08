if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import logging

import torch.cuda
from transformers import AutoModelForSequenceClassification

from bert_finetuning.config import BertOptimConfig
from bert_finetuning.data_loader import GermanDataLoader
from bert_finetuning.eval import eval_model_classification_report
from bert_finetuning.train import train_model
from defense.abstain_training.config import DataPaths
from defense.abstain_training.data import GermanAdversarialData

logger = logging.getLogger(__name__)


def set_parser_arguments(parser):
    parser.add_argument(
        "--root-directory",
        help="The root directory of the datasets. Default: CWD",
    )

    parser.add_argument(
        "--dataset",
        choices=("hasoc", "germeval"),
        required=True,
        help="The dataset to use.",
    )

    parser.add_argument(
        "--model",
        default="deepset/gbert-base",
        help="The path to the model. "
             "The model is loaded via `transformers.AutoModelForSequenceClassification`. "
             "By default 'deepset/gbert-base'",
    )

    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="The number of epochs to train for. Default: 10",
    )

    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="The batch-size to use. Default: 8",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Whether to not use CUDA (e.g. for debugging). By default tries to use CUDA.",
    )

    parser.add_argument(
        "--model-output-directory",
        default="./model",
        help="Where to store model checkpoints. Defaults to './model'"
    )


def main(args):
    assert hasattr(args, "root_path")  # can be None
    assert args.dataset is not None
    assert args.model is not None
    assert args.epoch is not None
    assert args.batch_size is not None
    assert args.no_cuda is not None
    assert args.model_output_directory is not None

    epochs = args.epochs
    logger.info("Initiated training for %d epochs!", epochs)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA specified but not available. Switching to CPU...")

    data_path = DataPaths(root_path=args.root_directory)
    if args.dataset == "hasoc":
        data_path = data_path.HASOC
    elif args.dataset == "germeval":
        data_path = data_path.GERMEVAL
    else:
        raise NotImplementedError("Unknown dataset!")

    model_name = args.model
    logger.info("Loading datasets.")
    data_loaders = GermanDataLoader(
        data_path,
        model_name,
        do_cleansing=False,
        max_sequence_length=128,
        batch_size=8,
        dataset_cls=GermanAdversarialData,
    )
    logger.info("Done loading datasets.")

    # load model and optimizers etc.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    logger.info("Loaded model.")
    optim_config = BertOptimConfig(
        model=model, train_dataloader=data_loaders.train_dataloader, epochs=epochs
    )
    logger.info("Loaded optimizer.")

    # execute the training routine
    logger.info("Starting training...")
    model = train_model(
        model=model,
        optimizer=optim_config.optimizer,
        scheduler=optim_config.scheduler,
        train_dataloader=data_loaders.train_dataloader,
        validation_dataloader=data_loaders.validation_dataloader,
        epochs=epochs,
        device=device,
        model_name=model_name,
        save_model_as=args.model_output_directory,
    )

    logger.info("Training finished! Evaluating model on test set...")
    eval_model_classification_report(
        model=model,
        test_dataloader=data_loaders.test_dataloader,
        device=device,
    )

    logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_parser_arguments(parser)
    args = parser.parse_args()

    main(args)
