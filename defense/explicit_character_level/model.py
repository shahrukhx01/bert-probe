import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


class GermanHateSpeechModel:
    """
    Model wrapper for transformers BERT model for intregration
    for executing defense on models
    """

    def __init__(self, model_name_path):
        self.load_model(model_name_path)

    def load_model(self, model_name_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_name_path).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        self.model = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    def __call__(self, text_input_list):
        return self.model(text_input_list)
