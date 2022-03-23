from textattack.models.wrappers import ModelWrapper
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


class GermanHateSpeechModelWrapper(ModelWrapper):
    """
    Model wrapper for transformers BERT model for intregration with TextAttack
    for executing attacks on models
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
        model_predictions = self.model(
            text_input_list, pad_to_max_length=True, truncation=True
        )
        predictions = []
        for prediction in model_predictions:
            score = prediction["score"]
            if prediction["label"] == "LABEL_1":
                predictions.append([1 - score, score])
            else:
                predictions.append([score, 1 - score])
        return predictions
