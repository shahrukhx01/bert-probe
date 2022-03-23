from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("./model")
model.push_to_hub("<model-name>")

tokenizer = AutoTokenizer.from_pretrained("./model")
tokenizer.push_to_hub("<model-name>")
