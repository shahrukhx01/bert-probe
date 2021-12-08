from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("./model")
model.push_to_hub("buy-sell-intent-classifier-bert-mini")

tokenizer = AutoTokenizer.from_pretrained("./model")
tokenizer.push_to_hub("buy-sell-intent-classifier-bert-mini")
