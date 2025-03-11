# save as download_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the model to download
model_name = "bert-base-uncased"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                          num_labels=3)  # Adjust for your task

# Save the model and tokenizer locally
model.save_pretrained("./original_model")
tokenizer.save_pretrained("./original_model")

print(f"Model {model_name} downloaded and saved locally.")