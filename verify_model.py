# save as verify_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the original and fine-tuned models
original_tokenizer = AutoTokenizer.from_pretrained("./original_model")
original_model = AutoModelForSequenceClassification.from_pretrained("./original_model")

fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")

# Set models to evaluation mode
original_model.eval()
fine_tuned_model.eval()

# Test examples
test_examples = [
    "Metal frame with rectangular shape partially covered in coral",
    "Round buoy with solar panel on top",
    "Natural rock formation with crevices housing small fish"
]

# Expected classes (for human reference)
expected_classes = ["man-made object", "round/spherical object", "natural formation"]

# Compare predictions
print("Comparing model predictions:")
print("-" * 50)

for i, example in enumerate(test_examples):
    # Process with original model
    inputs = original_tokenizer(example, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        original_output = original_model(**inputs)
    
    # The original model wasn't trained for this task, so predictions will be random
    original_pred = torch.argmax(original_output.logits, dim=1).item()
    
    # Process with fine-tuned model
    inputs = fine_tuned_tokenizer(example, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        fine_tuned_output = fine_tuned_model(**inputs)
    
    fine_tuned_pred = torch.argmax(fine_tuned_output.logits, dim=1).item()
    
    print(f"Example {i+1}: '{example}'")
    print(f"  Original model prediction: {original_pred}")
    print(f"  Fine-tuned model prediction: {fine_tuned_pred} ({expected_classes[fine_tuned_pred]})")
    print("-" * 50)

# Compare model outputs on same input to show differences in learned representations
test_text = "Cylindrical object partially buried in sand"
inputs = original_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    original_logits = original_model(**inputs).logits
    
inputs = fine_tuned_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    fine_tuned_logits = fine_tuned_model(**inputs).logits

print("Model output comparison for:", test_text)
print(f"Original model logits: {original_logits[0].tolist()}")
print(f"Fine-tuned model logits: {fine_tuned_logits[0].tolist()}")

# The difference in logits distribution demonstrates that the model has learned from the new data