# save as auv_object_classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class UnderwaterObjectClassifier:
    def __init__(self, model_path="./fine_tuned_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.classes = ["man-made object", "round/spherical object", "natural formation"]
    
    def classify(self, description):
        """Classify an underwater object based on its text description"""
        inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get prediction
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
        
        return {
            "class_id": prediction,
            "class_name": self.classes[prediction],
            "confidence": confidence
        }

# Test the classifier
if __name__ == "__main__":
    classifier = UnderwaterObjectClassifier()
    
    # Test with some examples
    test_descriptions = [
        "Square metallic box with antennas",
        "Smooth round object reflecting sonar signals",
        "Irregular formation with plant growth",
        "Long cylindrical pipe with rust marks",
        "Dome-shaped object with camera lens"
    ]
    
    for description in test_descriptions:
        result = classifier.classify(description)
        print(f"Description: {description}")
        print(f"Classification: {result['class_name']} (Confidence: {result['confidence']:.2f})")
        print("-" * 50)