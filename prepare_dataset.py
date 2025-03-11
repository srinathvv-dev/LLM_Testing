# save as prepare_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Sample underwater object detection data (in real scenario, use your actual data)
data = {
    'text': [
        "Cylindrical metal object on the seabed with rust marks",
        "Smooth rounded rock with algae growth on western side",
        "Rectangular container partially buried in sand",
        "Round object with antenna-like protrusions",
        "Long pipe segment with marine growth",
        "Spherical object with smooth surface reflecting sonar",
        "Coral formation with branching structure",
        "Square metal plate with mounting holes",
        "Curved hull fragment with peeling paint",
        "Elongated torpedo-like shape with fins",
        "Cuboid box with open lid on sandy bottom",
        "Triangular structure with sharp edges",
        "Disk-shaped object half-covered by sediment",
        "Bottle with narrow neck lying horizontally",
        "Propeller with three blades and center hub",
        "Anchor with chain attached to one end",
        "Concrete block with embedded metal rings",
        "Tire with visible tread pattern",
        "Fishing net tangled around cylindrical object",
        "Camera housing with glass dome front"
    ],
    'label': [
        0, 2, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    ]
}

# 0: man-made object, 1: round/spherical object, 2: natural formation

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Save datasets
train_dataset.save_to_disk("./underwater_data/train")
test_dataset.save_to_disk("./underwater_data/test")

print("Dataset prepared and saved.")
print(f"Training examples: {len(train_dataset)}")
print(f"Testing examples: {len(test_dataset)}")