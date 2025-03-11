# Underwater Object Classification Using Fine-Tuned LLM

## Overview
This project fine-tunes a **BERT-based model** for classifying underwater objects based on textual descriptions. The model is trained on a **custom dataset** and evaluated for performance. The classification task assigns objects into one of three categories:

- **Man-made object** (Label: `0`)
- **Round/spherical object** (Label: `1`)
- **Natural formation** (Label: `2`)

## Project Flow

1. **Download a Pretrained Model** (`download_model.py`)
2. **Prepare Dataset** (`prepare_dataset.py`)
3. **Fine-tune the Model** (`finetune_model.py`)
4. **Verify the Model's Performance** (`verify_model.py`)
5. **Deploy a Classifier for Object Recognition** (`auv_object_classifier.py`)

---

## 1. Cloning the Repository & Setting Up the Environment

**Clone the Repository:**
```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

**Set Up Python Virtual Environment:**
```bash
python3 -m venv auv_llm_env
source auv_llm_env/bin/activate  # (For Windows: auv_llm_env\Scripts\activate)
pip install -r requirements.txt
```

---

## 2. Downloading a Pretrained Model
### `download_model.py`
This script downloads a **BERT-based model** (`bert-base-uncased`) and saves it locally.

**Steps:**
- Fetches the **tokenizer** and **model** from Hugging Face.
- Saves them in the `./original_model/` directory.

**Run Command:**
```bash
python download_model.py
```

---

## 3. Preparing the Dataset
### `prepare_dataset.py`
This script generates a sample dataset, splits it into training and testing sets, and converts it into **Hugging Face Dataset format**.

**Steps:**
- Defines a dataset with text descriptions and their respective labels.
- Splits the dataset into **train (80%)** and **test (20%)**.
- Saves the dataset in `./underwater_data/train/` and `./underwater_data/test/`.

**Run Command:**
```bash
python prepare_dataset.py
```

---

## 4. Fine-tuning the Model
### `finetune_model.py`
This script fine-tunes the BERT model on the prepared dataset using **Hugging Face's Trainer API**.

**Steps:**
- Loads the training and test datasets.
- Tokenizes the text using `AutoTokenizer`.
- Defines evaluation metrics (`accuracy`, `F1-score`).
- Configures the training settings (`TrainingArguments`).
- Trains the model and saves it as `./fine_tuned_model/`.

**Run Command:**
```bash
python finetune_model.py
```

---

## 5. Verifying the Model
### `verify_model.py`
This script compares the **original** and **fine-tuned** models by making predictions on test examples.

**Steps:**
- Loads both models (`original` and `fine-tuned`).
- Predicts categories for sample descriptions.
- Prints model outputs to compare logits and final classifications.

**Run Command:**
```bash
python verify_model.py
```

---

## 6. Deploying the Classifier
### `auv_object_classifier.py`
This script provides a **Python class** for classifying underwater objects using the fine-tuned model.

**Steps:**
- Loads the fine-tuned model and tokenizer.
- Defines a `classify()` function that predicts the object category.
- Runs test cases to classify new descriptions.

**Run Command:**
```bash
python auv_object_classifier.py
```

---

## Example Classification Output
```bash
Description: Square metallic box with antennas
Classification: man-made object (Confidence: 0.89)
--------------------------------------------------
Description: Smooth round object reflecting sonar signals
Classification: round/spherical object (Confidence: 0.93)
--------------------------------------------------
Description: Irregular formation with plant growth
Classification: natural formation (Confidence: 0.85)
```

---

## Troubleshooting
### If Results are Incorrect or Unstable:
1. **Check Dataset Quality:** Ensure training data is correctly labeled.
2. **Increase Training Epochs:** Modify `num_train_epochs` in `TrainingArguments`.
3. **Adjust Tokenization:** Experiment with different `max_length` settings.
4. **Inspect Model Outputs:** Run `verify_model.py` to compare logit differences.

---

## Requirements
Install dependencies using:
```bash
pip install transformers datasets torch scikit-learn evaluate pandas
```

---

## Summary
This project demonstrates how to **fine-tune** and **deploy** an NLP model for classifying underwater objects based on text descriptions. The fine-tuned model improves predictions over the pretrained model and can be integrated into an **AUV mission system**.

