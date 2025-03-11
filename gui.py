import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import subprocess
import threading
import os

# Global variables
MODEL_PATH = "./original_model"
DATASET_PATH = "./underwater_data"
FINE_TUNED_PATH = "./fine_tuned_model"

# Function to download the model
def download_model():
    def run_download():
        output_text.insert(tk.END, "Downloading model...\n")
        try:
            result = subprocess.run(
                ["python", "download_model.py"],
                capture_output=True,
                text=True
            )
            output_text.insert(tk.END, result.stdout)
            output_text.insert(tk.END, result.stderr)
            output_text.insert(tk.END, "Model downloaded successfully!\n")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=run_download).start()

# Function to prepare the dataset
def prepare_dataset():
    def run_prepare():
        output_text.insert(tk.END, "Preparing dataset...\n")
        try:
            result = subprocess.run(
                ["python", "prepare_dataset.py"],
                capture_output=True,
                text=True
            )
            output_text.insert(tk.END, result.stdout)
            output_text.insert(tk.END, result.stderr)
            output_text.insert(tk.END, "Dataset prepared successfully!\n")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=run_prepare).start()

# Function to fine-tune the model
def fine_tune_model():
    def run_fine_tune():
        output_text.insert(tk.END, "Fine-tuning model...\n")
        try:
            result = subprocess.run(
                ["python", "finetune_model.py"],
                capture_output=True,
                text=True
            )
            output_text.insert(tk.END, result.stdout)
            output_text.insert(tk.END, result.stderr)
            output_text.insert(tk.END, "Model fine-tuned successfully!\n")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=run_fine_tune).start()

# Function to verify the model
def verify_model():
    def run_verify():
        output_text.insert(tk.END, "Verifying model...\n")
        try:
            result = subprocess.run(
                ["python", "verify_model.py"],
                capture_output=True,
                text=True
            )
            output_text.insert(tk.END, result.stdout)
            output_text.insert(tk.END, result.stderr)
            output_text.insert(tk.END, "Model verification complete!\n")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=run_verify).start()

# Function to classify using the fine-tuned model
def classify_object():
    description = classification_input.get("1.0", tk.END).strip()
    if not description:
        messagebox.showwarning("Input Error", "Please enter a description.")
        return

    def run_classify():
        output_text.insert(tk.END, f"Classifying: {description}\n")
        try:
            result = subprocess.run(
                ["python", "auv_object_classifier.py"],
                input=description,
                capture_output=True,
                text=True
            )
            output_text.insert(tk.END, result.stdout)
            output_text.insert(tk.END, result.stderr)
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=run_classify).start()

# Create the Tkinter GUI
root = tk.Tk()
root.title("Underwater Object Classification Pipeline")
root.geometry("800x600")

# Output console
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20)
output_text.pack(pady=10)

# Buttons for each step
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

download_button = tk.Button(button_frame, text="1. Download Model", command=download_model)
download_button.grid(row=0, column=0, padx=5)

prepare_button = tk.Button(button_frame, text="2. Prepare Dataset", command=prepare_dataset)
prepare_button.grid(row=0, column=1, padx=5)

fine_tune_button = tk.Button(button_frame, text="3. Fine-Tune Model", command=fine_tune_model)
fine_tune_button.grid(row=0, column=2, padx=5)

verify_button = tk.Button(button_frame, text="4. Verify Model", command=verify_model)
verify_button.grid(row=0, column=3, padx=5)

# Classification input
classification_frame = tk.Frame(root)
classification_frame.pack(pady=10)

tk.Label(classification_frame, text="Enter object description:").grid(row=0, column=0, padx=5)
classification_input = tk.Text(classification_frame, width=50, height=3)
classification_input.grid(row=0, column=1, padx=5)

classify_button = tk.Button(classification_frame, text="Classify", command=classify_object)
classify_button.grid(row=0, column=2, padx=5)

# Run the Tkinter event loop
root.mainloop()