import os
import sys
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_DIRECTORY = "./all-MiniLM-L6-v2-model"

def download_model():
    """
    Downloads a language model and its tokenizer from Hugging Face
    and saves them to a local directory.
    """
    print(f"Starting download of model: {MODEL_NAME}")
    
    # Check if the save directory already exists. If not, create it.
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: {SAVE_DIRECTORY}")

    try:
        # Step 1: Download the tokenizer.
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Step 2: Download the model itself.
        print("Downloading model... (this may take a moment)")
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        # Step 3: Save both the tokenizer and the model to our specified directory.
        print(f"Saving files to {SAVE_DIRECTORY}")
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        model.save_pretrained(SAVE_DIRECTORY)
        
        print("\nDownload and save complete!")
        print(f"Model files are now located in the '{SAVE_DIRECTORY}' folder.")

    except Exception as e:
        # If anything goes wrong (e.g., no internet connection), print an error.
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        print("Please check your internet connection and the model name.", file=sys.stderr)
        sys.exit(1) # Exit the script with an error code.

# This standard Python block ensures that the download_model() function
# is called only when you run the script directly from the command line.
if __name__ == "__main__":
    download_model()
