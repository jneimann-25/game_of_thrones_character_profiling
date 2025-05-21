import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
import os

#note requires "pip install pandas torch transformers datasets accelerate" in terminal

# --- Configuration ---
DIALOGUES_PATH = "Game_of_Thrones_Main_Characters_Dialogue.csv" 
CHARACTER_NAME = "Jon Snow"
BASE_MODEL = "gpt2"  
OUTPUT_MODEL_DIR = f"./models/{CHARACTER_NAME.replace(' ', '_')}" 
NUM_TRAIN_EPOCHS = 3       
BATCH_SIZE = 4            
LEARNING_RATE = 5e-5       
SAVE_STEPS = 500           
LOGGING_STEPS = 100       
MAX_LENGTH = 128           

# --- 1. Load and Filter Data ---
print(f"Loading dialogues from: {DIALOGUES_PATH}")
try:
    df = pd.read_csv(DIALOGUES_PATH)
    print(f"Loaded {len(df)} total lines.")
except FileNotFoundError:
    print(f"ERROR: File not found at {DIALOGUES_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Filtering for character: {CHARACTER_NAME}")
jon_snow_dialogue = df[df['Name'] == CHARACTER_NAME]['Sentence'].tolist()

if not jon_snow_dialogue:
    print(f"ERROR: No dialogue found for {CHARACTER_NAME}. Check the 'Name' column in your CSV.")
    exit()

print(f"Found {len(jon_snow_dialogue)} lines for {CHARACTER_NAME}.")

# --- 2. Prepare Dataset ---
data_dict = {"text": [line + " <|endoftext|>" for line in jon_snow_dialogue]}
dataset = Dataset.from_dict(data_dict)
print("Created Hugging Face Dataset.")

# --- 3. Load Tokenizer and Model ---
print(f"Loading base model and tokenizer: {BASE_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Set padding token 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id 

except Exception as e:
    print(f"Error loading model or tokenizer '{BASE_MODEL}': {e}")
    print("Ensure the model name is correct and you have an internet connection.")
    exit()

# --- 4. Tokenize Data ---
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length", 
        max_length=MAX_LENGTH,
    )
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

print(f"Tokenizing dataset (max length: {MAX_LENGTH})...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,          
    remove_columns=["text"] 
)
print("Tokenization complete.")
print(f"Sample tokenized input: {tokenized_dataset[0]['input_ids'][:20]}...")
print(f"Sample labels: {tokenized_dataset[0]['labels'][:20]}...")

# --- 5. Set Up Training ---


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) 


os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_steps=SAVE_STEPS,
    save_total_limit=2,       
    logging_steps=LOGGING_STEPS,
    fp16=torch.cuda.is_available(), 
    report_to="none",          
)

print("Training arguments configured:")
print(training_args)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# --- 6. Run Fine-Tuning ---
print("\n--- Starting Fine-Tuning ---")
if torch.cuda.is_available():
    print("CUDA (GPU) is available. Training on GPU.")
else:
    print("WARNING: CUDA not available. Training on CPU will be very slow.")

try:
    trainer.train()
    print("\n--- Fine-Tuning Finished ---")
except Exception as e:
    print(f"\n--- Error during training: {e} ---")
    exit()


# --- 7. Save Final Model ---
print(f"Saving the final fine-tuned model to: {OUTPUT_MODEL_DIR}")
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR) 
print("Model and tokenizer saved successfully.")

print(f"\nFine-tuned model for {CHARACTER_NAME} is ready in '{OUTPUT_MODEL_DIR}'.")
print("You can now modify your chatbot's `initialize_generation_model` function")
print(f"to load from this path: '{OUTPUT_MODEL_DIR}'")

