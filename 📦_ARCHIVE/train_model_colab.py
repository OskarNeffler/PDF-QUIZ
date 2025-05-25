# -*- coding: utf-8 -*-
"""
# PDF-till-Quiz: T5 Model Training for Question Generation

This Colab notebook trains a T5 model for question generation from text passages.
Copy this code to Google Colab and run the cells in sequence.

## HOW TO USE THIS NOTEBOOK:
1. Create a new Colab notebook at https://colab.research.google.com
2. Copy each section (between ### SECTION X START ### and ### SECTION X END ###) to separate cells
3. Run the cells in order from top to bottom
4. To use hyperparameter tuning, uncomment the last line in Section 9

IMPORTANT: Make sure to select a GPU runtime in Colab:
Runtime -> Change runtime type -> Hardware accelerator -> GPU
"""

### SECTION 1 START: CONFIGURATION ###
# %%
# === CONFIGURATION ===
# Change these parameters to customize your training

# Select T5 model size
MODEL_NAME = "t5-small"  # Options: t5-small, t5-base, t5-large

# Output directory for the trained model
OUTPUT_DIR = "question_generation_model"

# Training configuration
BATCH_SIZE = 8          # Reduce if you get GPU memory errors
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01     # Weight decay regularization parameter
NUM_EPOCHS = 40         # Maximum training epochs
PATIENCE = 10           # Early stopping after this many epochs without improvement
SAVE_STEPS = 0.25       # Save and evaluate 4 times per epoch
WARMUP_RATIO = 0.1      # 10% of training steps used for warmup

# Data configuration
MAX_INPUT_LENGTH = 512  # Max length for context
MAX_TARGET_LENGTH = 64  # Max length for questions

# To limit training data during development (uncomment to activate)
# MAX_TRAIN_SAMPLES = 10000
# MAX_EVAL_SAMPLES = 1000
MAX_TRAIN_SAMPLES = None  # Use the entire dataset
MAX_EVAL_SAMPLES = None   # Use the entire validation set
### SECTION 1 END ###

### SECTION 2 START: CHECK GPU AVAILABILITY ###
# %% [markdown]
"""
## 1. Check GPU Availability

First, we'll check if a GPU is available to speed up training:
"""

# %%
import torch

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
    # Show GPU memory
    try:
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) // 1024 // 1024} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) // 1024 // 1024} MB")
    except:
        print("Could not retrieve GPU memory information")
else:
    print("Warning: No GPU found! Training will be very slow on CPU.")
    print("Make sure you've selected GPU runtime in Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU")
### SECTION 2 END ###

### SECTION 3 START: INSTALL PACKAGES ###
# %% [markdown]
"""
## 2. Install Required Packages

Now we'll install all necessary Python packages:
"""

# %%
# Install transformers, datasets and other necessary packages
!pip install transformers datasets nltk rouge-score sacrebleu matplotlib seaborn tensorboard

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
### SECTION 3 END ###

### SECTION 4 START: IMPORT LIBRARIES ###
# %% [markdown]
"""
## 3. Import Libraries

Load all necessary libraries and set up logging:
"""

# %%
import os
import sys
import time
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Transformers and datasets
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset, set_caching_enabled

# NLP libraries
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Enable dataset caching
set_caching_enabled(True)
### SECTION 4 END ###

### SECTION 5 START: DATA PREPROCESSING ###
# %% [markdown]
"""
## 4. Data Preprocessing

Now we'll load the SQuAD dataset and prepare it for training a question generation model.

For question generation, we reverse the usual SQuAD task:
- **Input**: Context (text passage)
- **Output**: Question

We use a special prefix ("generate question: ") to instruct the T5 model to generate a question.
"""

# %%
def load_and_preprocess_squad():
    """
    Load and preprocess the SQuAD dataset for question generation.
    """
    start_time = time.time()
    logger.info("Loading SQuAD dataset...")
    
    # Load SQuAD (v1.1 by default, which contains only answerable questions)
    dataset = load_dataset("squad")
    
    # Print dataset statistics
    logger.info(f"Dataset loaded. Training size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")
    
    # Limit the dataset if specified
    if MAX_TRAIN_SAMPLES is not None:
        logger.info(f"Limiting training data to {MAX_TRAIN_SAMPLES} examples")
        dataset['train'] = dataset['train'].select(range(min(MAX_TRAIN_SAMPLES, len(dataset['train']))))
    
    if MAX_EVAL_SAMPLES is not None:
        logger.info(f"Limiting validation data to {MAX_EVAL_SAMPLES} examples")
        dataset['validation'] = dataset['validation'].select(range(min(MAX_EVAL_SAMPLES, len(dataset['validation']))))
    
    # Initialize tokenizer
    logger.info(f"Initializing tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    
    # Preprocessing function for question generation
    def preprocess_function(examples):
        contexts = examples["context"]
        questions = examples["question"]
        
        # QG inputs: "generate question: {context}"
        # This instructs the T5 model to generate a question
        inputs = ["generate question: " + context for context in contexts]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True, 
            padding="max_length"
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                questions, 
                max_length=MAX_TARGET_LENGTH, 
                truncation=True, 
                padding="max_length"
            )
        
        # Set labels (targets)
        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding-token-id with -100 so these tokens are ignored in loss calculation
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label] 
            for label in model_inputs["labels"]
        ]
        
        return model_inputs
    
    # Apply preprocessing to the dataset
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset for question generation"
    )
    
    # Set format to PyTorch tensors
    processed_dataset.set_format(type="torch")
    
    end_time = time.time()
    logger.info(f"Preprocessing completed in {(end_time - start_time):.2f} seconds")
    
    return processed_dataset, tokenizer
### SECTION 5 END ###

### SECTION 6 START: EVALUATION METRICS ###
# %% [markdown]
"""
## 5. Evaluation Metrics

Define functions for calculating evaluation metrics for the generated questions.
We use:
- ROUGE-L: Measures the longest common subsequence between generated questions and reference questions
- Exact match: Proportion of questions that exactly match the reference questions
"""

# %%
def compute_metrics(eval_preds):
    """
    Calculate evaluation metrics for generated questions.
    """
    preds, labels = eval_preds
    
    # If we get a tuple from model.generate(), just take the predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode predictions and reference answers
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 with padding token ID (since -100 is ignored by loss)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Exact match (EM)
    exact_match = np.mean([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
    
    # ROUGE-L metric
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(label, pred)['rougeL'].fmeasure for pred, label in zip(decoded_preds, decoded_labels)]
    rouge_l = np.mean(rouge_scores)
    
    return {
        "exact_match": exact_match,
        "rouge_l": rouge_l
    }
### SECTION 6 END ###

### SECTION 7 START: TRAINING FUNCTION ###
# %% [markdown]
"""
## 6. Train the T5 Model

Now we'll configure and train the T5 model for question generation:
"""

# %%
def train_t5_for_question_generation():
    """
    Train the T5 model for question generation with early stopping.
    """
    # Load and prepare data
    processed_dataset, tokenizer = load_and_preprocess_squad()
    
    # Initialize the model
    logger.info(f"Initializing model: {MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Calculate training steps
    train_dataset_size = len(processed_dataset["train"])
    eval_dataset_size = len(processed_dataset["validation"])
    steps_per_epoch = train_dataset_size // BATCH_SIZE
    total_training_steps = steps_per_epoch * NUM_EPOCHS
    
    logger.info(f"Training data size: {train_dataset_size}")
    logger.info(f"Validation data size: {eval_dataset_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total number of steps: {total_training_steps}")
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch // 4,  # Evaluate 4 times per epoch
        save_strategy="steps",
        save_steps=steps_per_epoch // 4,  # Save 4 times per epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,  # Use the global WEIGHT_DECAY parameter
        save_total_limit=3,  # Save only the 3 best checkpoints
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",  # Optimize for ROUGE-L
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        warmup_ratio=WARMUP_RATIO,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=steps_per_epoch // 10,  # Log 10 times per epoch
        report_to="tensorboard"
    )
    
    # Data collator for seq2seq tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize Trainer with early stopping
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60  # in minutes
    logger.info(f"Training completed in {training_time:.2f} minutes")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training statistics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate the model one last time
    logger.info("Evaluating final model...")
    eval_metrics = trainer.evaluate(
        max_length=MAX_TARGET_LENGTH, 
        num_beams=4
    )
    
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Compile training metrics for visualization
    history = {"loss": [], "eval_loss": [], "eval_rouge_l": [], "eval_exact_match": []}
    log_history = trainer.state.log_history
    
    for log in log_history:
        if "loss" in log:
            history["loss"].append((log.get("step", 0), log["loss"]))
        if "eval_loss" in log:
            step = log.get("step", 0)
            history["eval_loss"].append((step, log["eval_loss"]))
            if "eval_rouge_l" in log:
                history["eval_rouge_l"].append((step, log["eval_rouge_l"]))
            if "eval_exact_match" in log:
                history["eval_exact_match"].append((step, log["eval_exact_match"]))
    
    return model, tokenizer, history
### SECTION 7 END ###

### SECTION 8 START: VISUALIZATION ###
# %% [markdown]
"""
## 7. Visualize Training Results

Function for visualizing the training progress with loss and metrics:
"""

# %%
def visualize_training_results(history):
    """
    Visualize training results with matplotlib.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 1, 1)
    
    # Training loss
    if history["loss"]:
        steps, values = zip(*history["loss"])
        plt.plot(steps, values, label='Training Loss', color='blue')
    
    # Validation loss
    if history["eval_loss"]:
        steps, values = zip(*history["eval_loss"])
        plt.plot(steps, values, label='Validation Loss', color='red')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Evaluation Metrics
    plt.subplot(2, 1, 2)
    
    # ROUGE-L
    if history["eval_rouge_l"]:
        steps, values = zip(*history["eval_rouge_l"])
        plt.plot(steps, values, label='ROUGE-L', color='green')
    
    # Exact match
    if history["eval_exact_match"]:
        steps, values = zip(*history["eval_exact_match"])
        plt.plot(steps, values, label='Exact Match', color='purple')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics During Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
    plt.show()
### SECTION 8 END ###

### SECTION 9 START: MODEL TESTING ###
# %% [markdown]
"""
## 8. Test the Model

Write a function to test the trained model on examples:
"""

# %%
def test_model_on_examples(model, tokenizer, examples=None):
    """
    Test the trained model on some examples.
    
    Args:
        model: Trained T5 model
        tokenizer: T5 tokenizer
        examples: List of example contexts to generate questions from
    """
    if examples is None:
        examples = [
            "Python is a high-level programming language known for its readability and simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "A database is an organized collection of structured information, or data, typically stored electronically in a computer system. A database is usually controlled by a database management system (DBMS).",
            "Git is a distributed version control system that tracks changes in any set of computer files, usually used for coordinating work among programmers collaboratively developing source code during software development."
        ]
    
    # Move the model to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    for i, context in enumerate(examples, 1):
        print(f"\nExample {i}: {context}")
        
        # Prepare input
        input_text = "generate question: " + context
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate multiple questions with beam search
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                num_return_sequences=3,  # Generate 3 different questions
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode the generated questions
        questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        print("Generated questions:")
        for j, question in enumerate(questions, 1):
            print(f"{j}. {question}")
### SECTION 9 END ###

### SECTION 10 START: RUN TRAINING ###
# %% [markdown]
"""
## 9. Run the Training and Visualize Results

Now we run the training and visualize the results:
"""

# %%
# Run the training
model, tokenizer, history = train_t5_for_question_generation()

# Visualize the training results
visualize_training_results(history)

# Test the model on some examples
test_model_on_examples(model, tokenizer)
### SECTION 10 END ###

### SECTION 11 START: MODEL DOWNLOAD ###
# %% [markdown]
"""
## 10. Package the Model for Download

To use the model locally in your PDF-to-Quiz application, we need to download the trained model from Colab:
"""

# %%
# Create a zip file with the trained model
!zip -r question_generation_model.zip {OUTPUT_DIR}

# Show download link
from google.colab import files
files.download('question_generation_model.zip')

print("""
Download Instructions:
1. Download the zip file above containing the trained model
2. Unzip it on your local computer
3. Use the model in PDF-to-Quiz by running:
   python start.py --pdf document.pdf --model_path ./question_generation_model --use_model
""")
### SECTION 11 END ###

### SECTION 12 START: CPU CONVERSION ###
# %% [markdown]
"""
## 11. Convert the Model for CPU Usage

If you need to run the model on CPU locally (without GPU), it can be useful to convert the model to a lighter format:
"""

# %%
def convert_model_for_cpu_inference():
    """
    Convert the model for CPU inference by removing unnecessary training artifacts.
    """
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR)
    tokenizer = T5Tokenizer.from_pretrained(OUTPUT_DIR)
    
    # CPU-friendly version
    cpu_output_dir = f"{OUTPUT_DIR}_cpu"
    os.makedirs(cpu_output_dir, exist_ok=True)
    
    # Set the model to evaluation mode and save
    model.eval()
    model.save_pretrained(cpu_output_dir, include_optimizer=False)
    tokenizer.save_pretrained(cpu_output_dir)
    
    # Add README file for CPU model
    with open(os.path.join(cpu_output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# T5 Question Generation Model for PDF-to-Quiz (CPU version)

This model is optimized for CPU inference.

## Usage
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("{cpu_output_dir}")
tokenizer = T5Tokenizer.from_pretrained("{cpu_output_dir}")

# Prepare input
context = "Your text here"
input_text = "generate question: " + context
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate a question
outputs = model.generate(input_ids, max_length=64)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(question)
```
""")
    
    # Create zip for download
    !zip -r {OUTPUT_DIR}_cpu.zip {cpu_output_dir}
    
    # Show download link for the CPU model
    files.download(f"{OUTPUT_DIR}_cpu.zip")
    
    print(f"""
CPU model is ready for download.
This model is optimized for inference on CPU.
    
Usage in PDF-to-Quiz:
python start.py --pdf document.pdf --model_path ./{cpu_output_dir} --use_model
""")

# Uncomment to convert the model for CPU usage
# convert_model_for_cpu_inference()
### SECTION 12 END ###

### SECTION 13 START: HYPERPARAMETER TUNING ###
# %% [markdown]
"""
## 12. Hyperparameter Tuning

In this section, we'll search for the optimal hyperparameters for our model by trying different random combinations of:
- Learning rates
- Batch sizes
- Weight decay values

We'll use a random search approach, evaluating each combination and tracking the best performance.
"""

# %%
def train_with_hyperparameters(learning_rate, batch_size, weight_decay, num_epochs=10):
    """
    Train the model with specific hyperparameters and return evaluation metrics
    
    Args:
        learning_rate: Learning rate to use
        batch_size: Batch size for training
        weight_decay: Weight decay regularization
        num_epochs: Number of epochs (reduced for hyperparameter search)
        
    Returns:
        Best evaluation metrics
    """
    # Create a specific output directory for this configuration
    hp_output_dir = f"{OUTPUT_DIR}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}"
    
    # Load and preprocess the dataset
    processed_dataset, tokenizer = load_and_preprocess_squad()
    
    # Initialize the model
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Calculate training steps
    steps_per_epoch = len(processed_dataset["train"]) // batch_size
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=hp_output_dir,
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch,  # Evaluate once per epoch
        save_strategy="steps",
        save_steps=steps_per_epoch,  # Save once per epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=1,  # Save only the best checkpoint
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(hp_output_dir, "logs"),
        logging_steps=steps_per_epoch // 2,
        report_to="none"  # Disable reporting to save resources
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Smaller patience for hyperparameter search
    )
    
    # Train the model
    logger.info(f"Training with lr={learning_rate}, bs={batch_size}, wd={weight_decay}")
    try:
        trainer.train()
        
        # Evaluate the model
        eval_metrics = trainer.evaluate(
            max_length=MAX_TARGET_LENGTH,
            num_beams=4
        )
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Clean up to save space
        import shutil
        if os.path.exists(hp_output_dir):
            shutil.rmtree(hp_output_dir)
            
        return eval_metrics
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {"eval_rouge_l": 0.0, "eval_exact_match": 0.0, "eval_loss": float('inf')}

# %%
def run_hyperparameter_tuning():
    """
    Perform a random search over hyperparameters to find the best configuration.
    Uses random values within ranges instead of a fixed grid.
    """
    # Define hyperparameter ranges - CUSTOMIZE THESE FOR YOUR NEEDS
    lr_range = (2e-5, 5e-4)  # Learning rate range (more focused for T5)
    batch_sizes = [4, 8, 16]  # Possible batch sizes
    wd_range = (0.01, 0.1)    # Weight decay range (higher values for more regularization)
    
    # Number of random configurations to try
    num_trials = 8  # Use 8 trials for faster results, increase to 12-15 for better tuning
    
    # Track results
    results = []
    
    # Store original output dir
    original_output_dir = OUTPUT_DIR
    
    # Run random search
    best_score = -float('inf')
    best_config = None
    
    logger.info(f"Starting random hyperparameter search with {num_trials} trials...")
    
    for trial in range(num_trials):
        # Generate random hyperparameters
        lr = np.exp(np.random.uniform(np.log(lr_range[0]), np.log(lr_range[1])))  # Log-uniform sampling
        bs = np.random.choice(batch_sizes)
        wd = np.random.uniform(wd_range[0], wd_range[1])
        
        try:
            # Skip configurations that would likely cause OOM errors
            if torch.cuda.is_available() and bs > 8 and MODEL_NAME in ["t5-large", "t5-3b", "t5-11b"]:
                logger.info(f"Skipping lr={lr:.6f}, bs={bs}, wd={wd:.6f} due to potential memory issues")
                # Generate new batch size and retry
                trial -= 1
                continue
            
            logger.info(f"Trial {trial+1}/{num_trials}: Testing lr={lr:.6f}, bs={bs}, wd={wd:.6f}")
            
            # Train and evaluate with this config
            metrics = train_with_hyperparameters(
                learning_rate=lr,
                batch_size=bs,
                weight_decay=wd,
                num_epochs=5  # Use fewer epochs for faster tuning
            )
            
            # Store results
            config_result = {
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "rouge_l": metrics.get("eval_rouge_l", 0),
                "exact_match": metrics.get("eval_exact_match", 0),
                "loss": metrics.get("eval_loss", float('inf'))
            }
            results.append(config_result)
            
            # Update best configuration
            if config_result["rouge_l"] > best_score:
                best_score = config_result["rouge_l"]
                best_config = config_result.copy()
            
            logger.info(f"Completed trial {trial+1}: ROUGE-L = {config_result['rouge_l']:.4f}")
            
        except Exception as e:
            logger.error(f"Error with config lr={lr:.6f}, bs={bs}, wd={wd:.6f}: {str(e)}")
            continue
    
    # Display results table
    print("\nHyperparameter Tuning Results:")
    print("-" * 80)
    print(f"{'Learning Rate':<15} {'Batch Size':<15} {'Weight Decay':<15} {'ROUGE-L':<15} {'Exact Match':<15}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x["rouge_l"], reverse=True):
        print(f"{result['learning_rate']:<15.6f} {result['batch_size']:<15} {result['weight_decay']:<15.4f} {result['rouge_l']:<15.4f} {result['exact_match']:<15.4f}")
    
    print("\nBest Configuration:")
    if best_config:
        print(f"Learning Rate: {best_config['learning_rate']:.6f}")
        print(f"Batch Size: {best_config['batch_size']}")
        print(f"Weight Decay: {best_config['weight_decay']:.4f}")
        print(f"ROUGE-L Score: {best_config['rouge_l']:.4f}")
        print(f"Exact Match: {best_config['exact_match']:.4f}")
        
        # Update global configuration with best parameters
        print("\nUpdating configuration with best parameters for final training...")
        global LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY
        LEARNING_RATE = best_config['learning_rate']
        BATCH_SIZE = best_config['batch_size']
        # Also update weight decay in the global configuration
        WEIGHT_DECAY = best_config['weight_decay']
        # Set output dir back to original
        global OUTPUT_DIR
        OUTPUT_DIR = original_output_dir
        
        return best_config
    else:
        print("No valid configurations found")
        return None
### SECTION 13 END ###

### SECTION 14 START: FULL PIPELINE ###
# %% [markdown]
"""
## 13. Full Training Pipeline with Optimal Hyperparameters

Now we'll train the final model using the optimal hyperparameters found through our random search.
"""

# %%
def run_full_pipeline_with_tuning():
    """
    Run the complete pipeline with hyperparameter tuning:
    1. Find optimal hyperparameters
    2. Train the final model with those parameters
    3. Visualize and test the model
    """
    print("Step 1: Hyperparameter Tuning")
    print("=" * 50)
    best_config = run_hyperparameter_tuning()
    
    if best_config:
        print("\nStep 2: Final Model Training")
        print("=" * 50)
        print(f"Training with optimal parameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, WD={best_config['weight_decay']}")
        
        # Train the final model
        model, tokenizer, history = train_t5_for_question_generation()
        
        print("\nStep 3: Visualizing Results")
        print("=" * 50)
        # Visualize training results
        visualize_training_results(history)
        
        print("\nStep 4: Testing Model")
        print("=" * 50)
        # Test the model
        test_model_on_examples(model, tokenizer)
        
        return model, tokenizer
    else:
        print("Hyperparameter tuning failed. Using default parameters.")
        model, tokenizer, history = train_t5_for_question_generation()
        visualize_training_results(history)
        test_model_on_examples(model, tokenizer)
        return model, tokenizer

# %%
# Run the full pipeline with hyperparameter tuning
# Uncomment to run the tuning process (takes several hours)
# final_model, final_tokenizer = run_full_pipeline_with_tuning()
### SECTION 14 END ###

### SECTION 15 START: INSTRUCTIONS ###
# %% [markdown]
"""
## Instructions for Running this Notebook

1. **Basic Training**: Run Sections 1-11 to train with default parameters
2. **CPU Export**: Run Section 12 to create a CPU-optimized version
3. **Hyperparameter Tuning**: Run Sections 13-14 for full hyperparameter tuning (warning: takes significant time)

Choose the approach that fits your needs:
- For quick results: Use the default parameters
- For best performance: Run the full hyperparameter tuning

Note that hyperparameter tuning can take several hours to complete.
"""
### SECTION 15 END ### 