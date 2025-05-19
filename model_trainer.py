import os
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_squad_for_qg(model_name="t5-small", max_input_length=512, max_target_length=64):
    """
    Preprocess SQuAD dataset for question generation task.
    
    Args:
        model_name: Name of the T5 model to use
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        
    Returns:
        Processed dataset ready for training
    """
    logger.info("Loading SQuAD dataset...")
    # Load SQuAD (v1.1 by default, which includes only answerable questions)
    dataset = load_dataset("squad")
    
    # Optional: Print dataset statistics
    logger.info(f"Dataset loaded. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")
    
    # Initialize tokenizer
    logger.info(f"Initializing tokenizer: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Preprocess function for question generation
    def preprocess_function(examples):
        # Extract contexts and questions
        contexts = examples["context"]
        questions = examples["question"]
        
        # QG inputs: "generate question: {context}"
        inputs = ["generate question: " + context for context in contexts]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        targets = tokenizer(questions, max_length=max_target_length, truncation=True, padding="max_length")
        
        # Set the labels (targets)
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs
    
    # Apply preprocessing to the dataset
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset for question generation"
    )
    
    return processed_dataset, tokenizer


def train_qg_model(model_name="t5-small", output_dir="model_output", batch_size=8, learning_rate=5e-5, num_epochs=3):
    """
    Train a T5 model for question generation using SQuAD.
    
    Args:
        model_name: Name of the T5 model to use
        output_dir: Directory to save the trained model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
    """
    # Preprocess dataset
    processed_dataset, tokenizer = preprocess_squad_for_qg(model_name)
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        report_to="none"  # Disable wandb, tensorboard etc.
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Define metrics for evaluation
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode generated questions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple exact match metric
        exact_match = np.mean([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
        
        return {"exact_match": exact_match}
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the best model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


def test_question_generation(model, tokenizer, context, num_questions=3):
    """
    Generate questions from a given context using the trained model.
    
    Args:
        model: Trained question generation model
        tokenizer: Tokenizer for the model
        context: Text context to generate questions from
        num_questions: Number of questions to generate (using beam search)
    """
    # Prepare the input
    input_text = "generate question: " + context
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Generate multiple questions using beam search
    outputs = model.generate(
        input_ids,
        max_length=64,
        num_beams=num_questions * 2,
        num_return_sequences=num_questions,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the generated questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    print("\nGenerated Questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a question generation model using SQuAD")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Model name to use (e.g., t5-small, t5-base)")
    parser.add_argument("--output_dir", type=str, default="model_output", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--test", action="store_true", help="Test the model after training")
    args = parser.parse_args()
    
    # Train the model
    model, tokenizer = train_qg_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Test the model if requested
    if args.test:
        test_context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The Denver Broncos defeated the Carolina Panthers 24–10 to win their third Super Bowl championship."
        test_question_generation(model, tokenizer, test_context) 