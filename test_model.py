import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

def test_model():
    print("Testing the question generation model...")
    
    try:
        # Load model and tokenizer from Hugging Face Hub (t5-small as a fallback)
        print("Loading model and tokenizer from Hugging Face Hub...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print(f"Model and tokenizer loaded successfully from {model_name}.")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Sample text
        context = "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions on data."
        
        # Prepare input
        input_text = f"generate question: {context}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate questions
        print("Generating questions...")
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            num_return_sequences=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and print the generated questions
        print("\nGenerated Questions:")
        for i, output in enumerate(outputs, 1):
            question = tokenizer.decode(output, skip_special_tokens=True)
            print(f"{i}. {question}")
        
        # Now let's try to load our local model
        print("\nNow trying to load our local fine-tuned model...")
        try:
            local_model_path = os.path.join("question_generation_model", "best_model")
            local_model = T5ForConditionalGeneration.from_pretrained(local_model_path)
            print("Successfully loaded local model!")
            
            # Generate questions with the local model
            local_model = local_model.to(device)
            local_outputs = local_model.generate(
                input_ids,
                max_length=64,
                num_beams=4,
                num_return_sequences=3,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Decode and print the generated questions
            print("\nGenerated Questions with local model:")
            for i, output in enumerate(local_outputs, 1):
                question = tokenizer.decode(output, skip_special_tokens=True)
                print(f"{i}. {question}")
                
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
        
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model() 