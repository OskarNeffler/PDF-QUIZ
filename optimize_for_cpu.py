import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import shutil

def convert_model_for_cpu_inference(model_dir="question_generation_model", output_dir=None):
    """
    Convert the model for CPU inference by removing unnecessary training artifacts.
    
    Args:
        model_dir: Directory containing the model
        output_dir: Output directory for the CPU-optimized model (default: model_dir_cpu)
    """
    if output_dir is None:
        output_dir = f"{model_dir}_cpu"
    
    print(f"Converting model from {model_dir} to CPU-optimized version in {output_dir}")
    
    try:
        # Load the base T5 model first
        print("Loading base T5 model...")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Try to load from PyTorch checkpoint
        checkpoint_path = os.path.join(model_dir, "best_model")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            
            # Load the state dict from the checkpoint
            try:
                # If it's directly a state dict file
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Check if the checkpoint is a dictionary containing state_dict
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load the state dict into the model
                model.load_state_dict(state_dict)
                print("Successfully loaded state dict from checkpoint")
                
            except Exception as e:
                print(f"Error loading checkpoint directly, trying as a directory: {str(e)}")
                # If the checkpoint is a directory containing model files
                model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        
        else:
            print(f"Could not find best_model, trying checkpoint.pt...")
            checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Check if the checkpoint is a dictionary containing state_dict
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load the state dict into the model
                model.load_state_dict(state_dict)
                print("Successfully loaded state dict from checkpoint.pt")
            else:
                raise ValueError("No model checkpoints found in the specified directory")
        
        # Try loading tokenizer from the model directory
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        else:
            print("Tokenizer not found in model directory, loading from t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the model to evaluation mode and save without optimizer states
        print("Setting model to evaluation mode")
        model.eval()
        
        print(f"Saving CPU-optimized model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Add README file
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"""# T5 Question Generation Model (CPU version)

This model is optimized for CPU inference.

## Usage
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("{output_dir}")
tokenizer = T5Tokenizer.from_pretrained("{output_dir}")

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
        
        print(f"""
CPU model optimization complete.
The model is ready for inference on CPU in: {output_dir}

Usage with generate_questions.py:
python generate_questions.py --pdf your_document.pdf --model_path {output_dir} --count 5
""")
        
        return True
    
    except Exception as e:
        print(f"Error converting model for CPU inference: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a trained T5 model for CPU inference")
    parser.add_argument("--model_dir", default="question_generation_model", 
                        help="Directory containing the trained model")
    parser.add_argument("--output_dir", default=None, 
                        help="Output directory for the CPU-optimized model")
    
    args = parser.parse_args()
    
    convert_model_for_cpu_inference(args.model_dir, args.output_dir) 