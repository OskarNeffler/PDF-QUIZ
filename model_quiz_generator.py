#!/usr/bin/env python3
"""
🤖 Model-Based Quiz Generator
Generate quiz questions from extracted PDF texts using trained T5 model
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ModelQuizGenerator:
    """Generate quiz questions using trained T5 model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_success = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained T5 model"""
        try:
            print(f"🤖 Loading model from: {model_path}")
            print(f"📱 Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            self.setup_success = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.setup_success = False
            return False
    
    def generate_questions_from_text(self, text: str, num_questions: int = 5, 
                                   max_input_length: int = 512, 
                                   max_output_length: int = 64) -> List[str]:
        """Generate questions from text using the T5 model"""
        if not self.setup_success:
            print("❌ Model not loaded")
            return []
        
        try:
            # Split text into chunks if it's too long
            chunks = self._split_text(text, max_input_length - 20)
            questions = []
            
            print(f"📝 Processing {len(chunks)} text chunks...")
            
            for i, chunk in enumerate(chunks[:num_questions]):
                print(f"   Processing chunk {i+1}/{min(len(chunks), num_questions)}")
                
                # Generate simple question first
                input_text = f"generate question: {chunk}"
                
                # Tokenize
                inputs = self.tokenizer(
                    input_text,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate questions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=4,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode generated questions
                generated_questions = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                
                for question in generated_questions:
                    if question.strip() and question not in questions:
                        # Convert to multiple choice format
                        mc_question = self._convert_to_multiple_choice(question.strip(), chunk)
                        questions.append(mc_question)
                
                if len(questions) >= num_questions:
                    break
            
            print(f"✅ Generated {len(questions)} multiple choice questions")
            return questions[:num_questions]
            
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return []
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit the model's input length"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _convert_to_multiple_choice(self, question: str, context: str) -> str:
        """Convert a simple question to multiple choice format"""
        import random
        
        # Extract key terms from context for creating alternatives
        words = context.split()
        key_terms = [word for word in words if len(word) > 4 and word.isalpha()]
        
        # Create plausible alternatives
        alternatives = []
        
        # Add some generic programming/ML alternatives
        generic_options = [
            "TensorFlow", "PyTorch", "Python", "JavaScript", "model.fit()", 
            "neural network", "deep learning", "machine learning", "API",
            "Sequential", "Functional", "Dense", "Convolutional", "LSTM"
        ]
        
        # Mix context-specific and generic alternatives
        all_options = key_terms + generic_options
        random.shuffle(all_options)
        
        # Take first 3 unique options for wrong answers
        wrong_answers = []
        for option in all_options:
            if len(wrong_answers) < 3 and option.lower() not in question.lower():
                wrong_answers.append(option)
        
        # Pad with generic if needed
        while len(wrong_answers) < 3:
            wrong_answers.append(f"Option {len(wrong_answers) + 1}")
        
        # Create correct answer (simplified - could be improved)
        if "what" in question.lower() and "name" in question.lower():
            correct_answer = "Keras"  # Context-specific
        elif "how" in question.lower():
            correct_answer = "Using specific methods"
        else:
            correct_answer = "As described in documentation"
        
        # Randomize order
        all_answers = [correct_answer] + wrong_answers[:3]
        random.shuffle(all_answers)
        
        # Find correct answer position
        correct_pos = all_answers.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_pos]
        
        # Format multiple choice question
        mc_question = f"{question}\n"
        for i, answer in enumerate(all_answers):
            letter = ['A', 'B', 'C', 'D'][i]
            mc_question += f"{letter}) {answer}\n"
        mc_question += f"Correct: {correct_letter}"
        
        return mc_question

def load_extracted_texts() -> Dict[str, str]:
    """Load all texts from extracted_texts directory"""
    extracted_texts_dir = Path("extracted_texts")
    texts = {}
    
    if not extracted_texts_dir.exists():
        print("❌ extracted_texts directory not found")
        return texts
    
    text_files = list(extracted_texts_dir.glob("*.txt"))
    
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            texts[text_file.stem] = content
            print(f"📖 Loaded {text_file.name}: {len(content)} characters")
        except Exception as e:
            print(f"❌ Error loading {text_file.name}: {e}")
    
    return texts

def main():
    print("🤖 Model-Based Quiz Generator")
    print("=" * 40)
    
    # Check for model path argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Look for common model directories
        possible_paths = [
            "question_generation_model",
            "question_generation_model_cpu", 
            "Models/squad_simple_cpu",
            "../question_generation_model"
        ]
        
        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            print("❌ No model found. Please provide model path:")
            print("Usage: python model_quiz_generator.py <model_path>")
            print("\nExpected model directories:")
            for path in possible_paths:
                print(f"  - {path}")
            return
    
    # Load extracted texts
    print("\n📚 Loading extracted texts...")
    texts = load_extracted_texts()
    
    if not texts:
        print("❌ No extracted texts found. Run PDF extraction first!")
        return
    
    # Initialize quiz generator
    print(f"\n🤖 Initializing model...")
    quiz_gen = ModelQuizGenerator(model_path)
    
    if not quiz_gen.setup_success:
        print("❌ Failed to load model")
        return
    
    # Generate questions for each text
    print(f"\n🔄 Generating quiz questions...")
    
    for text_name, text_content in texts.items():
        print(f"\n📄 Processing: {text_name}")
        print(f"   Text length: {len(text_content)} characters")
        
        questions = quiz_gen.generate_questions_from_text(
            text_content, 
            num_questions=5
        )
        
        if questions:
            print(f"\n✅ Generated {len(questions)} questions for {text_name}:")
            print("-" * 50)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            print("-" * 50)
            
            print(f"📋 Questions generated fresh from text (not saved)")
        else:
            print(f"❌ No questions generated for {text_name}")

if __name__ == "__main__":
    main() 