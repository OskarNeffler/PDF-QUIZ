#!/usr/bin/env python3
"""
🧪 Simple Quiz Test
Test quiz generation using pre-trained T5 model
"""
import os
from pathlib import Path
from typing import List, Dict
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

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

def generate_simple_questions(text: str, num_questions: int = 5) -> List[str]:
    """Generate questions using pre-trained T5-small model"""
    try:
        print("🤖 Loading T5-small model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")
        
        # Load pre-trained T5 model
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Split text into chunks
        words = text.split()
        chunk_size = 100  # Smaller chunks for better questions
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Only use substantial chunks
                chunks.append(chunk)
        
        print(f"📝 Processing {len(chunks)} text chunks...")
        
        questions = []
        for i, chunk in enumerate(chunks[:num_questions]):
            print(f"   Processing chunk {i+1}/{min(len(chunks), num_questions)}")
            
            # Try different question generation prompts
            prompts = [
                f"generate question: {chunk}",
                f"question: {chunk}",
                f"ask about: {chunk}"
            ]
            
            for prompt in prompts:
                try:
                    inputs = tokenizer(
                        prompt,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=64,
                            num_beams=3,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            early_stopping=True,
                            do_sample=False
                        )
                    
                    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    for question in generated:
                        if question.strip() and question not in questions and "?" in question:
                            questions.append(question.strip())
                            break
                
                except Exception as e:
                    print(f"   Error with prompt: {e}")
                    continue
                
                if len(questions) >= num_questions:
                    break
            
            if len(questions) >= num_questions:
                break
        
        print(f"✅ Generated {len(questions)} questions")
        return questions
        
    except Exception as e:
        print(f"❌ Error in question generation: {e}")
        return []

def main():
    print("🧪 Simple Quiz Test with Pre-trained T5")
    print("=" * 40)
    
    # Load extracted texts
    print("\n📚 Loading extracted texts...")
    texts = load_extracted_texts()
    
    if not texts:
        print("❌ No extracted texts found. Run PDF extraction first!")
        return
    
    # Generate questions for each text
    print(f"\n🔄 Testing quiz generation...")
    
    for text_name, text_content in texts.items():
        print(f"\n📄 Processing: {text_name}")
        print(f"   Text length: {len(text_content)} characters")
        
        # Use only first 1000 characters for testing
        test_text = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
        
        questions = generate_simple_questions(test_text, num_questions=3)
        
        if questions:
            print(f"\n✅ Generated {len(questions)} test questions for {text_name}:")
            print("-" * 50)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            print("-" * 50)
        else:
            print(f"❌ No questions generated for {text_name}")
        
        print()  # Empty line between texts

if __name__ == "__main__":
    main() 