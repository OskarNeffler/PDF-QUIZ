#!/usr/bin/env python3
"""
🎯 Enhanced PDF-to-Quiz System
A simplified version that focuses on what actually works
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import random
import fitz  # PyMuPDF for direct PDF extraction

class EnhancedQuizSystem:
    """Simple, effective quiz generation system"""
    
    def __init__(self):
        self.extracted_texts = {}
        self.load_available_texts()
    
    def load_available_texts(self):
        """Load all available extracted texts"""
        texts_dir = Path("extracted_texts")
        if texts_dir.exists():
            for txt_file in texts_dir.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Clean up the text
                        content = self.clean_text(content)
                        self.extracted_texts[txt_file.stem] = content
                        print(f"📖 Loaded {txt_file.name}: {len(content):,} characters")
                except Exception as e:
                    print(f"❌ Error loading {txt_file}: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and prepare text for quiz generation"""
        # Remove page headers/footers
        text = re.sub(r'--- Page \d+ ---\\n', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\\n\\s*\\n', '\\n\\n', text)
        # Remove very short lines (likely headers/footers)
        lines = text.split('\\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 10]
        return '\\n'.join(cleaned_lines)
    
    def extract_pdf_directly(self, pdf_path: str, save_as: str = None) -> str:
        """Extract text directly from PDF without AI"""
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            
            print(f"📖 Extracting from {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    all_text.append(text)
                    print(f"   Page {page_num + 1}: {len(text):,} chars")
            
            doc.close()
            combined_text = '\\n\\n'.join(all_text)
            
            # Save if requested
            if save_as:
                self.save_text(combined_text, save_as)
                self.extracted_texts[save_as] = self.clean_text(combined_text)
            
            print(f"✅ Extracted {len(combined_text):,} characters")
            return combined_text
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    
    def save_text(self, text: str, filename: str):
        """Save text to extracted_texts directory"""
        Path("extracted_texts").mkdir(exist_ok=True)
        filepath = Path("extracted_texts") / f"{filename}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"💾 Saved to {filepath}")
    
    def generate_simple_questions(self, text_name: str, num_questions: int = 10) -> List[Dict]:
        """Generate simple, effective questions from text"""
        if text_name not in self.extracted_texts:
            print(f"❌ Text '{text_name}' not found")
            return []
        
        text = self.extracted_texts[text_name]
        questions = []
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Filter good sentences for questions
        good_sentences = [s for s in sentences if self.is_good_sentence(s)]
        
        if len(good_sentences) < num_questions:
            print(f"⚠️  Only {len(good_sentences)} good sentences found")
            num_questions = len(good_sentences)
        
        # Randomly select sentences
        selected = random.sample(good_sentences, num_questions)
        
        for i, sentence in enumerate(selected, 1):
            question = self.create_question_from_sentence(sentence, i)
            if question:
                questions.append(question)
        
        print(f"✅ Generated {len(questions)} questions from {text_name}")
        return questions
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def is_good_sentence(self, sentence: str) -> bool:
        """Check if sentence is good for questions"""
        # Filter criteria
        if len(sentence) < 30 or len(sentence) > 200:
            return False
        if sentence.count(',') > 5:  # Too complex
            return False
        if any(word in sentence.lower() for word in ['figure', 'table', 'equation', 'ref']):
            return False
        if sentence.count('(') != sentence.count(')'):  # Unbalanced brackets
            return False
        
        return True
    
    def create_question_from_sentence(self, sentence: str, question_num: int) -> Optional[Dict]:
        """Create a multiple choice question from a sentence"""
        try:
            # Find key terms in the sentence
            words = sentence.split()
            
            # Look for potential answers (nouns, technical terms)
            candidates = []
            for word in words:
                if (len(word) > 4 and 
                    word[0].isupper() and 
                    not word.lower() in ['this', 'that', 'these', 'those', 'they', 'them']):
                    candidates.append(word.strip('.,!?;:'))
            
            if not candidates:
                return None
            
            # Select the answer
            answer = random.choice(candidates)
            
            # Create the question by replacing the answer with a blank
            question_text = sentence.replace(answer, "______", 1)
            
            # Generate wrong answers
            wrong_answers = self.generate_wrong_answers(answer)
            
            # Create choices
            choices = [answer] + wrong_answers[:3]  # 4 choices total
            random.shuffle(choices)
            
            correct_index = choices.index(answer)
            
            return {
                "question_number": question_num,
                "question": f"Complete the sentence: {question_text}",
                "choices": choices,
                "correct_answer": correct_index,
                "explanation": f"From the text: '{sentence}'"
            }
            
        except Exception as e:
            print(f"⚠️  Error creating question: {e}")
            return None
    
    def generate_wrong_answers(self, correct_answer: str) -> List[str]:
        """Generate plausible wrong answers"""
        # Simple wrong answer generation
        wrong_answers = []
        
        # Common technical terms that could be distractors
        common_terms = [
            "algorithm", "model", "system", "method", "approach", "technique",
            "neural", "attention", "transformer", "encoder", "decoder", "layer",
            "training", "learning", "optimization", "performance", "accuracy",
            "network", "architecture", "mechanism", "computation", "function"
        ]
        
        # Add some random terms
        for term in common_terms:
            if term.lower() != correct_answer.lower() and len(wrong_answers) < 5:
                wrong_answers.append(term.capitalize())
        
        # Add variations of the correct answer
        if len(correct_answer) > 3:
            wrong_answers.append(correct_answer + "s")
            wrong_answers.append("Non-" + correct_answer.lower())
        
        return wrong_answers[:3]
    
    def create_summary(self, text_name: str) -> str:
        """Create a summary of the text"""
        if text_name not in self.extracted_texts:
            return f"Text '{text_name}' not found"
        
        text = self.extracted_texts[text_name]
        
        # Simple summarization: first paragraph + key sentences
        paragraphs = text.split('\\n\\n')
        first_paragraph = paragraphs[0] if paragraphs else ""
        
        # Find sentences with key terms
        key_terms = ["transformer", "attention", "neural", "model", "algorithm"]
        key_sentences = []
        
        sentences = self.split_into_sentences(text)
        for sentence in sentences[:50]:  # Look at first 50 sentences
            if any(term in sentence.lower() for term in key_terms):
                key_sentences.append(sentence)
                if len(key_sentences) >= 3:
                    break
        
        summary = f"**Summary of {text_name}:**\\n\\n"
        summary += f"{first_paragraph}\\n\\n"
        summary += "**Key Points:**\\n"
        for i, sentence in enumerate(key_sentences, 1):
            summary += f"{i}. {sentence}\\n"
        
        summary += f"\\n**Statistics:**\\n"
        summary += f"- Total characters: {len(text):,}\\n"
        summary += f"- Estimated reading time: {len(text) // 1000} minutes\\n"
        
        return summary
    
    def list_available_texts(self) -> Dict[str, int]:
        """List all available texts with their sizes"""
        return {name: len(content) for name, content in self.extracted_texts.items()}

if __name__ == "__main__":
    # Test the system
    quiz_system = EnhancedQuizSystem()
    
    print("\\n🎯 Enhanced Quiz System Ready!")
    print("\\nAvailable texts:")
    for name, size in quiz_system.list_available_texts().items():
        print(f"  📄 {name}: {size:,} characters")
    
    # Test question generation
    if quiz_system.extracted_texts:
        first_text = list(quiz_system.extracted_texts.keys())[0]
        print(f"\\n🧪 Testing with '{first_text}'...")
        
        questions = quiz_system.generate_simple_questions(first_text, 3)
        
        for q in questions:
            print(f"\\n❓ Question {q['question_number']}: {q['question']}")
            for i, choice in enumerate(q['choices']):
                print(f"   {chr(65+i)}. {choice}")
            print(f"   ✅ Answer: {chr(65+q['correct_answer'])}") 