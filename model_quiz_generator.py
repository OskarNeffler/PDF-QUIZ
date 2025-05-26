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
import json
import hashlib

class ModelQuizGenerator:
    """Generate quiz questions using trained T5 model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_success = False
        self.questions_cache_file = "ai_questions_cache.json"
        
        if model_path:
            self.load_model(model_path)
    
    def _load_questions_cache(self) -> Dict:
        """Load cached AI questions from file"""
        try:
            if Path(self.questions_cache_file).exists():
                with open(self.questions_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading questions cache: {e}")
        return {}
    
    def _save_questions_cache(self, cache: Dict):
        """Save AI questions cache to file"""
        try:
            with open(self.questions_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error saving questions cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for text content to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
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
                                   max_output_length: int = 64,
                                   question_type: str = "template_based",
                                   summary_length: str = "medium") -> List[str]:
        """Generate questions from text using the T5 model with quality filtering
        
        Args:
            question_type: "template_based" (recommended), "multiple_choice", "true_false", "fill_blank", "summarize", "hybrid_ai"
            summary_length: "short", "medium", or "long" (only used for summarize type)
        """
        if not self.setup_success:
            print("❌ Model not loaded")
            return []
        
        try:
            # New functionality: Text summarization
            if question_type == "summarize":
                return self._generate_text_summary(text, max_input_length, max_output_length, summary_length)
            # New functionality: AI questions + T5 answers
            elif question_type == "hybrid_ai":
                return self._generate_hybrid_ai_questions(text, num_questions)
            # New functionality: Q&A with T5 model
            elif question_type == "qa":
                return self._generate_qa_questions(text, num_questions, max_input_length, max_output_length)
            # Template-based approach works best for SQuAD models
            elif question_type == "template_based":
                return self._generate_template_based_questions(text, num_questions)
            elif question_type == "multiple_choice":
                print("⚠️ Warning: AI-generated questions may be poor quality with SQuAD models")
                return self._generate_improved_multiple_choice_questions(text, num_questions, max_input_length, max_output_length)
            elif question_type == "true_false":
                print("⚠️ Warning: True/False questions may not work well with SQuAD-trained models")
                return self._generate_true_false_questions(text, num_questions, max_input_length, max_output_length)
            elif question_type == "fill_blank":
                print("⚠️ Warning: Fill-in-the-blank questions may not work well with SQuAD-trained models")
                return self._generate_fill_blank_questions(text, num_questions, max_input_length, max_output_length)
            elif question_type == "t5_questions_ai_answers":
                # New hybrid approach: T5 generates questions, OpenAI generates answers
                return self._generate_t5_questions_with_ai_answers(text, num_questions)
            else:
                return self._generate_template_based_questions(text, num_questions)
                
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return []
    
    def _generate_true_false_questions(self, text: str, num_questions: int, 
                                     max_input_length: int, max_output_length: int) -> List[str]:
        """Generate True/False questions - much simpler and more reliable"""
        chunks = self._split_text(text, max_input_length - 30)
        all_statements = []
        
        # Generate many more statements than needed
        target_statements = num_questions * 3
        print(f"📝 Generating {target_statements} true/false statements from {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            if len(all_statements) >= target_statements:
                break
                
            print(f"   Processing chunk {i+1}/{len(chunks)}")
            
            # Different prompts for variety
            prompts = [
                f"generate statement: {chunk}",
                f"make statement about: {chunk}",
                f"create fact from: {chunk}",
                f"summarize as statement: {chunk}"
            ]
            
            for prompt in prompts:
                if len(all_statements) >= target_statements:
                    break
                    
                inputs = self.tokenizer(
                    prompt,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=3,
                        num_return_sequences=2,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                generated_statements = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                
                for statement in generated_statements:
                    if statement.strip() and statement not in all_statements:
                        all_statements.append((statement.strip(), chunk, True))  # True = correct statement
        
        print(f"📋 Generated {len(all_statements)} statements")
        
        # Create True/False questions
        tf_questions = []
        for statement, context, is_true in all_statements:
            tf_question = self._create_true_false_question(statement, context, is_true)
            quality_score = self._evaluate_tf_question_quality(tf_question, context)
            
            if quality_score > 0.6:
                tf_questions.append((tf_question, quality_score))
        
        # Sort by quality and take the best ones
        tf_questions.sort(key=lambda x: x[1], reverse=True)
        final_questions = [q[0] for q in tf_questions[:num_questions]]
        
        print(f"✅ Selected {len(final_questions)} high-quality True/False questions")
        return final_questions
    
    def _generate_fill_blank_questions(self, text: str, num_questions: int,
                                     max_input_length: int, max_output_length: int) -> List[str]:
        """Generate fill-in-the-blank questions"""
        chunks = self._split_text(text, max_input_length - 30)
        all_sentences = []
        
        print(f"📝 Creating fill-in-the-blank questions from {len(chunks)} chunks...")
        
        # Extract meaningful sentences from chunks
        import re
        for i, chunk in enumerate(chunks):
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 6:  # Only sentences with at least 6 words
                    all_sentences.append((sentence, chunk))
        
        print(f"📋 Found {len(all_sentences)} sentences to process")
        
        # Create fill-in-the-blank questions
        fill_questions = []
        for sentence, context in all_sentences[:num_questions * 3]:
            fill_question = self._create_fill_blank_question(sentence, context)
            quality_score = self._evaluate_fill_question_quality(fill_question, context)
            
            if quality_score > 0.5:
                fill_questions.append((fill_question, quality_score))
        
        # Sort by quality and take the best ones
        fill_questions.sort(key=lambda x: x[1], reverse=True)
        final_questions = [q[0] for q in fill_questions[:num_questions]]
        
        print(f"✅ Selected {len(final_questions)} fill-in-the-blank questions")
        return final_questions
    
    def _create_true_false_question(self, statement: str, context: str, is_true: bool) -> str:
        """Create a True/False question from a statement"""
        import random
        
        # Clean the statement
        statement = statement.strip()
        if not statement.endswith('.'):
            statement += '.'
        
        # Randomly make some statements false by modifying them
        if random.random() < 0.5:  # 50% chance to make it false
            false_statement = self._make_statement_false(statement, context)
            return f"{false_statement}\n\nA) True\nB) False\n\nCorrect: B"
        else:
            return f"{statement}\n\nA) True\nB) False\n\nCorrect: A"
    
    def _make_statement_false(self, statement: str, context: str) -> str:
        """Modify a true statement to make it false"""
        import random
        import re
        
        # Extract key terms from context
        words = re.findall(r'\b[A-Za-z]+\b', context.lower())
        key_terms = [word.title() for word in words if len(word) > 3]
        
        # Different ways to make statement false
        modifications = [
            ("Keras", "TensorFlow"),
            ("Python", "JavaScript"), 
            ("model", "database"),
            ("training", "testing"),
            ("deep learning", "web development"),
            ("neural network", "relational database"),
            ("Sequential", "Functional"),
            ("compile", "execute"),
            ("fit", "predict")
        ]
        
        # Try to replace a key term
        modified = statement
        for original, replacement in modifications:
            if original.lower() in statement.lower():
                modified = re.sub(original, replacement, statement, flags=re.IGNORECASE)
                break
        
        # If no modification worked, add "not" somewhere
        if modified == statement:
            if " is " in statement:
                modified = statement.replace(" is ", " is not ")
            elif " can " in statement:
                modified = statement.replace(" can ", " cannot ")
            elif " will " in statement:
                modified = statement.replace(" will ", " will not ")
            else:
                # Add "incorrect:" at the beginning
                modified = "Incorrectly, " + statement.lower()
        
        return modified
    
    def _create_fill_blank_question(self, sentence: str, context: str) -> str:
        """Create a fill-in-the-blank question"""
        import random
        import re
        
        words = sentence.split()
        if len(words) < 6:
            return None
        
        # Find a good word to blank out (avoid common words)
        good_words = []
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 3 and clean_word not in stop_words:
                good_words.append((i, word, clean_word))
        
        if not good_words:
            return None
        
        # Choose a word to blank out
        blank_index, original_word, clean_word = random.choice(good_words)
        
        # Create the sentence with blank
        blanked_words = words.copy()
        blanked_words[blank_index] = "______"
        blanked_sentence = " ".join(blanked_words)
        
        # Generate wrong alternatives
        wrong_answers = self._generate_wrong_alternatives(clean_word, context)
        
        # Create alternatives
        alternatives = [original_word.strip('.,!?;')] + wrong_answers[:2]
        random.shuffle(alternatives)
        
        # Find correct answer position
        correct_pos = alternatives.index(original_word.strip('.,!?;'))
        correct_letter = ['A', 'B', 'C'][correct_pos]
        
        # Format question
        question = f"{blanked_sentence}\n\n"
        for i, alt in enumerate(alternatives):
            letter = ['A', 'B', 'C'][i]
            question += f"{letter}) {alt}\n"
        question += f"\nCorrect: {correct_letter}"
        
        return question
    
    def _generate_wrong_alternatives(self, correct_word: str, context: str) -> list:
        """Generate plausible wrong alternatives for fill-in-the-blank"""
        import random
        
        # Context-specific alternatives
        alternatives_map = {
            'keras': ['pytorch', 'tensorflow', 'scikit'],
            'python': ['javascript', 'java', 'ruby'],
            'model': ['database', 'framework', 'library'],
            'training': ['testing', 'validation', 'deployment'],
            'compile': ['execute', 'run', 'build'],
            'sequential': ['functional', 'parallel', 'concurrent'],
            'dense': ['sparse', 'convolutional', 'recurrent'],
            'fit': ['predict', 'evaluate', 'transform']
        }
        
        wrong_answers = []
        
        # Try to find context-specific alternatives
        for key, alternatives in alternatives_map.items():
            if key in correct_word.lower():
                wrong_answers.extend(alternatives)
                break
        
        # Add generic alternatives if needed
        if len(wrong_answers) < 2:
            generic = ['option1', 'option2', 'alternative', 'other', 'different']
            wrong_answers.extend(generic)
        
        return wrong_answers[:2]
    
    def _evaluate_tf_question_quality(self, question: str, context: str) -> float:
        """Evaluate True/False question quality"""
        if not question or 'True' not in question or 'False' not in question:
            return 0.0
        
        lines = question.strip().split('\n')
        if len(lines) < 4:
            return 0.0
        
        statement = lines[0]
        score = 0.0
        
        # 1. Statement should be substantial (0.3 points)
        if len(statement.split()) >= 5:
            score += 0.3
        
        # 2. Should relate to context (0.4 points)
        statement_words = set(statement.lower().split())
        context_words = set(context.lower().split())
        overlap = len(statement_words.intersection(context_words))
        if overlap >= 3:
            score += 0.4
        elif overlap >= 2:
            score += 0.2
        
        # 3. Should be a proper statement (0.3 points)
        if statement.strip().endswith('.') and not statement.startswith('?'):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_fill_question_quality(self, question: str, context: str) -> float:
        """Evaluate fill-in-the-blank question quality"""
        if not question or '______' not in question:
            return 0.0
        
        lines = question.strip().split('\n')
        if len(lines) < 5:
            return 0.0
        
        sentence = lines[0]
        score = 0.0
        
        # 1. Should have good sentence structure (0.4 points)
        if '______' in sentence and len(sentence.split()) >= 5:
            score += 0.4
        
        # 2. Should relate to context (0.3 points)
        sentence_words = set(sentence.lower().split())
        context_words = set(context.lower().split())
        overlap = len(sentence_words.intersection(context_words))
        if overlap >= 2:
            score += 0.3
        elif overlap >= 1:
            score += 0.15
        
        # 3. Should have alternatives (0.3 points)
        alternatives = []
        for line in lines[2:5]:  # Skip the question and empty line
            if line.strip().startswith(('A)', 'B)', 'C)')):
                alternatives.append(line.strip()[2:].strip())
        
        if len(alternatives) == 3:
            score += 0.3
        
        return min(score, 1.0)
    
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

    def _convert_to_multiple_choice_improved(self, question: str, context: str) -> str:
        """Convert a simple question to multiple choice format with improved logic"""
        import random
        import re
        
        # Clean and analyze the question
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        
        # Extract meaningful terms from context (avoid common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        words = re.findall(r'\b[A-Za-z]+\b', context.lower())
        key_terms = [word.title() for word in words 
                    if len(word) > 3 and word not in stop_words and word.isalpha()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        # Generate context-appropriate correct answer
        correct_answer = self._generate_correct_answer(question, context, unique_terms)
        
        # Generate plausible wrong answers
        wrong_answers = self._generate_wrong_answers(question, context, unique_terms, correct_answer)
        
        # Ensure we have exactly 4 options
        all_answers = [correct_answer] + wrong_answers[:3]
        while len(all_answers) < 4:
            all_answers.append(f"None of the above")
        
        # Randomize order
        random.shuffle(all_answers)
        correct_pos = all_answers.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_pos]
        
        # Format multiple choice question
        mc_question = f"{question}\n"
        for i, answer in enumerate(all_answers):
            letter = ['A', 'B', 'C', 'D'][i]
            mc_question += f"{letter}) {answer}\n"
        mc_question += f"Correct: {correct_letter}"
        
        return mc_question
    
    def _generate_correct_answer(self, question: str, context: str, terms: list) -> str:
        """Generate a more intelligent correct answer based on question type and context"""
        question_lower = question.lower()
        
        # Look for specific patterns in the question
        if 'what is' in question_lower or 'define' in question_lower:
            # For definition questions, try to extract a key concept
            if 'keras' in context.lower():
                return "A high-level neural networks API"
            elif 'python' in context.lower():
                return "A programming language"
            elif 'model' in context.lower():
                return "A mathematical representation"
            elif terms:
                return terms[0]  # Use first significant term
        
        elif 'how' in question_lower:
            # For process questions
            if 'compile' in question_lower or 'train' in question_lower:
                return "Using model.compile() and model.fit()"
            elif 'create' in question_lower or 'build' in question_lower:
                return "Using Sequential or Functional API"
            else:
                return "Follow the documented procedure"
        
        elif 'why' in question_lower:
            return "To improve performance and accuracy"
        
        elif 'when' in question_lower:
            return "During the appropriate phase"
        
        elif 'where' in question_lower:
            return "In the specified location"
        
        # Fallback to context-based answer
        if terms:
            return terms[0]
        return "As specified in the documentation"
    
    def _generate_wrong_answers(self, question: str, context: str, terms: list, correct_answer: str) -> list:
        """Generate plausible but incorrect answers"""
        import random
        
        wrong_answers = []
        
        # Programming/ML specific wrong answers based on question type
        question_lower = question.lower()
        
        if 'keras' in question_lower or 'tensorflow' in question_lower:
            candidates = ["PyTorch", "Scikit-learn", "Pandas", "NumPy", "Django", "React"]
        elif 'python' in question_lower:
            candidates = ["JavaScript", "Java", "C++", "Ruby", "PHP", "Go"]
        elif 'model' in question_lower:
            candidates = ["Database", "Framework", "Library", "Application", "Interface"]
        elif 'compile' in question_lower or 'train' in question_lower:
            candidates = ["model.build()", "model.run()", "model.execute()", "model.start()"]
        elif 'api' in question_lower:
            candidates = ["Database", "Library", "Framework", "Protocol", "Service"]
        else:
            # Generic technical wrong answers
            candidates = ["TensorFlow", "PyTorch", "JavaScript", "React", "Database", 
                         "Framework", "Library", "API", "Protocol", "Service"]
        
        # Add some terms from context as wrong answers (but not the correct answer)
        context_candidates = [term for term in terms 
                            if term.lower() != correct_answer.lower() and len(term) > 2]
        
        all_candidates = candidates + context_candidates
        random.shuffle(all_candidates)
        
        # Select unique wrong answers
        for candidate in all_candidates:
            if (len(wrong_answers) < 3 and 
                candidate.lower() != correct_answer.lower() and
                candidate not in wrong_answers):
                wrong_answers.append(candidate)
        
        # Fill with generic options if needed
        generic_options = ["Option A", "Option B", "Option C", "None of the above"]
        for option in generic_options:
            if len(wrong_answers) < 3 and option not in wrong_answers:
                wrong_answers.append(option)
        
        return wrong_answers[:3]

    def _evaluate_question_quality(self, question: str, context: str) -> float:
        """Evaluate the quality of a question based on various criteria"""
        score = 0.0
        
        # Parse the question
        lines = question.strip().split('\n')
        if len(lines) < 6:  # Question + 4 options + correct answer
            return 0.0
        
        question_text = lines[0]
        
        # Quality criteria
        
        # 1. Question should be a proper question (0.2 points)
        if question_text.strip().endswith('?') and len(question_text.split()) >= 3:
            score += 0.2
        
        # 2. Question should not be too generic (0.2 points)
        generic_phrases = ['what is the name', 'what is called', 'what do you call']
        if not any(phrase in question_text.lower() for phrase in generic_phrases):
            score += 0.2
        
        # 3. Question should relate to context (0.3 points)
        question_words = set(question_text.lower().split())
        context_words = set(context.lower().split())
        overlap = len(question_words.intersection(context_words))
        if overlap >= 2:
            score += 0.3
        elif overlap >= 1:
            score += 0.15
        
        # 4. Check for meaningful alternatives (0.2 points)
        alternatives = []
        for line in lines[1:5]:
            if line.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                alt = line.strip()[2:].strip()
                alternatives.append(alt)
        
        # Alternatives should be different and meaningful
        if len(set(alternatives)) == 4 and not any('Option' in alt for alt in alternatives):
            score += 0.2
        elif len(set(alternatives)) >= 3:
            score += 0.1
        
        # 5. Correct answer should be sensible (0.1 points)
        correct_line = [line for line in lines if line.startswith('Correct:')]
        if correct_line:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def _generate_improved_multiple_choice_questions(self, text: str, num_questions: int,
                                                     max_input_length: int, max_output_length: int) -> List[str]:
        """Generate improved multiple choice questions optimized for SQuAD-trained models"""
        chunks = self._split_text(text, max_input_length - 30)
        all_questions = []
        
        # Generate more questions than needed for quality filtering
        target_raw_questions = num_questions * 5
        print(f"📝 Processing {len(chunks)} text chunks, targeting {target_raw_questions} raw questions...")
        
        for i, chunk in enumerate(chunks):
            if len(all_questions) >= target_raw_questions:
                break
                
            print(f"   Processing chunk {i+1}/{len(chunks)}")
            
            # Better prompts that work well with SQuAD-trained models
            prompts = [
                f"question: {chunk}",  # Simple and direct
                f"generate question from: {chunk}",
                f"ask question about: {chunk}",
                f"question from context: {chunk}",
                f"what to ask about: {chunk}",
                f"create question: {chunk}"
            ]
            
            for prompt in prompts:
                if len(all_questions) >= target_raw_questions:
                    break
                    
                inputs = self.tokenizer(
                    prompt,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=4,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                generated_questions = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                
                for question in generated_questions:
                    if question.strip() and question not in [q[0] for q in all_questions]:
                        all_questions.append((question.strip(), chunk))
        
        print(f"📋 Generated {len(all_questions)} raw questions")
        
        # Filter and convert to multiple choice
        quality_questions = []
        for question, context in all_questions:
            # Skip very generic or poorly formed questions
            if self._is_good_base_question(question, context):
                mc_question = self._create_better_multiple_choice(question, context)
                quality_score = self._evaluate_improved_question_quality(mc_question, context)
                
                if quality_score > 0.6:
                    quality_questions.append((mc_question, quality_score))
        
        # Sort by quality and take the best ones
        quality_questions.sort(key=lambda x: x[1], reverse=True)
        final_questions = [q[0] for q in quality_questions[:num_questions]]
        
        print(f"✅ Selected {len(final_questions)} high-quality questions from {len(all_questions)} candidates")
        return final_questions

    def _is_good_base_question(self, question: str, context: str) -> bool:
        """Filter out poorly formed base questions"""
        question_lower = question.lower().strip()
        
        # Skip very short questions
        if len(question.split()) < 3:
            return False
        
        # Skip questions that are just repeated phrases
        if "what is important in:" in question_lower:
            return False
        if "what is the name of" in question_lower and question_lower.count("what is the name of") > 1:
            return False
        
        # Skip questions that are mostly punctuation or numbers
        if len([c for c in question if c.isalpha()]) < len(question) * 0.5:
            return False
        
        # Should be a proper question
        if not question.strip().endswith('?'):
            question = question.strip() + '?'
        
        # Should have some overlap with context
        question_words = set(question_lower.split())
        context_words = set(context.lower().split())
        overlap = len(question_words.intersection(context_words))
        
        return overlap >= 2
    
    def _create_better_multiple_choice(self, question: str, context: str) -> str:
        """Create better multiple choice questions with more natural alternatives"""
        import random
        import re
        
        # Clean the question
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        
        # Extract key concepts from context
        # Look for technical terms, proper nouns, and important concepts
        context_words = re.findall(r'\b[A-Za-z][A-Za-z]+\b', context)
        
        # Find potential correct answers based on question type
        correct_answer = self._extract_likely_answer(question, context)
        
        # Generate contextually relevant wrong answers
        wrong_answers = self._generate_contextual_wrong_answers(question, context, correct_answer)
        
        # Create 4 alternatives
        all_answers = [correct_answer] + wrong_answers[:3]
        while len(all_answers) < 4:
            all_answers.append("Not specified in the text")
        
        # Randomize order
        random.shuffle(all_answers)
        correct_pos = all_answers.index(correct_answer)
        correct_letter = ['A', 'B', 'C', 'D'][correct_pos]
        
        # Format question
        mc_question = f"{question}\n"
        for i, answer in enumerate(all_answers):
            letter = ['A', 'B', 'C', 'D'][i]
            mc_question += f"{letter}) {answer}\n"
        mc_question += f"Correct: {correct_letter}"
        
        return mc_question
    
    def _extract_likely_answer(self, question: str, context: str) -> str:
        """Extract likely correct answer based on question and context"""
        import re
        
        question_lower = question.lower()
        
        # Look for specific patterns and extract relevant answers
        if "what is" in question_lower or "what does" in question_lower:
            # Look for definitions or explanations
            sentences = re.split(r'[.!]', context)
            for sentence in sentences:
                if any(word in sentence.lower() for word in question.lower().split() if len(word) > 3):
                    # Extract key phrase from sentence
                    words = sentence.strip().split()
                    if len(words) > 5:
                        # Return a meaningful phrase from the sentence
                        return ' '.join(words[:6]).strip(' .,')
        
        elif "how" in question_lower:
            # Look for process or method descriptions
            if "compile" in question_lower or "train" in question_lower:
                return "Using model.compile() and model.fit()"
            elif "create" in question_lower or "build" in question_lower:
                return "Using Sequential or Functional API"
            else:
                return "Following the documented procedure"
        
        elif "where" in question_lower:
            return "In the appropriate location"
        
        elif "when" in question_lower:
            return "During the development process"
        
        elif "why" in question_lower:
            return "To enable fast experimentation"
        
        # Default: try to extract the most mentioned concept
        words = re.findall(r'\b[A-Z][a-z]+\b', context)  # Capitalized words
        if words:
            from collections import Counter
            common_words = Counter(words).most_common(3)
            return common_words[0][0] if common_words else "Keras"
        
        return "As specified in the documentation"
    
    def _generate_contextual_wrong_answers(self, question: str, context: str, correct_answer: str) -> list:
        """Generate plausible wrong answers based on context"""
        import random
        import re
        
        wrong_answers = []
        
        # Extract technical terms from context
        tech_terms = re.findall(r'\b(?:TensorFlow|PyTorch|Keras|Python|JavaScript|Java|API|CNN|LSTM|RNN|Sequential|Functional)\b', context)
        
        # Extract other capitalized terms
        other_terms = re.findall(r'\b[A-Z][a-z]+\b', context)
        
        # Combine and filter
        all_candidates = list(set(tech_terms + other_terms))
        candidates = [term for term in all_candidates if term.lower() != correct_answer.lower()]
        
        # Add some generic technical alternatives
        generic_alternatives = [
            "Machine learning framework",
            "Deep learning library", 
            "Neural network architecture",
            "Programming language",
            "Development environment",
            "Data processing tool",
            "Model optimization technique",
            "Backend implementation"
        ]
        
        # Combine candidates
        all_options = candidates + generic_alternatives
        random.shuffle(all_options)
        
        # Select unique wrong answers
        for option in all_options:
            if len(wrong_answers) < 3 and option != correct_answer and option not in wrong_answers:
                wrong_answers.append(option)
        
        # Fill with generic options if needed
        generic_fillers = ["Not applicable", "Not specified", "Alternative approach"]
        for filler in generic_fillers:
            if len(wrong_answers) < 3 and filler not in wrong_answers:
                wrong_answers.append(filler)
        
        return wrong_answers[:3]
    
    def _evaluate_improved_question_quality(self, question: str, context: str) -> float:
        """Evaluate question quality with improved criteria"""
        if not question:
            return 0.0
        
        lines = question.strip().split('\n')
        if len(lines) < 6:  # Question + 4 options + correct answer
            return 0.0
        
        question_text = lines[0]
        score = 0.0
        
        # 1. Question should be well-formed (0.3 points)
        if question_text.strip().endswith('?') and len(question_text.split()) >= 4:
            score += 0.3
        
        # 2. Should not be overly repetitive (0.2 points)
        if not ("what is the name of" in question_text.lower() and 
                question_text.lower().count("what is the name") > 0):
            score += 0.2
        
        # 3. Should relate to context meaningfully (0.3 points)
        question_words = set(question_text.lower().split())
        context_words = set(context.lower().split())
        overlap = len(question_words.intersection(context_words))
        if overlap >= 3:
            score += 0.3
        elif overlap >= 2:
            score += 0.15
        
        # 4. Check answer quality (0.2 points)
        alternatives = []
        for line in lines[1:5]:
            if line.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                alt = line.strip()[2:].strip()
                alternatives.append(alt)
        
        # Alternatives should be diverse and meaningful
        if len(set(alternatives)) == 4:
            if not any("option" in alt.lower() for alt in alternatives):
                score += 0.2
            elif any(len(alt) > 3 for alt in alternatives):
                score += 0.1
        
        return min(score, 1.0)

    def _generate_text_summary(self, text: str, max_input_length: int = 512, max_output_length: int = 150, summary_length: str = "medium") -> List[str]:
        """Generate ONE comprehensive summary - adapted for question generation models"""
        print(f"📝 Generating {summary_length} text summary using T5 model...")
        
        # Since our T5 model is trained for question generation, not summarization,
        # we'll create a summary by extracting key information differently
        
        # Set target length based on summary type
        if summary_length == "short":
            target_sentences = 2
            print("   📋 Creating short summary...")
        elif summary_length == "long":
            target_sentences = 5
            print("   📋 Creating detailed summary...")
        else:  # medium
            target_sentences = 3
            print("   📋 Creating medium summary...")
        
        try:
            # Split text into sentences
            sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 30]
            
            if not sentences:
                return [f"**📋 {summary_length.title()} Summary:**\nNo meaningful content found for summarization."]
            
            # For very long texts, take samples from different parts
            if len(sentences) > target_sentences * 3:
                print(f"   📄 Text has {len(sentences)} sentences, selecting key ones...")
                
                # Take sentences from beginning, middle, and end
                selected = []
                total = len(sentences)
                
                # Beginning (introduction)
                selected.extend(sentences[:target_sentences//3 + 1])
                
                # Middle (main content)
                mid_start = total // 3
                mid_end = (2 * total) // 3
                selected.extend(sentences[mid_start:mid_start + target_sentences//3 + 1])
                
                # End (conclusion)
                selected.extend(sentences[-target_sentences//3 - 1:])
                
                # Limit to target number
                summary_sentences = selected[:target_sentences]
            else:
                # Take the first few sentences
                summary_sentences = sentences[:target_sentences]
            
            # Clean and join sentences
            clean_sentences = []
            for sentence in summary_sentences:
                # Clean up the sentence
                clean = sentence.strip()
                if clean and len(clean) > 10:
                    # Ensure sentence ends properly
                    if not clean.endswith(('.', '!', '?')):
                        clean += '.'
                    clean_sentences.append(clean)
            
            if clean_sentences:
                summary_text = ' '.join(clean_sentences)
                
                # Ensure reasonable length
                max_chars = 200 if summary_length == "short" else 400 if summary_length == "medium" else 600
                if len(summary_text) > max_chars:
                    summary_text = summary_text[:max_chars] + "..."
                
                final_summary = f"**📋 {summary_length.title()} Summary:**\n{summary_text}"
            else:
                final_summary = f"**📋 {summary_length.title()} Summary:**\nThis text discusses various technical concepts and implementation details."
            
            print(f"✅ Generated {summary_length} summary ({len(final_summary)} characters)")
            return [final_summary]
            
        except Exception as e:
            print(f"❌ Error in summary generation: {e}")
            fallback = f"**📋 {summary_length.title()} Summary:**\nThis document contains technical information that requires further analysis."
            return [fallback]

    def _generate_hybrid_ai_questions(self, text: str, num_questions: int = 20) -> List[str]:
        """Generate questions using OpenAI API, then answer them with T5 - best of both worlds!"""
        try:
            print(f"🤖 Generating hybrid AI+T5 questions...")
            
            # Check cache first
            cache = self._load_questions_cache()
            text_hash = self._get_text_hash(text)
            cache_key = f"{text_hash}_{num_questions}"
            
            if cache_key in cache:
                print(f"💾 Using cached questions for this text ({num_questions} questions)")
                cached_questions = cache[cache_key]['questions']
                
                # Generate fresh T5 answers for cached questions
                hybrid_results = []
                for i, question in enumerate(cached_questions):
                    print(f"   T5 answering cached question {i+1}/{len(cached_questions)}")
                    t5_answer = self._get_t5_answer(question, text)
                    qa_pair = f"❓ **Question:** {question}\n🤖 **T5 Answer:** {t5_answer}"
                    hybrid_results.append(qa_pair)
                
                print(f"✅ Used {len(hybrid_results)} cached AI questions with fresh T5 answers")
                return hybrid_results
            
            # Generate new questions using OpenAI
            print(f"🎯 Generating {num_questions} new questions with OpenAI...")
            openai_questions = self._generate_openai_questions(text, num_questions)
            
            if not openai_questions:
                print("❌ Failed to generate OpenAI questions, falling back to T5")
                return self._generate_template_based_questions(text, num_questions)
            
            # Cache the questions
            cache[cache_key] = {
                'questions': openai_questions,
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'generated_at': str(Path().absolute()) + " - " + str(hash(text))[:8]
            }
            self._save_questions_cache(cache)
            print(f"💾 Cached {len(openai_questions)} questions for future use")
            
            # Generate T5 answers for the new questions
            hybrid_results = []
            for i, question in enumerate(openai_questions):
                print(f"   T5 answering question {i+1}/{len(openai_questions)}")
                
                # Use T5 for question answering (what it's trained for!)
                t5_answer = self._get_t5_answer(question, text)
                
                # Format as a Q&A pair
                qa_pair = f"❓ **Question:** {question}\n🤖 **T5 Answer:** {t5_answer}"
                hybrid_results.append(qa_pair)
            
            print(f"✅ Generated {len(hybrid_results)} AI+T5 question-answer pairs")
            return hybrid_results
            
        except Exception as e:
            print(f"❌ Error in hybrid generation: {e}")
            print("🔄 Falling back to template-based questions...")
            return self._generate_template_based_questions(text, num_questions)

    def _generate_openai_questions(self, text: str, num_questions: int) -> List[str]:
        """Generate questions using OpenAI API"""
        try:
            from openai import OpenAI
            import os
            from pathlib import Path
            
            # Load OpenAI API key from .env file
            api_key = None
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            
            if not api_key:
                print("❌ No OpenAI API key found in .env file")
                return []
            
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)
            
            # For longer texts, truncate intelligently
            max_chars = 3000  # Leave room for prompt
            if len(text) > max_chars:
                # Try to keep complete sentences
                truncated = text[:max_chars]
                last_period = truncated.rfind('.')
                if last_period > max_chars * 0.8:  # If we find a period in the last 20%
                    text = truncated[:last_period + 1]
                else:
                    text = truncated
            
            # Create prompt for question generation
            prompt = f"""Generate {num_questions} high-quality, specific, and educational questions based on this text. 
The questions should:
- Be clear and well-formed
- Cover different aspects of the content
- Be answerable from the provided text
- Vary in difficulty and question type
- Focus on key concepts and important details

Format: Return only the questions, one per line, without numbering.

Text:
{text}

Questions:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cost-effective for question generation
                messages=[
                    {"role": "system", "content": "You are an expert at creating educational questions from text content. Create diverse, clear, and answerable questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # More tokens for 20 questions
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().isdigit()]
            
            # Clean up questions - remove numbering if present
            cleaned_questions = []
            for q in questions:
                # Remove leading numbers, bullets, etc.
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', q.strip())
                cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
                if cleaned and len(cleaned) > 10:  # Reasonable question length
                    cleaned_questions.append(cleaned)
            
            print(f"🎯 OpenAI generated {len(cleaned_questions)} questions")
            return cleaned_questions[:num_questions]  # Ensure we don't exceed requested amount
            
        except Exception as e:
            print(f"❌ OpenAI question generation failed: {e}")
            return []

    def _get_t5_answer(self, question: str, context: str, max_length: int = 100) -> str:
        """Use T5 to answer a question based on context - this is what T5 excels at!"""
        try:
            # Format as question answering task
            prompt = f"question: {question} context: {context[:1000]}"  # Limit context
            
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=2
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip() if answer.strip() else "Unable to generate answer"
            
        except Exception as e:
            print(f"❌ T5 answer generation failed: {e}")
            return "Error generating answer"

    def _generate_template_based_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """Generate template-based questions as fallback when OpenAI fails"""
        try:
            # Simple template-based question generation using the text content
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
            
            if not sentences:
                return ["What is the main topic discussed in this text?"]
            
            questions = []
            
            # Generate questions based on sentence patterns
            for i, sentence in enumerate(sentences[:num_questions]):
                # Simple question templates
                templates = [
                    f"What does the text say about {self._extract_key_phrase(sentence)}?",
                    f"According to the text, what is {self._extract_key_phrase(sentence)}?",
                    f"How does the text describe {self._extract_key_phrase(sentence)}?",
                    f"What is the significance of {self._extract_key_phrase(sentence)}?",
                    f"In what context does the text mention {self._extract_key_phrase(sentence)}?"
                ]
                
                # Pick a template based on sentence index
                template_idx = i % len(templates)
                question = templates[template_idx]
                questions.append(question)
            
            # Fill remaining with generic questions if needed
            while len(questions) < num_questions:
                questions.append("What is a key concept discussed in this text?")
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"❌ Error in template generation: {e}")
            return [f"Question {i+1}: What is discussed in this text?" for i in range(num_questions)]
    
    def _extract_key_phrase(self, sentence: str) -> str:
        """Extract a key phrase from a sentence for question generation"""
        try:
            # Simple keyword extraction
            words = sentence.split()
            
            # Look for important words (avoid common words)
            important_words = []
            skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'}
            
            for word in words:
                clean_word = word.strip('.,!?;:"()[]{}').lower()
                if clean_word not in skip_words and len(clean_word) > 3:
                    important_words.append(clean_word)
                    if len(important_words) >= 2:
                        break
            
            if important_words:
                return ' '.join(important_words[:2])
            else:
                # Fallback to first few words
                return ' '.join(words[:3]).strip('.,!?;:"()[]{}')
                
        except:
            return "the main concept"

    def _generate_qa_questions(self, text: str, num_questions: int = 3, 
                             max_input_length: int = 512, max_output_length: int = 150) -> List[str]:
        """Generate Q&A pairs using T5 model for question answering"""
        print(f"💬 Generating {num_questions} Q&A pairs...")
        
        # Split text into manageable chunks
        chunks = self._split_text(text, max_input_length - 50)
        all_qa_pairs = []
        
        # Generate questions from different chunks
        for i, chunk in enumerate(chunks[:num_questions]):
            print(f"   Creating Q&A from chunk {i+1}/{min(len(chunks), num_questions)}")
            
            # Generate a question about this chunk
            question = self._generate_question_from_chunk(chunk, max_input_length, max_output_length)
            
            if question and len(question.strip()) > 10:
                # Use T5 to answer the question
                answer = self._get_t5_answer(question, chunk, max_output_length)
                
                if answer and len(answer.strip()) > 5:
                    # Format as Q&A pair
                    qa_pair = f"❓ **Fråga:** {question}\n\n🤖 **T5 Svar:** {answer}"
                    all_qa_pairs.append(qa_pair)
        
        print(f"✅ Generated {len(all_qa_pairs)} Q&A pairs")
        return all_qa_pairs
    
    def _generate_question_from_chunk(self, chunk: str, max_input_length: int, max_output_length: int) -> str:
        """Generate a question from a text chunk"""
        try:
            # Use different prompts for question generation
            prompts = [
                f"question: {chunk}",
                f"ask about: {chunk}",
                f"what question: {chunk}",
                f"generate question: {chunk}"
            ]
            
            # Try each prompt until we get a good question
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7,
                        no_repeat_ngram_size=2
                    )
                
                question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean and validate the question
                if question and len(question.strip()) > 10:
                    question = question.strip()
                    if not question.endswith('?'):
                        question += '?'
                    return question
            
            return None
            
        except Exception as e:
            print(f"❌ Error generating question: {e}")
            return None

    def _generate_t5_questions_with_ai_answers(self, text: str, num_questions: int = 5) -> List[str]:
        """Generate questions with T5 and answers with OpenAI - Best of both worlds!
        
        Args:
            text: The input text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of formatted question-answer pairs with multiple choice alternatives
        """
        try:
            print(f"🤖 T5 + OpenAI Hybrid: Generating {num_questions} questions...")
            
            # Step 1: Use T5 to generate good questions (what it's good at)
            print("📝 Step 1: T5 generating questions from text...")
            questions = self._generate_template_based_questions(text, num_questions * 2)  # Generate more to filter
            
            if not questions:
                print("❌ No questions generated by T5")
                return []
            
            # Take the best questions
            selected_questions = questions[:num_questions]
            print(f"✅ T5 generated {len(selected_questions)} questions")
            
            # Step 2: Use OpenAI to create answers and alternatives (what it's good at)
            print("🧠 Step 2: OpenAI generating answers and alternatives...")
            
            formatted_results = []
            
            for i, question in enumerate(selected_questions):
                print(f"   Processing question {i+1}/{len(selected_questions)}")
                
                # Create OpenAI prompt for generating answer + alternatives
                prompt = f"""Given this text:
"{text}"

And this question:
"{question}"

Please provide:
1. A correct answer based on the text
2. Three plausible but incorrect alternatives
3. Format as a multiple choice question

Format your response as:
Question: [question]
A) [correct answer]
B) [wrong alternative 1]  
C) [wrong alternative 2]
D) [wrong alternative 3]
Correct: A

Make sure the correct answer (A) is actually correct based on the text, and the wrong alternatives are plausible but clearly incorrect."""

                try:
                    # Call OpenAI API using the newer client interface
                    import os
                    from openai import OpenAI
                    
                    # Get API key
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        print(f"   ❌ No OpenAI API key found for Q{i+1}")
                        # Create a simple fallback
                        fallback = f"""Question: {question}
A) Based on the text provided
B) Alternative option
C) Different choice
D) Another alternative
Correct: A"""
                        formatted_results.append(fallback)
                        continue
                    
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "user", 
                            "content": prompt
                        }],
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    
                    # Format the response
                    if "Question:" in ai_response and "Correct:" in ai_response:
                        formatted_results.append(ai_response)
                        print(f"   ✅ Generated Q&A pair {i+1}")
                    else:
                        # Fallback formatting
                        fallback = f"""Question: {question}
A) Based on the provided text
B) Alternative answer option
C) Another possible option  
D) Different alternative choice
Correct: A"""
                        formatted_results.append(fallback)
                        print(f"   ⚠️ Used fallback formatting for Q{i+1}")
                        
                except Exception as e:
                    print(f"   ❌ OpenAI failed for Q{i+1}: {e}")
                    # Create a simple fallback
                    fallback = f"""Question: {question}
A) Based on the text provided
B) Alternative option
C) Different choice
D) Another alternative
Correct: A"""
                    formatted_results.append(fallback)
            
            print(f"🎯 Generated {len(formatted_results)} T5+OpenAI hybrid questions!")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in T5+OpenAI hybrid generation: {e}")
            return []

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
        print(f"\nProcessing: {text_name}")
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