import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    """Class for loading and using a pre-trained question generation model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the model inference class.
        
        Args:
            model_path: Path to the pre-trained model directory
        """
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # If model path is provided, load the model
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Check if the model files exist
            if not os.path.exists(model_path):
                logger.error(f"Model directory {model_path} does not exist")
                return False
                
            # Load the model and tokenizer
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully to {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_questions(self, context: str, num_questions: int = 3, max_length: int = 64) -> List[str]:
        """
        Generate questions from a given context using the loaded model.
        
        Args:
            context: Text context to generate questions from
            num_questions: Number of questions to generate
            max_length: Maximum length of generated questions
            
        Returns:
            List of generated questions
        """
        if not self.model_loaded:
            logger.error("No model loaded. Call load_model() first.")
            return []
        
        try:
            # Prepare the input
            input_text = "generate question: " + context
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate questions
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_beams=num_questions * 2,  # Beam search with 2x beams as questions
                    num_return_sequences=num_questions,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode the generated questions
            questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model_loaded
    
    def generate_questions_from_sections(self, sections: List[Dict[str, Any]], 
                                        count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate questions from document sections using the trained model.
        
        Args:
            sections: List of document sections with content
            count: Number of questions to generate
            
        Returns:
            List of generated questions with answers
        """
        if not self.model_loaded:
            logger.error("No model loaded. Call load_model() first.")
            return []
        
        questions = []
        
        # Generate questions from each section
        for section in sections:
            # Skip sections with very little text
            if len(section.get('text', '').strip()) < 50:
                continue
                
            # Generate questions for this section
            section_questions = self.generate_questions(
                context=section['text'],
                num_questions=min(3, count - len(questions))  # Generate up to 3 questions per section
            )
            
            # Convert to question dictionaries and add to list
            for question in section_questions:
                if len(questions) >= count:
                    break
                    
                # Create a question object
                question_obj = {
                    'question': question,
                    'answer': "See explanation",  # This model doesn't generate answers
                    'explanation': f"This question is based on the following text: \"{section['text'][:200]}...\"",
                    'type': 'model_generated',
                    'source_sentence': section['text']
                }
                
                questions.append(question_obj)
                
            # Stop if we have enough questions
            if len(questions) >= count:
                break
        
        return questions[:count]


# Singleton instance
_model_instance = None

def get_model_instance(model_path: Optional[str] = None) -> ModelInference:
    """
    Get the singleton model inference instance.
    
    Args:
        model_path: Optional path to initialize or replace the model
        
    Returns:
        ModelInference instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = ModelInference(model_path)
    elif model_path:
        _model_instance.load_model(model_path)
        
    return _model_instance 