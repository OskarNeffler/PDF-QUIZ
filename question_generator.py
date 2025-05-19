import random
import re
import spacy
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Any, Tuple, Optional
import os
import json

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


class QuestionGenerator:
    """Base class for question generators."""
    
    def __init__(self):
        self.question_type = None
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a question from a sentence."""
        raise NotImplementedError("Subclasses must implement generate()")


class MultipleChoiceGenerator(QuestionGenerator):
    """Generate multiple choice questions."""
    
    def __init__(self):
        super().__init__()
        self.question_type = 'multiple_choice'
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sentence_text = sentence['text']
        doc = nlp(sentence_text)
        
        # Find a suitable entity or key term to ask about
        target_entities = []
        
        # Try to find named entities first
        for entity in section['entities']:
            if entity['text'] in sentence_text:
                target_entities.append((entity['text'], entity['label']))
        
        # If no entities found, look for key nouns
        if not target_entities:
            for token in doc:
                if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop and len(token.text) > 3:
                    target_entities.append((token.text, token.pos_))
        
        if not target_entities:
            return None  # No suitable entities found
        
        # Select a random entity to ask about
        target_entity, entity_type = random.choice(target_entities)
        
        # Create the question
        question_text = self._create_question_text(sentence_text, target_entity, entity_type)
        
        # Create distractors (wrong answers)
        distractors = self._create_distractors(target_entity, entity_type, section)
        
        # If we couldn't generate enough distractors, return None
        if len(distractors) < 3:
            return None
        
        # Take the first 3 distractors
        options = [target_entity] + distractors[:3]
        random.shuffle(options)
        
        # Create explanation
        explanation = f"The correct answer is '{target_entity}' as mentioned in the text: \"{sentence_text}\""
        
        return {
            'question': question_text,
            'options': options,
            'answer': target_entity,
            'explanation': explanation,
            'type': self.question_type,
            'source_sentence': sentence_text
        }
    
    def _create_question_text(self, sentence: str, entity: str, entity_type: str) -> str:
        """Create a question text based on sentence type and entity."""
        # Replace the entity with a question placeholder
        question = sentence.replace(entity, "___")
        
        # Convert statement to question
        if question.endswith('.'):
            question = question[:-1] + '?'
        else:
            question = question + '?'
        
        # Add a prefix based on entity type
        if entity_type == 'PERSON':
            prefix = "Who is"
        elif entity_type == 'GPE' or entity_type == 'LOC':
            prefix = "Which location is"
        elif entity_type == 'DATE' or entity_type == 'TIME':
            prefix = "When is"
        elif entity_type == 'PERCENT' or entity_type == 'MONEY' or entity_type == 'QUANTITY':
            prefix = "What amount is"
        else:
            prefix = "What is"
        
        # Simple conversion to question format
        question = f"{prefix} the correct option to fill in the blank: {question}"
        
        return question
    
    def _create_distractors(self, correct_answer: str, entity_type: str, section: Dict[str, Any]) -> List[str]:
        """Generate plausible but incorrect options."""
        distractors = []
        
        # Method 1: Use similar entities from the same section
        similar_entities = []
        for entity in section['entities']:
            if entity['text'] != correct_answer and entity['label'] == entity_type:
                similar_entities.append(entity['text'])
        
        # Add some of these similar entities as distractors
        if similar_entities:
            random.shuffle(similar_entities)
            distractors.extend(similar_entities[:min(2, len(similar_entities))])
        
        # Method 2: For nouns, use WordNet to find related terms
        if len(distractors) < 3 and entity_type in ('NOUN', 'PROPN'):
            synsets = wordnet.synsets(correct_answer)
            
            if synsets:
                # Get hyponyms and hypernyms (more specific and more general terms)
                related_terms = []
                
                for synset in synsets:
                    # Get hyponyms (more specific terms)
                    for hyponym in synset.hyponyms():
                        related_terms.append(hyponym.lemma_names()[0].replace('_', ' '))
                    
                    # Get hypernyms (more general terms)
                    for hypernym in synset.hypernyms():
                        related_terms.append(hypernym.lemma_names()[0].replace('_', ' '))
                
                # Filter and add as distractors
                filtered_terms = [term for term in related_terms if term != correct_answer]
                if filtered_terms:
                    random.shuffle(filtered_terms)
                    distractors.extend(filtered_terms[:min(2, len(filtered_terms))])
        
        # Method 3: Use key terms from other paragraphs
        if len(distractors) < 3:
            all_terms = []
            for paragraph_terms in section['paragraph_key_terms']:
                for term, score in paragraph_terms:
                    if term != correct_answer:
                        all_terms.append(term)
            
            if all_terms:
                random.shuffle(all_terms)
                distractors.extend(all_terms[:min(3 - len(distractors), len(all_terms))])
        
        # Ensure we have unique distractors
        distractors = list(set(distractors))
        
        return distractors


class TrueFalseGenerator(QuestionGenerator):
    """Generate true/false questions."""
    
    def __init__(self):
        super().__init__()
        self.question_type = 'true_false'
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sentence_text = sentence['text']
        
        # Determine if we should keep the statement true or make it false
        make_false = random.choice([True, False])
        
        if make_false:
            # Create a false statement by modifying the original
            modified_sentence, modification = self._create_false_statement(sentence_text, section)
            
            if not modified_sentence:
                return None  # Could not create a false statement
            
            question_text = f"True or False: {modified_sentence}"
            answer = "False"
            explanation = f"The statement is false. {modification}"
        else:
            # Keep the statement true
            question_text = f"True or False: {sentence_text}"
            answer = "True"
            explanation = f"The statement is true as stated in the text."
        
        return {
            'question': question_text,
            'options': ["True", "False"],
            'answer': answer,
            'explanation': explanation,
            'type': self.question_type,
            'source_sentence': sentence_text
        }
    
    def _create_false_statement(self, sentence: str, section: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Create a false statement by modifying the original."""
        doc = nlp(sentence)
        
        # Strategy 1: Replace a named entity with another of the same type
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
        
        if entities:
            # Choose a random entity to replace
            entity_text, entity_label, start, end = random.choice(entities)
            
            # Find replacement entities of the same type from other sentences
            replacement_entities = []
            for e in section['entities']:
                if e['text'] != entity_text and e['label'] == entity_label:
                    replacement_entities.append(e['text'])
            
            if replacement_entities:
                replacement = random.choice(replacement_entities)
                modified = sentence[:start] + replacement + sentence[end:]
                return modified, f"The original text mentions '{entity_text}', not '{replacement}'."
        
        # Strategy 2: Negate the statement
        negation_terms = {
            "is": "is not",
            "are": "are not",
            "was": "was not",
            "were": "were not",
            "has": "does not have",
            "have": "do not have",
            "can": "cannot",
            "will": "will not"
        }
        
        for term, negation in negation_terms.items():
            pattern = r'\b' + term + r'\b'
            if re.search(pattern, sentence):
                modified = re.sub(pattern, negation, sentence, count=1)
                return modified, f"The original statement states the opposite."
        
        # Strategy 3: Replace numbers with different ones
        numbers = re.findall(r'\b\d+\b', sentence)
        if numbers:
            number = random.choice(numbers)
            # Generate a different number (±20-50%)
            original = int(number)
            change = random.uniform(0.2, 0.5) * original
            new_number = original + random.choice([-1, 1]) * int(change)
            new_number = max(1, new_number)  # Ensure it's positive
            
            pattern = r'\b' + number + r'\b'
            modified = re.sub(pattern, str(new_number), sentence, count=1)
            return modified, f"The original text mentions '{number}', not '{new_number}'."
        
        # Could not create a false statement
        return None, ""


class FillInBlankGenerator(QuestionGenerator):
    """Generate fill-in-the-blank questions."""
    
    def __init__(self):
        super().__init__()
        self.question_type = 'fill_in_blank'
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sentence_text = sentence['text']
        doc = nlp(sentence_text)
        
        # Find potential words to blank out
        candidates = []
        
        for token in doc:
            # Look for important words (nouns, verbs, adjectives, numbers)
            if (token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM') and 
                not token.is_stop and 
                len(token.text) > 3):
                candidates.append((token.text, token.idx, token.idx + len(token.text)))
        
        if not candidates:
            return None  # No suitable words to blank out
        
        # Choose a random candidate
        word, start, end = random.choice(candidates)
        
        # Create the question with a blank
        question_text = sentence_text[:start] + "___________" + sentence_text[end:]
        
        return {
            'question': f"Fill in the blank: {question_text}",
            'answer': word,
            'explanation': f"The missing word is '{word}' as stated in the original text: \"{sentence_text}\"",
            'type': self.question_type,
            'source_sentence': sentence_text
        }


class DefinitionGenerator(QuestionGenerator):
    """Generate definition questions."""
    
    def __init__(self):
        super().__init__()
        self.question_type = 'definition'
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check if this sentence is a definition
        is_definition = sentence['suitability'].get('is_definition', False)
        
        # If not a definition but we have definitions in the section, use one of those
        if not is_definition:
            if not section['definitions']:
                return None  # No definitions available
            
            # Choose a random definition
            definition = random.choice(section['definitions'])
            term = definition['term']
            definition_text = definition['definition']
            
            # Create the question
            question_text = f"What is the definition of '{term}'?"
            answer = definition_text
            explanation = f"The term '{term}' is defined in the text as: \"{definition_text}\""
        else:
            # This sentence is itself a definition, try to extract the term and definition
            for definition in section['definitions']:
                if definition['definition'] == sentence['text']:
                    term = definition['term']
                    definition_text = definition['definition']
                    
                    # Create the question
                    question_text = f"Define the term '{term}'."
                    answer = definition_text
                    explanation = f"The term '{term}' is defined in the text as: \"{definition_text}\""
                    break
            else:
                return None  # Could not find this sentence in the definitions
        
        return {
            'question': question_text,
            'answer': answer,
            'explanation': explanation,
            'type': self.question_type,
            'source_sentence': sentence['text']
        }


class SyntaxUsageGenerator(QuestionGenerator):
    """Generate syntax usage questions."""
    
    def __init__(self):
        super().__init__()
        self.question_type = 'syntax_usage'
    
    def generate(self, sentence: Dict[str, Any], section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sentence_text = sentence['text']
        
        # Check if the sentence contains syntax-related terms
        syntax_terms = ['function', 'method', 'class', 'procedure', 'syntax', 'command', 'parameter', 'argument']
        contains_syntax = any(term in sentence_text.lower() for term in syntax_terms)
        
        if not contains_syntax:
            return None
        
        # Find code-like elements in the sentence
        code_pattern = r'`([^`]+)`|"([^"]+)"|\'([^\']+)\'|\b([\w\.]+\([\w\s,]*\))'
        matches = re.findall(code_pattern, sentence_text)
        
        code_elements = []
        for match in matches:
            for group in match:
                if group:
                    code_elements.append(group)
        
        if not code_elements:
            return None  # No code-like elements found
        
        # Choose a random code element
        code_element = random.choice(code_elements)
        
        # Create the question
        question_types = [
            f"What is the correct syntax for {code_element}?",
            f"How would you properly use {code_element} in code?",
            f"Which of the following illustrates the correct usage of {code_element}?"
        ]
        
        question_text = random.choice(question_types)
        
        return {
            'question': question_text,
            'answer': code_element,
            'explanation': f"The syntax '{code_element}' is explained in the text: \"{sentence_text}\"",
            'type': self.question_type,
            'source_sentence': sentence_text
        }


def generate_questions(analyzed_sections: List[Dict[str, Any]], 
                      question_types: List[str] = None,
                      count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate questions from analyzed document sections.
    
    Args:
        analyzed_sections: List of sections with analysis
        question_types: Types of questions to generate (if None, all types are used)
        count: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    # Initialize question generators
    generators = {
        'multiple_choice': MultipleChoiceGenerator(),
        'true_false': TrueFalseGenerator(),
        'fill_in_blank': FillInBlankGenerator(),
        'definition': DefinitionGenerator(),
        'syntax_usage': SyntaxUsageGenerator()
    }
    
    # Use specified question types or all available
    if question_types is None:
        question_types = list(generators.keys())
    
    # Filter to only use valid question types
    valid_types = [qt for qt in question_types if qt in generators]
    
    if not valid_types:
        return []  # No valid question types specified
    
    # Collect all question-worthy sentences from all sections
    all_sentences = []
    for section in analyzed_sections:
        for sentence in section.get('question_worthy_sentences', []):
            # Check if this sentence is suitable for any of our question types
            if any(qt in sentence.get('recommended_question_types', []) for qt in valid_types):
                all_sentences.append((sentence, section))
    
    if not all_sentences:
        return []  # No suitable sentences found
    
    # Shuffle sentences to get a mix of sections
    random.shuffle(all_sentences)
    
    # Generate questions
    questions = []
    attempts = 0
    max_attempts = min(len(all_sentences) * len(valid_types), count * 3)  # Limit attempts
    
    while len(questions) < count and attempts < max_attempts:
        # Choose a random sentence
        sentence, section = random.choice(all_sentences)
        
        # Filter question types to those suitable for this sentence
        suitable_types = [qt for qt in valid_types 
                         if qt in sentence.get('recommended_question_types', [])]
        
        if not suitable_types:
            attempts += 1
            continue
        
        # Choose a random question type from suitable types
        question_type = random.choice(suitable_types)
        
        # Try to generate a question
        try:
            question = generators[question_type].generate(sentence, section)
            if question and not any(q['question'] == question['question'] for q in questions):
                questions.append(question)
        except Exception as e:
            print(f"Error generating {question_type} question: {str(e)}")
        
        attempts += 1
    
    return questions[:count]  # Ensure we return exactly the requested number 