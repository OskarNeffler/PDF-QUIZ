import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Any, Tuple
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


def analyze_content(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze document content to identify important concepts, entities, and structures.
    
    Args:
        sections: List of document sections from the PDF processor
        
    Returns:
        Enhanced sections with additional analysis
    """
    enhanced_sections = []
    stop_words = set(stopwords.words('english'))
    
    for section in sections:
        # Create enhanced section with original content
        enhanced = section.copy()
        
        # Extract sentences
        sentences = sent_tokenize(section['text'])
        enhanced['sentences'] = sentences
        
        # Use spaCy for entity recognition
        doc = nlp(section['text'])
        
        # Extract named entities
        entities = [{'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char} 
                   for ent in doc.ents]
        enhanced['entities'] = entities
        
        # Extract key terms using TF-IDF on paragraph level
        if len(section['paragraphs']) > 1:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf_matrix = vectorizer.fit_transform(section['paragraphs'])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms for each paragraph
            paragraph_terms = []
            for i, paragraph in enumerate(section['paragraphs']):
                if len(paragraph.strip()) > 0:
                    # Get scores for this paragraph
                    scores = tfidf_matrix[i].toarray()[0]
                    # Get indices of top scores
                    top_indices = scores.argsort()[-5:][::-1]  # Top 5 terms
                    # Get terms and scores
                    top_terms = [(feature_names[j], scores[j]) for j in top_indices if scores[j] > 0]
                    paragraph_terms.append(top_terms)
                else:
                    paragraph_terms.append([])
            
            enhanced['paragraph_key_terms'] = paragraph_terms
        else:
            enhanced['paragraph_key_terms'] = []
        
        # Identify definitions using patterns
        definitions = extract_definitions(section['text'])
        enhanced['definitions'] = definitions
        
        # Find potential question-worthy sentences
        enhanced['question_worthy_sentences'] = identify_question_worthy_sentences(
            sentences, entities, definitions
        )
        
        enhanced_sections.append(enhanced)
    
    return enhanced_sections


def extract_definitions(text: str) -> List[Dict[str, str]]:
    """
    Extract potential definitions from text using patterns.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of dictionaries with term and definition
    """
    definitions = []
    
    # Pattern 1: "Term is/are definition"
    pattern1 = re.compile(r'([A-Z][^.!?:]*?)\s+(?:is|are)\s+([^.!?]*[.!?])', re.DOTALL)
    matches = pattern1.findall(text)
    for term, definition in matches:
        if 3 < len(term.strip()) < 100 and len(definition.strip()) > 10:
            definitions.append({
                'term': term.strip(),
                'definition': definition.strip()
            })
    
    # Pattern 2: "Term: definition"
    pattern2 = re.compile(r'([A-Z][^.!?:]*?):\s+([^.!?]*[.!?])', re.DOTALL)
    matches = pattern2.findall(text)
    for term, definition in matches:
        if 3 < len(term.strip()) < 100 and len(definition.strip()) > 10:
            definitions.append({
                'term': term.strip(),
                'definition': definition.strip()
            })
    
    return definitions


def identify_question_worthy_sentences(
    sentences: List[str], 
    entities: List[Dict[str, Any]], 
    definitions: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Identify sentences that are good candidates for generating questions.
    
    Args:
        sentences: List of sentences from the text
        entities: Named entities found in the text
        definitions: Definitions found in the text
        
    Returns:
        List of dictionaries with sentence information and question type suitability
    """
    worthy_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence_info = {
            'text': sentence,
            'index': i,
            'suitability': {}
        }
        
        # Check length (not too short, not too long)
        length = len(sentence.split())
        if 5 <= length <= 40:
            sentence_info['suitability']['length'] = 'good'
        elif length < 5:
            sentence_info['suitability']['length'] = 'too_short'
        else:
            sentence_info['suitability']['length'] = 'too_long'
        
        # Check if contains named entities
        has_entities = any(entity['text'] in sentence for entity in entities)
        sentence_info['suitability']['has_entities'] = has_entities
        
        # Check if part of a definition
        is_definition = any(definition['definition'] == sentence for definition in definitions)
        sentence_info['suitability']['is_definition'] = is_definition
        
        # Check for numerical data (good for multiple choice)
        has_numbers = bool(re.search(r'\d+', sentence))
        sentence_info['suitability']['has_numbers'] = has_numbers
        
        # Determine best question types for this sentence
        question_types = []
        
        if has_entities:
            question_types.append('multiple_choice')
        
        if is_definition:
            question_types.append('definition')
        
        if has_numbers:
            question_types.append('true_false')
        
        if re.search(r'\b(is|are|was|were|has|have|had|can|could|should|would|will)\b', sentence):
            question_types.append('true_false')
        
        if length > 15:
            question_types.append('fill_in_blank')
        
        if re.search(r'\b(function|method|class|procedure|syntax|command|parameter|argument)\b', sentence.lower()):
            question_types.append('syntax_usage')
        
        sentence_info['recommended_question_types'] = question_types
        
        # Only add if suitable for at least one question type
        if question_types:
            worthy_sentences.append(sentence_info)
    
    return worthy_sentences 