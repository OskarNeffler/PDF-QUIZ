import argparse
import os
from pdf_processor import extract_text_from_pdf
from text_analyzer import analyze_content
from model_inference import get_model_instance

def main():
    parser = argparse.ArgumentParser(description='Generate questions from a PDF file using a fine-tuned model')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--model_path', default='question_generation_model', help='Path to model directory')
    parser.add_argument('--count', type=int, default=10, help='Number of questions to generate')
    parser.add_argument('--output', help='Output file for generated questions (optional)')
    args = parser.parse_args()
    
    # Extract text from PDF
    print(f"Extracting text from {args.pdf}...")
    text_sections = extract_text_from_pdf(args.pdf)
    print(f"Extracted {len(text_sections)} sections.")
    
    # Analyze content
    print("Analyzing content...")
    analyzed_sections = analyze_content(text_sections)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = get_model_instance(args.model_path)
    
    # Check if model loaded successfully
    if not model.is_model_loaded():
        print("Error: Could not load the model. Exiting.")
        return
    
    print(f"Generating {args.count} questions...")
    questions = model.generate_questions_from_sections(
        analyzed_sections,
        count=args.count
    )
    
    # Print and save questions
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, q in enumerate(questions, 1):
                f.write(f"Question {i}: {q['question']}\n")
                f.write(f"Answer: {q['answer']}\n")
                if 'explanation' in q:
                    f.write(f"Explanation: {q['explanation']}\n")
                f.write("\n")
        print(f"Questions saved to {args.output}")
    else:
        for i, q in enumerate(questions, 1):
            print(f"Question {i}: {q['question']}")
            print(f"Answer: {q['answer']}")
            if 'explanation' in q:
                print(f"Explanation: {q['explanation']}")
            print()

if __name__ == "__main__":
    main() 