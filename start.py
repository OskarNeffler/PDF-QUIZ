import argparse
import os
from pdf_processor import extract_text_from_pdf
from text_analyzer import analyze_content
from question_generator import generate_questions
from ui.interface import run_ui

# Lägg till import för modell-inference
from model_inference import get_model_instance

def main():
    parser = argparse.ArgumentParser(description='PDF-to-Quiz: Generate quiz questions from PDF documents')
    parser.add_argument('--pdf', help='Path to PDF file')
    parser.add_argument('--output', help='Output file for generated questions')
    parser.add_argument('--question_types', nargs='+', 
                        choices=['multiple_choice', 'true_false', 'fill_in_blank', 'definition', 'syntax_usage'],
                        default=['multiple_choice', 'true_false', 'fill_in_blank'],
                        help='Types of questions to generate')
    parser.add_argument('--count', type=int, default=10, help='Number of questions to generate')
    parser.add_argument('--ui', action='store_true', help='Launch the GUI instead of CLI')
    
    # Ny parameter för modellsökväg
    parser.add_argument('--model_path', help='Path to trained question generation model (optional)')
    
    # Ny parameter för att använda modellen för frågegeneration
    parser.add_argument('--use_model', action='store_true', help='Use trained model for question generation')
    
    args = parser.parse_args()
    
    if args.ui:
        run_ui()
        return
    
    if not args.pdf:
        print("Error: PDF file path is required when not using UI")
        return
    
    # Extract text from PDF
    print(f"Extracting text from {args.pdf}...")
    text_sections = extract_text_from_pdf(args.pdf)
    
    # Analyze content
    print("Analyzing content...")
    analyzed_sections = analyze_content(text_sections)
    
    # Generera frågor
    print(f"Generating {args.count} questions...")
    
    # Om modellsökväg angetts och use_model är aktiverad, använd modellen
    if args.use_model and args.model_path:
        print(f"Using trained model from {args.model_path}")
        model = get_model_instance(args.model_path)
        
        if model.is_model_loaded():
            questions = model.generate_questions_from_sections(
                analyzed_sections,
                count=args.count
            )
        else:
            print("Error: Could not load model, falling back to rule-based generation")
            questions = generate_questions(
                analyzed_sections, 
                question_types=args.question_types,
                count=args.count
            )
    else:
        # Använd vanlig regelbaserad generering
        questions = generate_questions(
            analyzed_sections, 
            question_types=args.question_types,
            count=args.count
        )
    
    # Output questions
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, q in enumerate(questions, 1):
                f.write(f"Question {i}: {q['question']}\n")
                if 'options' in q:
                    for j, option in enumerate(q['options']):
                        f.write(f"  {chr(65+j)}. {option}\n")
                f.write(f"Answer: {q['answer']}\n")
                if 'explanation' in q:
                    f.write(f"Explanation: {q['explanation']}\n")
                f.write("\n")
        print(f"Questions saved to {args.output}")
    else:
        for i, q in enumerate(questions, 1):
            print(f"Question {i}: {q['question']}")
            if 'options' in q:
                for j, option in enumerate(q['options']):
                    print(f"  {chr(65+j)}. {option}")
            print(f"Answer: {q['answer']}")
            if 'explanation' in q:
                print(f"Explanation: {q['explanation']}")
            print()

if __name__ == "__main__":
    main()
