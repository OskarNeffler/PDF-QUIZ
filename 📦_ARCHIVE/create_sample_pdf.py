from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

def convert_text_to_pdf(input_file, output_file):
    """
    Convert a text file to a PDF document.
    """
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Create a PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Define styles for different heading levels
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18))
    styles.add(ParagraphStyle(name='Heading2Custom', parent=styles['Heading2'], fontSize=14))
    styles.add(ParagraphStyle(name='Heading3Custom', parent=styles['Heading3'], fontSize=12))
    
    # Create the content
    story = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            story.append(Spacer(1, 12))
            continue
        
        # Determine the style based on the line content
        if line.startswith('# '):
            # Level 1 heading (Title)
            text = line[2:]
            style = styles['MainTitle']
        elif line.startswith('## '):
            # Level 2 heading
            text = line[3:]
            style = styles['Heading2Custom']
        elif line.startswith('### '):
            # Level 3 heading
            text = line[4:]
            style = styles['Heading3Custom']
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point
            text = '• ' + line[2:]
            style = styles['Normal']
        elif line[0].isdigit() and line[1] == '.':
            # Numbered list
            text = line
            style = styles['Normal']
        else:
            # Normal paragraph
            text = line
            style = styles['Normal']
        
        paragraph = Paragraph(text, style)
        story.append(paragraph)
        
        # Add a small space after each paragraph
        story.append(Spacer(1, 6))
    
    # Build the PDF
    doc.build(story)
    print(f"PDF created successfully: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python create_sample_pdf.py input.txt output.pdf")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_text_to_pdf(input_file, output_file) 