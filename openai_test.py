#!/usr/bin/env python3
"""
🚀 OpenAI PDF Test
Simple test using restricted OpenAI API key
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add openai system to path
sys.path.append(str(Path(__file__).parent / "openai_system"))

def main():
    print("🚀 OpenAI PDF Test (Restricted API Key)")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ No OPENAI_API_KEY found in .env")
        print("\n📋 Quick setup:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Create restricted key")
        print("3. Add to .env: OPENAI_API_KEY='your-key'")
        return
    
    # Check budget settings
    max_cost = float(os.getenv('MAX_COST_PER_SESSION', '1.00'))
    warn_threshold = float(os.getenv('WARN_COST_THRESHOLD', '0.50'))
    
    print(f"\n🔑 API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"💰 Budget: ${max_cost} per session (warn at ${warn_threshold})")
    
    try:
        from openai_pdf_extractor import OpenAIPDFExtractor
        
        print("\n1️⃣ Testing OpenAI connection...")
        extractor = OpenAIPDFExtractor()
        
        if not extractor.test_connection():
            print("❌ API connection failed")
            return
        
        # Find PDFs (limited by MAX_PDFS_PER_TEST)
        max_pdfs = int(os.getenv('MAX_PDFS_PER_TEST', '2'))
        print(f"\n2️⃣ Finding PDFs (max {max_pdfs})...")
        
        pdf_paths = []
        for search_dir in ['best_cheatsheets', 'cheatsheets', '.']:
            if Path(search_dir).exists():
                pdfs = list(Path(search_dir).glob('*.pdf'))
                pdf_paths.extend(pdfs[:1])  # Max 1 per directory
        
        pdf_paths = pdf_paths[:max_pdfs]  # Respect MAX_PDFS_PER_TEST
        
        if not pdf_paths:
            print("❌ No PDFs found")
            print("💡 Add some PDFs to test with")
            return
        
        print(f"📚 Found {len(pdf_paths)} PDFs:")
        for pdf in pdf_paths:
            print(f"   - {pdf}")
        
        # Test extraction
        print("\n3️⃣ Testing PDF extraction...")
        for i, pdf in enumerate(pdf_paths):
            print(f"\n📖 Processing {pdf.name} ({i+1}/{len(pdf_paths)})...")
            
            # Check remaining budget
            remaining_budget = max_cost - extractor.get_session_cost()
            if remaining_budget <= 0:
                print(f"💰 Budget exhausted (${extractor.get_session_cost():.3f})")
                break
            
            text = extractor.extract_text(str(pdf))
            
            if text:
                print(f"✅ Success: {len(text)} characters")
                print(f"📝 Sample: {text[:150]}...")
            else:
                print("❌ Failed")
            
            print("-" * 40)
        
        print(f"\n✅ Test completed!")
        print(f"💰 Total cost: ${extractor.get_session_cost():.3f}")
        print("💡 Ready for quiz generation!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "pip install" not in str(e):
            print("\n💡 Make sure you have:")
            print("pip install openai python-dotenv PyMuPDF")

if __name__ == "__main__":
    main() 