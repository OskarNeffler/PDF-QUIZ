#!/usr/bin/env python3
"""
🚀 Super Simple PDF Test
Only needs API key - no gcloud, no service account!
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add gemini system to path
sys.path.append(str(Path(__file__).parent / "gemini_system"))

def main():
    print("🚀 Super Simple PDF Test (API Key Only)")
    print("=" * 45)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ No GEMINI_API_KEY found in .env")
        print("\n📋 Quick setup:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Create API key")
        print("3. Add to .env: GEMINI_API_KEY='your-key'")
        print("4. Run this script again")
        return
    
    # Test API
    print(f"\n🔑 API Key: {api_key[:10]}...{api_key[-5:]}")
    
    try:
        from simple_gemini_extractor import SimpleGeminiExtractor
        
        print("\n1️⃣ Testing Gemini connection...")
        extractor = SimpleGeminiExtractor()
        
        if not extractor.test_connection():
            print("❌ API connection failed")
            return
        
        # Find PDFs
        print("\n2️⃣ Finding PDFs...")
        pdf_paths = []
        for search_dir in ['best_cheatsheets', 'cheatsheets', '.']:
            if Path(search_dir).exists():
                pdfs = list(Path(search_dir).glob('*.pdf'))
                pdf_paths.extend(pdfs[:1])  # Max 1 per directory
        
        pdf_paths = pdf_paths[:2]  # Max 2 total to save costs
        
        if not pdf_paths:
            print("❌ No PDFs found")
            print("💡 Add some PDFs to test with")
            return
        
        print(f"📚 Found {len(pdf_paths)} PDFs:")
        for pdf in pdf_paths:
            print(f"   - {pdf}")
        
        # Test extraction
        print("\n3️⃣ Testing PDF extraction...")
        for pdf in pdf_paths:
            print(f"\n📖 Processing {pdf.name}...")
            text = extractor.extract_text(str(pdf))
            
            if text:
                print(f"✅ Success: {len(text)} characters")
                print(f"📝 Sample: {text[:150]}...")
            else:
                print("❌ Failed")
            
            print("-" * 30)
        
        print("\n✅ Test completed!")
        print("💡 If this works, you're ready to build quiz generators!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have: pip install google-generativeai")

if __name__ == "__main__":
    main() 