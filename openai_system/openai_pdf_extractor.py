#!/usr/bin/env python3
"""
🔥 OpenAI PDF Extractor
Clean PDF text extraction using GPT-4 Vision API
Simple setup - just API key needed!
"""
import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import fitz  # PyMuPDF for PDF to image conversion
from openai import OpenAI

class OpenAIPDFExtractor:
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self.setup_success = False
        
        # Settings from .env
        self.default_model = os.getenv('DEFAULT_MODEL', 'gpt-4o')
        self.backup_model = os.getenv('BACKUP_MODEL', 'gpt-4o-mini')
        self.max_cost = float(os.getenv('MAX_COST_PER_SESSION', '1.00'))
        self.warn_threshold = float(os.getenv('WARN_COST_THRESHOLD', '0.50'))
        
        self.current_cost = 0.0
        
        if self.api_key:
            self._setup_openai()
        else:
            print("❌ No OpenAI API key provided")
            self._show_api_key_help()
    
    def _setup_openai(self):
        """Setup OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            # Test connection
            self.client.models.list()
            self.setup_success = True
            print(f"✅ OpenAI configured (API key only)")
            print(f"💰 Budget: ${self.max_cost} per session")
        except Exception as e:
            print(f"❌ OpenAI setup failed: {e}")
            self._show_api_key_help()
    
    def _show_api_key_help(self):
        """Show how to get API key"""
        print("\n" + "="*50)
        print("🔑 OPENAI API KEY NEEDED")
        print("="*50)
        print("\n📋 Get API key:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Click 'Create new secret key'")
        print("3. Set restrictions (budget limits)")
        print("4. Add to .env: OPENAI_API_KEY='your-key-here'")
        print("\n✅ Simple setup - just API key!")
        print("="*50 + "\n")
    
    def _pdf_to_images(self, pdf_path: str, max_pages: int = 5) -> list:
        """Convert PDF pages to base64 images"""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            pages_to_process = min(len(doc), max_pages)
            
            print(f"📄 Converting {pages_to_process} pages to images...")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                # Good resolution for OCR
                mat = fitz.Matrix(2.0, 2.0)  
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                images.append(img_base64)
            
            doc.close()
            print(f"✅ Converted {len(images)} pages")
            return images
            
        except Exception as e:
            print(f"❌ PDF conversion error: {e}")
            return []
    
    def _estimate_cost(self, num_images: int, estimated_output_words: int = 500) -> float:
        """Estimate API cost"""
        # GPT-4o pricing (approximate)
        cost_per_image = 0.01  # ~$0.01 per image
        cost_per_1k_output_tokens = 0.03  # ~$0.03 per 1K output tokens
        
        image_cost = num_images * cost_per_image
        text_cost = (estimated_output_words / 1000) * cost_per_1k_output_tokens
        
        return image_cost + text_cost
    
    def _check_budget(self, estimated_cost: float) -> bool:
        """Check if we're within budget"""
        total_cost = self.current_cost + estimated_cost
        
        if total_cost > self.max_cost:
            print(f"💰 Budget exceeded! Current: ${self.current_cost:.3f}, Estimated: ${estimated_cost:.3f}, Max: ${self.max_cost}")
            return False
        
        if total_cost > self.warn_threshold:
            print(f"⚠️  Cost warning: ${total_cost:.3f} (limit: ${self.max_cost})")
        
        return True
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF using GPT-4 Vision"""
        if not self.setup_success:
            return None
        
        try:
            print(f"📖 Processing {Path(pdf_path).name} with OpenAI...")
            
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path)
            if not images:
                return None
            
            # Check budget
            estimated_cost = self._estimate_cost(len(images))
            if not self._check_budget(estimated_cost):
                return None
            
            # Prepare images for API
            image_messages = []
            for i, img_base64 in enumerate(images):
                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            # Create prompt
            prompt = """Extract all text content from these PDF pages.

Return clean, readable text that captures:
- Main content and key information
- Technical details and facts
- Important statements and concepts

Skip headers, footers, page numbers, and navigation elements.
Format as natural, flowing text suitable for reading and quiz generation.

Combine all pages into one coherent text."""

            # Make API call
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_messages
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            # Update cost tracking
            self.current_cost += estimated_cost
            
            text = response.choices[0].message.content
            if text:
                print(f"✅ OpenAI extracted {len(text)} characters")
                print(f"💰 Session cost: ${self.current_cost:.3f}")
                return text
            else:
                print("❌ No text extracted")
                return None
                
        except Exception as e:
            print(f"❌ OpenAI extraction error: {e}")
            return None
    
    def extract_for_quiz(self, pdf_path: str) -> Optional[str]:
        """Extract text optimized for quiz generation"""
        if not self.setup_success:
            return None
        
        try:
            images = self._pdf_to_images(pdf_path)
            if not images:
                return None
            
            estimated_cost = self._estimate_cost(len(images))
            if not self._check_budget(estimated_cost):
                return None
            
            image_messages = []
            for img_base64 in images:
                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            prompt = """Extract content from these PDF pages for creating quiz questions.

Focus on:
- Key facts and important information
- Technical concepts and definitions
- Clear, factual statements
- Educational content

Return well-formed sentences that would be good for true/false questions.
Skip metadata, headers, footers, and page numbers.
Each sentence should be complete and factually verifiable.

Format as clear, separate statements."""

            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_messages
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            self.current_cost += estimated_cost
            
            text = response.choices[0].message.content
            if text:
                print(f"✅ Quiz-optimized text: {len(text)} characters")
                print(f"💰 Session cost: ${self.current_cost:.3f}")
                return text
            else:
                return None
                
        except Exception as e:
            print(f"❌ Quiz extraction error: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test if OpenAI API is working"""
        if not self.setup_success:
            return False
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper for testing
                messages=[{"role": "user", "content": "Test: What is 2+2?"}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                print(f"✅ API test successful: {response.choices[0].message.content[:50]}...")
                return True
            return False
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def get_session_cost(self) -> float:
        """Get current session cost"""
        return self.current_cost
    
    def reset_session_cost(self):
        """Reset session cost counter"""
        self.current_cost = 0.0 