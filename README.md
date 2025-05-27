# 🎯 PDF-to-Quiz Helper

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![T5 Model](https://img.shields.io/badge/model-T5-orange.svg)](https://huggingface.co/transformers/model_doc/t5.html)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI-black.svg)](https://openai.com/)

> 🚀 **Transform any PDF into interactive quiz questions with AI-powered question generation!**

An intelligent PDF-to-Quiz system that leverages **T5 transformers** and **OpenAI GPT** to automatically extract text from PDFs and generate high-quality quiz questions. Perfect for students, educators, and anyone who wants to create study materials from documents.

## ✨ Features

### 🎨 **Modern Web Interface**
- **Split-screen PDF viewer** - Read documents while answering questions
- **Toggle-able PDF panel** - Show/hide PDF with smooth animations
- **Responsive design** - Works on desktop and mobile
- **Real-time progress tracking** - See your quiz progress live

### 🤖 **AI-Powered Question Generation**
- **T5 Transformer Model** - Fine-tuned for Swedish ML course content
- **OpenAI Integration** - GPT-powered answer generation and alternatives
- **Hybrid AI System** - Combines T5 questions with OpenAI explanations
- **Multiple Question Types**:
  - 📝 Multiple Choice Questions
  - 🔤 Fill-in-the-blank
  - 💬 Q&A Style Questions
  - 🎯 AI-generated with alternatives

### 📄 **Smart PDF Processing**
- **Automatic text extraction** - Support for complex PDF layouts
- **Intelligent chunking** - Breaks down large documents efficiently
- **PDF caching** - Faster processing on repeated use
- **Auto PDF detection** - Automatically finds matching PDFs for text files

### 🎯 **Interactive Quiz Experience**
- **Immediate feedback** - Get answers right away
- **Score tracking** - See your performance metrics
- **Question navigation** - Move forward/backward through questions
- **Answer explanations** - Understand why answers are correct

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip package manager
- OpenAI API key (optional, for hybrid mode)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd QUIZ_HELPER
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up OpenAI (Optional)**
```bash
# Create openai_system/openai_config.env
echo "OPENAI_API_KEY=your_api_key_here" > openai_system/openai_config.env
```

5. **Start the server**
```bash
python web_interface.py
```

6. **Open your browser**
```
http://localhost:5001
```

## 📖 Usage Guide

### 1. **Upload & Process PDFs**
- **Smart Processing**: Upload PDFs for intelligent text extraction
- **Direct Processing**: Quick processing for simple documents
- **Text Files**: Use pre-extracted text files from `extracted_texts/`

### 2. **Load AI Model**
- Click "Load Model" to initialize the T5 transformer
- Model loads from `models/trained_model_best/`
- First load takes ~30 seconds, subsequent loads are faster

### 3. **Generate Quiz Questions**
Choose your question type:
- **🤖 AI Questions**: T5-generated questions with quality filtering
- **📝 Fill-in-blank**: Automatically create fill-in-the-blank questions
- **💬 Q&A Style**: Open-ended questions and answers
- **🎯 Hybrid**: Best of T5 + OpenAI (requires API key)

### 4. **Take Interactive Quiz**
- Questions appear with **PDF viewer on the side**
- Toggle PDF visibility with the "📖 Show/Hide PDF" button
- Answer questions and get immediate feedback
- Navigate through questions at your own pace

### 5. **View Results**
- See your score and detailed breakdown
- Review correct answers and explanations
- Start new quiz sessions anytime

## 📁 Project Structure

```
QUIZ_HELPER/
├── 📄 web_interface.py          # Main Flask web application
├── 🤖 model_quiz_generator.py   # T5 model and question generation
├── 📊 smart_pdf_processor.py    # Intelligent PDF processing
├── 🔧 enhanced_quiz_system.py   # Advanced quiz generation
├── 📑 direct_pdf_extractor.py   # Simple PDF text extraction
├── 📄 pdf_to_text.py           # PDF processing utilities
├── 📁 models/                   # T5 model files
│   └── trained_model_best/     # Fine-tuned T5 model
├── 📁 openai_system/           # OpenAI integration
├── 📁 extracted_texts/         # Processed text files
├── 📁 best_cheatsheets/        # Premium PDF collection
├── 📁 cheatsheets/             # Study material PDFs
├── 📁 uploads/                 # User uploaded files
├── 📄 smart_pdf_cache.json     # Smart processing cache
├── 📄 pdf_cache.json          # PDF processing cache
└── 📄 setup_openai.txt        # OpenAI setup instructions
```

## 🛠️ Technical Stack

### **Backend**
- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face T5 model
- **PyMuPDF** - PDF processing
- **OpenAI API** - GPT integration

### **Frontend**
- **HTML5/CSS3** - Modern responsive design
- **JavaScript** - Interactive functionality
- **Local Storage** - State persistence

### **AI/ML**
- **T5 (Text-to-Text Transfer Transformer)** - Question generation
- **SQuAD-trained models** - Question-answer understanding
- **OpenAI GPT** - Answer generation and alternatives

## 🎯 Advanced Features

### **Automatic PDF Detection**
The system automatically searches for matching PDFs in:
- `best_cheatsheets/` - High-quality study materials
- `cheatsheets/` - General study PDFs  
- `uploads/` - User uploaded files

Example: `keras.txt` → `best_cheatsheets/keras.pdf`

### **Smart PDF Processing**
- **Hybrid extraction** - Combines multiple extraction methods
- **Layout preservation** - Maintains document structure
- **Caching system** - Stores processed results for faster access
- **Error handling** - Graceful fallbacks for problematic PDFs

### **Question Quality Filtering**
- **Confidence scoring** - Only shows high-quality questions
- **Duplicate detection** - Removes similar questions
- **Length filtering** - Ensures appropriate question length
- **Answer validation** - Verifies answer quality

## 🎨 Screenshots

### Main Interface
```
🌐 PDF-to-Quiz Helper
┌─────────────────────────────────────────────────────────────┐
│ 📁 Load Model  📄 Process PDF  🎯 Generate Quiz             │
├─────────────────────────────────────────────────────────────┤
│ 📊 Available Texts:                                         │
│ ☑️ keras.txt (6266 chars)                                   │
│ ☑️ python_basics_cheat_sheet.txt (1402 chars)              │
│ ☑️ Attention_is_all_you_need_v7_SMART.txt (39542 chars)    │
└─────────────────────────────────────────────────────────────┘
```

### Quiz Interface
```
┌─────────────────────┬───────────────────────────────────────┐
│ 📖 PDF Viewer       │ ❓ Quiz Question                      │
│ [Toggle: Hide PDF]  │ Question 1 of 5                      │
├─────────────────────┤                                       │
│                     │ What is the main innovation of the    │
│   📄 Document       │ Transformer architecture?            │
│   Content Here      │                                       │
│                     │ ○ A) Recurrent connections           │
│                     │ ○ B) Attention mechanism             │
│                     │ ○ C) Convolutional layers           │
│                     │ ○ D) LSTM cells                      │
│                     │                                       │
│                     │ [Submit Answer] [Next Question]       │
└─────────────────────┴───────────────────────────────────────┘
```

## 📊 Performance

### **Question Generation Speed**
- **T5 Model**: ~2-3 seconds per question
- **Hybrid Mode**: ~5-8 seconds per question (with OpenAI)
- **Batch Processing**: 5 questions in ~15-30 seconds

### **PDF Processing**
- **Simple PDFs**: 1-3 seconds
- **Complex PDFs**: 5-15 seconds  
- **Cached Results**: <1 second

### **Memory Usage**
- **T5 Model**: ~2GB RAM
- **Web Interface**: ~100MB RAM
- **Total System**: ~2.5GB RAM recommended

## 🔧 Configuration

### **Model Settings**
```python
# In model_quiz_generator.py
MODEL_PATH = "models/trained_model_best"
DEVICE = "cpu"  # or "cuda" for GPU
MAX_LENGTH = 512
CHUNK_SIZE = 500
```

### **Web Interface**
```python
# In web_interface.py
PORT = 5001
DEBUG = True
SECRET_KEY = "quiz_generator_secret_key_2024"
```

### **OpenAI Integration**
```env
# In openai_system/openai_config.env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
MAX_TOKENS=150
```

## 🐛 Troubleshooting

### **Common Issues**

**🔴 Port 5001 already in use**
```bash
# Kill existing processes
pkill -f "python web_interface.py"
# Or use different port by editing web_interface.py
```

**🔴 Model loading fails**
```bash
# Check if model exists
ls -la models/trained_model_best/
# Verify model files are complete
```

**🔴 OpenAI API errors**
```bash
# Check API key in openai_system/openai_config.env
# Verify API key has sufficient credits
```

**🔴 PDF processing fails**
```bash
# Install additional dependencies
pip install pymupdf4llm
# Try direct processing instead of smart processing
```

### **Performance Optimization**

**🚀 Faster question generation**
- Use GPU if available: Set `DEVICE = "cuda"`
- Reduce `num_questions` for faster processing
- Use cached PDFs when possible

**🚀 Lower memory usage**
- Close other applications
- Use smaller `CHUNK_SIZE`
- Process one document at a time

## 🎓 Educational Use Cases

### **For Students**
- 📚 **Study Review**: Convert lecture PDFs into practice questions
- 🧠 **Self-Assessment**: Test knowledge before exams
- 📝 **Note Taking**: Generate questions while reading
- 🎯 **Exam Prep**: Create custom quiz sessions

### **For Educators**  
- 👩‍🏫 **Question Banks**: Generate questions from course materials
- 📊 **Assessment Creation**: Quick quiz generation for classes
- 🔄 **Content Review**: Verify important concepts are covered
- 🎨 **Interactive Learning**: Engage students with AI-generated content

### **For Researchers**
- 📄 **Paper Review**: Extract key concepts from research papers
- 🔍 **Literature Analysis**: Generate questions about methodologies
- 📚 **Knowledge Extraction**: Convert dense texts into digestible Q&As

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Development Setup**
```bash
git clone <repo-url>
cd QUIZ_HELPER
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Areas for Contribution**
- 🎨 **UI/UX Improvements**: Better responsive design
- 🤖 **AI Enhancements**: Better question generation algorithms  
- 📄 **PDF Processing**: Support for more document types
- 🌍 **Internationalization**: Multi-language support
- 🔧 **Performance**: Optimization and caching improvements

### **Pull Request Process**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Roadmap

### **Version 2.0 (Coming Soon)**
- 🎥 **Video Processing**: Extract quiz questions from video lectures
- 🌐 **Multi-language**: Support for multiple languages
- 👥 **Collaborative**: Share quizzes with others
- 📱 **Mobile App**: Native iOS/Android applications

### **Version 2.1**
- 🧮 **Analytics Dashboard**: Detailed performance metrics
- 🎮 **Gamification**: Points, badges, and leaderboards
- 🔗 **API Access**: RESTful API for integration
- ☁️ **Cloud Deployment**: Hosted solution option

## 🙏 Acknowledgments

- **Hugging Face** - For the excellent Transformers library
- **OpenAI** - For GPT API integration
- **PyMuPDF** - For robust PDF processing
- **Flask Community** - For the amazing web framework
- **Swedish ML Course** - For providing training data and use cases

## 📧 Contact & Support

- **🐛 Bug Reports**: Open an issue on GitHub
- **💡 Feature Requests**: Start a discussion
- **📧 General Questions**: Contact the maintainers
- **📚 Documentation**: Check the Wiki for detailed guides

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the ML learning community**

*Transform your PDFs into interactive learning experiences today!* 