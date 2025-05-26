#!/usr/bin/env python3
"""
🎯 Interactive Quiz Generator - Flask Web Interface
Play quiz questions one by one with immediate feedback
"""
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, session
import os
import sys
import subprocess
import json
import random
from pathlib import Path
from model_quiz_generator import ModelQuizGenerator, load_extracted_texts

app = Flask(__name__)
app.secret_key = 'quiz_generator_secret_key_2024'

# Global variables
quiz_generator = None
model_loaded = False

# HTML Template for main page
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Quiz Generator</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
        }
        .section h3 {
            color: #444;
            margin-top: 0;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .success {
            background: #28a745;
        }
        .warning {
            background: #ffc107;
            color: black;
        }
        input[type="text"], input[type="number"], select {
            padding: 8px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 14px;
            width: 300px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .quiz-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .text-info {
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #007bff;
        }
        
        /* PDF Upload Styles */
        .pdf-upload {
            border: 2px dashed #007bff;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            margin: 20px 0;
            transition: all 0.3s ease;
        }
        .pdf-upload.dragover {
            border-color: #28a745;
            background: #f0fff0;
        }
        .pdf-upload input[type="file"] {
            display: none;
        }
        .upload-text {
            font-size: 18px;
            color: #666;
            margin: 10px 0;
        }
        .upload-button {
            background: #28a745;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        .upload-button:hover {
            background: #218838;
        }
        
        /* Content Type Buttons */
        .content-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .content-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s ease;
            text-align: left;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .content-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .content-button.summary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .content-button.hybrid {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .content-button.quiz {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }
        .content-button.qa {
            background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%);
        }
        .content-button-desc {
            font-size: 12px;
            opacity: 0.9;
            margin-top: 5px;
        }
        
        /* Summary Length Options */
        .summary-options {
            display: none;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }
        .summary-length-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .summary-length-btn:hover {
            background: #5a6268;
        }
        .summary-length-btn.selected {
            background: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Interactive Quiz Generator</h1>
        
        {% if status %}
        <div class="status {{ status.type }}">
            {{ status.message }}
        </div>
        {% endif %}
        
        <!-- Model Section -->
        <div class="section">
            <h3>🤖 Model Status: {% if model_loaded %}✅ Loaded{% else %}❌ Not Loaded{% endif %}</h3>
            <form method="POST" action="/load_model">
                <label>Model Path:</label><br>
                <input type="text" name="model_path" value="{{ model_path }}" style="width: 400px;">
                <button type="submit" {% if model_loaded %}class="success"{% endif %}>
                    {% if model_loaded %}✅ Reload Model{% else %}🔄 Load Model{% endif %}
                </button>
            </form>
        </div>
        
        <!-- PDF Processing Section -->
        <div class="section">
            <h3>📄 Smart PDF Processor</h3>
            
            <div class="pdf-upload" onclick="document.getElementById('pdf-file').click()" 
                 ondrop="handleDrop(event)" 
                 ondragover="handleDragOver(event)" 
                 ondragleave="handleDragLeave(event)">
                <div class="upload-text">
                    <strong>🔀 Drag & Drop PDF here or click to select</strong><br>
                    <small>PyMuPDF extraction + AI cleanup (when API available)</small>
                </div>
                <input type="file" id="pdf-file" accept=".pdf" onchange="handleFileSelect(event)">
                <button type="button" class="upload-button">📁 Choose PDF File</button>
            </div>
            
            <div style="text-align: center; margin: 15px 0;">
                <strong>OR</strong>
            </div>
            
            <div style="text-align: center;">
                <form method="POST" action="/process_smart_pdf" style="display: inline;">
                    <button type="submit" style="background-color: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                        🚀 Process "Attention is All You Need" PDF
                    </button>
                </form>
                <p style="color: #666; font-size: 14px; margin-top: 10px;">
                    ⚡ Quick test with the included research paper
                </p>
            </div>
        </div>
        
        <!-- Content Generation Section -->
        <div class="section">
            <h3>🎲 Generate Content</h3>
            
            {% if extracted_texts %}
            <div class="quiz-info">
                <strong>📚 Available texts for processing:</strong><br>
                {% for text_name, text_content in extracted_texts.items() %}
                <div class="text-info">
                    📄 <strong>{{ text_name }}</strong> - {{ text_content|length|round|int }} characters
                </div>
                {% endfor %}
            </div>
            
            <div style="margin: 20px 0;">
                <label><strong>Select Text:</strong></label><br>
                <select id="text-selection" style="width: 400px;">
                    <option value="all">📚 All available texts</option>
                    {% for text_name in extracted_texts.keys() %}
                    <option value="{{ text_name }}">📄 Only: {{ text_name }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="content-buttons">
                <button class="content-button summary" onclick="selectContentType('summary')">
                    📝 Text Summarization
                    <div class="content-button-desc">Single comprehensive summary using T5 model</div>
                </button>
                
                <button class="content-button quiz" onclick="selectContentType('multiple_choice')">
                    🔤 Multiple Choice Quiz
                    <div class="content-button-desc">Interactive quiz with multiple choice questions</div>
                </button>
                
                <button class="content-button qa" onclick="selectContentType('qa')">
                    💬 Ask T5 Model
                    <div class="content-button-desc">Ask questions directly to your T5 model</div>
                </button>
            </div>
            
            <!-- Summary Length Options (hidden by default) -->
            <div id="summary-options" class="summary-options">
                <button type="button" class="summary-length-btn" data-length="short">📋 Short Summary</button>
                <button type="button" class="summary-length-btn selected" data-length="medium">📄 Medium Summary</button>
                <button type="button" class="summary-length-btn" data-length="long">📚 Long Summary</button>
            </div>
            
            <!-- Hidden form -->
            <form id="content-form" method="POST" action="/start_quiz" style="display: none;">
                <input type="hidden" id="question_type" name="question_type" value="">
                <input type="hidden" id="text_selection_input" name="text_selection" value="all">
                <input type="hidden" id="num_questions" name="num_questions" value="5">
                <input type="hidden" id="summary_length" name="summary_length" value="medium">
            </form>
            
            {% else %}
            <p style="color: #666; margin-top: 15px;">
                ℹ️ No extracted texts found. Process some PDFs first to create quiz questions.
            </p>
            {% endif %}
        </div>
        
        <!-- System Output -->
        <div class="section">
            <h3>📋 System Output</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; max-height: 200px; overflow-y: auto;">
                {{ output or "Ready to begin..." }}
            </div>
        </div>
    </div>
    
    <script>
        let selectedContentType = '';
        let selectedSummaryLength = 'medium';
        
        function selectContentType(type) {
            selectedContentType = type;
            document.getElementById('question_type').value = type;
            
            // Show/hide summary options
            const summaryOptions = document.getElementById('summary-options');
            if (type === 'summary') {
                summaryOptions.style.display = 'grid';
            } else {
                summaryOptions.style.display = 'none';
                
                // Handle chatbot (qa) type specially
                if (type === 'qa') {
                    showChatbot();
                    return; // Don't submit form for chatbot
                } else {
                    // Set default number of questions for other types
                    document.getElementById('num_questions').value = '5';
                    
                    // Submit immediately for non-summary, non-chatbot types
                    document.getElementById('text_selection_input').value = document.getElementById('text-selection').value;
                    document.getElementById('content-form').submit();
                }
            }
        }
        
        function showChatbot() {
            // Hide the content selection area and show chatbot interface
            const contentSection = document.querySelector('.section:nth-child(3)');
            contentSection.innerHTML = `
                <h3>💬 Chat with T5 Model</h3>
                <p>Ask your T5 model any question directly!</p>
                
                <div style="margin: 20px 0;">
                    <input type="text" id="chatbot-question" placeholder="Type your question here..." 
                           style="width: 70%; padding: 12px; font-size: 16px;" onkeypress="handleChatKeyPress(event)">
                    <button onclick="askT5Question()" style="padding: 12px 20px; font-size: 16px; margin-left: 10px;">Ask 💬</button>
                </div>
                
                <div id="chat-history" style="background: #f8f9fa; padding: 20px; border-radius: 10px; min-height: 200px; max-height: 400px; overflow-y: auto; border: 1px solid #ddd;">
                    <p style="color: #666; text-align: center; margin: 50px 0;">Ask me anything! I'll do my best to answer using my T5 knowledge.</p>
                </div>
                
                <div style="margin-top: 15px;">
                    <button onclick="location.reload()" style="background: #6c757d;">🔄 Back to Main Menu</button>
                </div>
            `;
            
            // Focus on the input field
            setTimeout(() => {
                document.getElementById('chatbot-question').focus();
            }, 100);
        }
        
        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                askT5Question();
            }
        }
        
        function askT5Question() {
            const questionInput = document.getElementById('chatbot-question');
            const question = questionInput.value.trim();
            
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            // Clear input and show loading
            questionInput.value = '';
            const chatHistory = document.getElementById('chat-history');
            
            // Add user question to chat
            chatHistory.innerHTML += `
                <div style="margin: 10px 0; text-align: right;">
                    <div style="background: #007bff; color: white; padding: 10px 15px; border-radius: 15px 15px 5px 15px; display: inline-block; max-width: 70%;">
                        <strong>Du:</strong> ${question}
                    </div>
                </div>
            `;
            
            // Add loading indicator
            chatHistory.innerHTML += `
                <div id="loading-message" style="margin: 10px 0;">
                    <div style="background: #e9ecef; color: #666; padding: 10px 15px; border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 70%;">
                        <strong>T5:</strong> Thinking... 🤔
                    </div>
                </div>
            `;
            
            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;
            
            // Send question to backend
            fetch('/ask_t5', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `user_question=${encodeURIComponent(question)}`
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                document.getElementById('loading-message').remove();
                
                if (data.success) {
                    // Add T5 response to chat
                    chatHistory.innerHTML += `
                        <div style="margin: 10px 0;">
                            <div style="background: #28a745; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 70%;">
                                <strong>T5:</strong> ${data.answer}
                            </div>
                        </div>
                    `;
                } else {
                    // Add error message to chat
                    chatHistory.innerHTML += `
                        <div style="margin: 10px 0;">
                            <div style="background: #dc3545; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 70%;">
                                <strong>Error:</strong> ${data.error}
                            </div>
                        </div>
                    `;
                }
                
                // Scroll to bottom
                chatHistory.scrollTop = chatHistory.scrollHeight;
            })
            .catch(error => {
                // Remove loading message
                const loadingMsg = document.getElementById('loading-message');
                if (loadingMsg) loadingMsg.remove();
                
                // Add error message to chat
                chatHistory.innerHTML += `
                    <div style="margin: 10px 0;">
                        <div style="background: #dc3545; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 70%;">
                            <strong>Error:</strong> Failed to get response from T5 model.
                        </div>
                    </div>
                `;
                
                // Scroll to bottom
                chatHistory.scrollTop = chatHistory.scrollHeight;
            });
        }
        
        // Handle summary length selection
        document.querySelectorAll('.summary-length-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                // Remove selected class from all buttons
                document.querySelectorAll('.summary-length-btn').forEach(b => b.classList.remove('selected'));
                // Add selected class to clicked button
                this.classList.add('selected');
                
                selectedSummaryLength = this.dataset.length;
                document.getElementById('summary_length').value = selectedSummaryLength;
                document.getElementById('num_questions').value = '1'; // Only 1 summary
                
                // Submit form
                document.getElementById('text_selection_input').value = document.getElementById('text-selection').value;
                document.getElementById('content-form').submit();
            });
        });
        
        // PDF Upload Functions
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                uploadPDF(files[0]);
            } else {
                alert('Please drop a PDF file.');
            }
        }
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file && file.type === 'application/pdf') {
                uploadPDF(file);
            }
        }
        
        function uploadPDF(file) {
            const formData = new FormData();
            formData.append('pdf', file);
            
            fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('PDF uploaded and processed successfully!');
                    location.reload();
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('Upload failed: ' + error);
            });
        }
    </script>
</body>
</html>
"""

# Quiz playing template
QUIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Quiz Question {{ current_question + 1 }}</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .quiz-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .progress {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 20px 0;
        }
        .progress-bar {
            background: #28a745;
            height: 20px;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .question-container {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #007bff;
        }
        .question-text {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        .alternative {
            background: white;
            border: 2px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 16px;
        }
        .alternative:hover {
            border-color: #007bff;
            background: #f0f8ff;
        }
        .alternative.correct {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .alternative.incorrect {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .alternative.disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }
        .feedback {
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
        }
        .feedback.correct {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .feedback.incorrect {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .nav-buttons {
            text-align: center;
            margin-top: 30px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        button.success {
            background: #28a745;
        }
        button.secondary {
            background: #6c757d;
        }
        .score-display {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="quiz-header">
            <h1>🎯 Quiz Question {{ current_question + 1 }} of {{ total_questions }}</h1>
            <div class="progress">
                <div class="progress-bar" style="width: {{ (current_question / total_questions * 100)|round }}%"></div>
            </div>
            <div class="score-display">Score: {{ score }} / {{ current_question }}</div>
        </div>
        
        <div class="question-container">
            <div class="question-text">
                {% if question_data.type == 'true_false' %}
                    <span style="color: #28a745; font-weight: bold;">🎯 True or False:</span><br>
                    {{ question_data.question }}
                {% elif question_data.type == 'fill_blank' %}
                    <span style="color: #007bff; font-weight: bold;">📝 Fill in the blank:</span><br>
                    {{ question_data.question }}
                {% else %}
                    <span style="color: #dc3545; font-weight: bold;">🔤 Multiple Choice:</span><br>
                    {{ question_data.question }}
                {% endif %}
            </div>
            
            {% for alt in question_data.alternatives %}
            <div class="alternative {% if answered %}{% if loop.index0 == question_data.correct_index %}correct{% elif user_answer == loop.index0 %}incorrect{% endif %} disabled{% endif %}" 
                 onclick="{% if not answered %}answerQuestion({{ loop.index0 }}){% endif %}">
                {% if question_data.type == 'fill_blank' %}
                    {{ ['A', 'B', 'C'][loop.index0] }}. {{ alt }}
                {% else %}
                    {{ ['A', 'B', 'C', 'D'][loop.index0] }}. {{ alt }}
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        {% if answered %}
        <div class="feedback {% if correct %}correct{% else %}incorrect{% endif %}">
            {% if correct %}
                ✅ Correct! Well done!
            {% else %}
                ❌ Incorrect. The correct answer was: {{ ['A', 'B', 'C', 'D'][question_data.correct_index] }}. {{ question_data.alternatives[question_data.correct_index] }}
            {% endif %}
        </div>
        {% endif %}
        
        <div class="nav-buttons">
            {% if answered %}
                {% if current_question + 1 < total_questions %}
                    <a href="/quiz/next"><button class="success">Next Question ➡️</button></a>
                {% else %}
                    <a href="/quiz/results"><button class="success">View Final Results 🏆</button></a>
                {% endif %}
            {% endif %}
            <a href="/"><button class="secondary">Back to Home 🏠</button></a>
        </div>
    </div>
    
    <script>
        function answerQuestion(answerIndex) {
            fetch('/quiz/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({answer: answerIndex})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
"""

# Q&A template for displaying question-answer pairs
QA_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>💬 Q&A Pair {{ current_question + 1 }}</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .qa-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .progress {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 20px 0;
        }
        .progress-bar {
            background: #ffd700;
            height: 20px;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .question-container {
            background: #fff3cd;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #ffc107;
        }
        .question-text {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #856404;
        }
        .answer-container {
            background: #d1ecf1;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #17a2b8;
        }
        .answer-text {
            font-size: 16px;
            line-height: 1.6;
            color: #0c5460;
        }
        .nav-buttons {
            text-align: center;
            margin-top: 30px;
        }
        button {
            background: #ffc107;
            color: #212529;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            background: #e0a800;
        }
        button.secondary {
            background: #6c757d;
            color: white;
        }
        .qa-counter {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="qa-header">
            <h1>💬 Q&A Session - Pair {{ current_question + 1 }} of {{ total_questions }}</h1>
            <div class="progress">
                <div class="progress-bar" style="width: {{ ((current_question + 1) / total_questions * 100)|round }}%"></div>
            </div>
            <div class="qa-counter">Q&A Pair {{ current_question + 1 }}</div>
        </div>
        
        <div class="question-container">
            <div class="question-text">
                ❓ <strong>Fråga:</strong><br>
                {{ question_data.question }}
            </div>
        </div>
        
        <div class="answer-container">
            <div class="answer-text">
                🤖 <strong>T5 Model Svar:</strong><br>
                {{ question_data.answer }}
            </div>
        </div>
        
        <div class="nav-buttons">
            {% if current_question + 1 < total_questions %}
                <a href="/quiz/next"><button>Nästa Q&A ➡️</button></a>
            {% else %}
                <a href="/quiz/results"><button>Visa Alla Q&A 📋</button></a>
            {% endif %}
            <a href="/"><button class="secondary">Hem 🏠</button></a>
        </div>
    </div>
</body>
</html>
"""

# Q&A Results template for showing all Q&A pairs
QA_RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📋 Q&A Session Sammanfattning</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .stats {
            background: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
            border-left: 4px solid #ffc107;
        }
        .qa-pair {
            background: #f8f9fa;
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #dee2e6;
        }
        .qa-header {
            background: #fff3cd;
            padding: 15px;
            border-bottom: 1px solid #ffc107;
            font-weight: bold;
            color: #856404;
        }
        .question-section {
            background: #fff3cd;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
        }
        .question-text {
            font-size: 16px;
            line-height: 1.5;
            color: #856404;
        }
        .answer-section {
            background: #d1ecf1;
            padding: 20px;
        }
        .answer-text {
            font-size: 16px;
            line-height: 1.6;
            color: #0c5460;
        }
        .nav-buttons {
            text-align: center;
            margin-top: 30px;
        }
        button {
            background: #ffc107;
            color: #212529;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            background: #e0a800;
        }
        button.secondary {
            background: #6c757d;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 Q&A Session Sammanfattning</h1>
        </div>
        
        <div class="stats">
            <strong>💬 Session Statistik:</strong><br>
            📊 Totalt antal Q&A par: {{ total_qa_pairs }}<br>
            🤖 Alla svar genererade av T5 modell
        </div>
        
        {% for qa in qa_pairs %}
        <div class="qa-pair">
            <div class="qa-header">Q&A Par {{ qa.number }}</div>
            <div class="question-section">
                <strong>❓ Fråga:</strong><br>
                <div class="question-text">{{ qa.question }}</div>
            </div>
            <div class="answer-section">
                <strong>🤖 T5 Svar:</strong><br>
                <div class="answer-text">{{ qa.answer }}</div>
            </div>
        </div>
        {% endfor %}
        
        <div class="nav-buttons">
            <a href="/"><button>Ny Session 🏠</button></a>
        </div>
    </div>
</body>
</html>
"""

# Results template
RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏆 Quiz Results</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .results-header {
            margin-bottom: 30px;
        }
        .final-score {
            font-size: 48px;
            font-weight: bold;
            color: #28a745;
            margin: 20px 0;
        }
        .percentage {
            font-size: 24px;
            color: #666;
            margin: 10px 0;
        }
        .performance {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
        }
        .excellent { background: #d4edda; color: #155724; }
        .good { background: #d1ecf1; color: #0c5460; }
        .average { background: #fff3cd; color: #856404; }
        .poor { background: #f8d7da; color: #721c24; }
        
        .question-review {
            text-align: left;
            margin: 30px 0;
        }
        .question-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 5px solid #007bff;
        }
        .question-item.correct {
            border-left-color: #28a745;
        }
        .question-item.incorrect {
            border-left-color: #dc3545;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        button.success {
            background: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="results-header">
            <h1>🏆 Quiz Complete!</h1>
            <div class="final-score">{{ score }} / {{ total_questions }}</div>
            <div class="percentage">{{ percentage }}%</div>
            
            <div class="performance {{ performance_class }}">
                {{ performance_message }}
            </div>
        </div>
        
        <div class="question-review">
            <h3>📋 Question Review:</h3>
            {% for result in question_results %}
            <div class="question-item {{ 'correct' if result.correct else 'incorrect' }}">
                <strong>Q{{ loop.index }}:</strong> {{ result.question }}<br>
                <strong>Your answer:</strong> {{ result.user_answer }}<br>
                {% if not result.correct %}
                <strong>Correct answer:</strong> {{ result.correct_answer }}
                {% endif %}
                <span style="float: right;">{{ '✅' if result.correct else '❌' }}</span>
            </div>
            {% endfor %}
        </div>
        
        <div style="margin-top: 30px;">
            <a href="/start_quiz"><button class="success">🔄 Take Another Quiz</button></a>
            <a href="/"><button>🏠 Back to Home</button></a>
        </div>
    </div>
</body>
</html>
"""

# Summary Template
SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📝 Text Summary - PDF Quiz Helper</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary-content { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #6f42c1; margin: 20px 0; white-space: pre-wrap; }
        .stats { background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; }
        .home-btn { background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin: 10px; }
        .home-btn:hover { background: #218838; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 AI-Generated Text Summary</h1>
        </div>
        
        <div class="stats">
            <strong>📊 Summary Statistics:</strong><br>
            📄 Texts processed: {{ total_texts }}<br>
            📝 Summaries generated: {{ total_summaries }}
        </div>
        
        <div class="summary-content">{{ summary_content }}</div>
        
        <div style="text-align: center; margin-top: 30px;">
            <a href="/" class="home-btn">🏠 Back to Home</a>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    global quiz_generator, model_loaded
    
    # Get extracted texts
    try:
        extracted_texts = load_extracted_texts()
    except:
        extracted_texts = {}
    
    return render_template_string(MAIN_TEMPLATE, 
                                model_loaded=model_loaded,
                                model_path="models/trained_model_best",
                                extracted_texts=extracted_texts,
                                output="System ready. Load model to begin generating quiz questions.")

@app.route('/load_model', methods=['POST'])
def load_model():
    global quiz_generator, model_loaded
    
    model_path = request.form.get('model_path', 'models/trained_model_best')
    
    try:
        quiz_generator = ModelQuizGenerator(model_path)
        if quiz_generator.setup_success:
            model_loaded = True
            status = {
                'type': 'success',
                'message': f'✅ Model loaded successfully from {model_path}'
            }
        else:
            model_loaded = False
            status = {
                'type': 'error', 
                'message': f'❌ Failed to load model from {model_path}'
            }
    except Exception as e:
        model_loaded = False
        status = {
            'type': 'error',
            'message': f'❌ Error loading model: {str(e)}'
        }
    
    try:
        extracted_texts = load_extracted_texts()
    except:
        extracted_texts = {}
    
    return render_template_string(MAIN_TEMPLATE,
                                model_loaded=model_loaded,
                                model_path=model_path,
                                status=status,
                                extracted_texts=extracted_texts,
                                output=f"Model loading attempt completed.")

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    pdf_path = request.form.get('pdf_path', '')
    
    if not pdf_path:
        status = {'type': 'error', 'message': '❌ Please provide a PDF path'}
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=model_loaded,
                                    model_path="models/trained_model_best",
                                    status=status)
    
    try:
        # Run PDF to text conversion
        result = subprocess.run([
            sys.executable, "pdf_to_text.py", pdf_path
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            status = {
                'type': 'success',
                'message': f'✅ PDF converted successfully: {os.path.basename(pdf_path)}'
            }
            output = result.stdout
        else:
            status = {
                'type': 'error',
                'message': f'❌ PDF conversion failed: {result.stderr}'
            }
            output = f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        status = {
            'type': 'error',
            'message': '❌ PDF conversion timed out (>2 minutes)'
        }
        output = "PDF conversion timed out"
    except Exception as e:
        status = {
            'type': 'error',
            'message': f'❌ Error processing PDF: {str(e)}'
        }
        output = f"Error: {str(e)}"
    
    try:
        extracted_texts = load_extracted_texts()
    except:
        extracted_texts = {}
    
    return render_template_string(MAIN_TEMPLATE,
                                model_loaded=model_loaded,
                                model_path="models/trained_model_best", 
                                status=status,
                                extracted_texts=extracted_texts,
                                output=output)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload and processing"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'message': 'No PDF file uploaded'})
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': 'Please upload a PDF file'})
        
        # Save uploaded file
        filename = file.filename
        filepath = Path(filename)
        file.save(str(filepath))
        
        # Process with Smart PDF processor
        from smart_pdf_processor import SmartPDFProcessor
        processor = SmartPDFProcessor()
        
        print(f"🔀 Processing uploaded PDF: {filename}")
        cleaned_text = processor.process_pdf(str(filepath))
        
        return jsonify({
            'success': True, 
            'message': f'Successfully processed {filename}! {len(cleaned_text):,} characters extracted.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    question_type = request.form.get('question_type', 'multiple_choice')
    num_questions_str = request.form.get('num_questions', '5')
    text_selection = request.form.get('text_selection', 'all')
    summary_length = request.form.get('summary_length', 'medium')
    
    # Handle empty num_questions
    try:
        num_questions = int(num_questions_str) if num_questions_str else 5
    except ValueError:
        num_questions = 5
    
    if not quiz_generator or not quiz_generator.setup_success:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=False,
                                    extracted_texts=load_extracted_texts(),
                                    output="❌ Model not loaded. Please load the model first.")
    
    try:
        all_texts = load_extracted_texts()
        if not all_texts:
            return render_template_string(MAIN_TEMPLATE, 
                                        model_loaded=True,
                                        extracted_texts={},
                                        output="❌ No extracted texts found. Please extract some PDFs first.")
        
        # Filter texts based on selection
        if text_selection == 'all':
            texts = all_texts
        else:
            if text_selection in all_texts:
                texts = {text_selection: all_texts[text_selection]}
                print(f"📄 Processing only selected text: {text_selection}")
            else:
                return render_template_string(MAIN_TEMPLATE, 
                                            model_loaded=True,
                                            extracted_texts=all_texts,
                                            output=f"❌ Selected text '{text_selection}' not found.")
        
        # Handle different content types
        if question_type == "summary":
            return handle_summarization(texts, num_questions, summary_length)
        elif question_type == "multiple_choice":
            return handle_quiz_generation(texts, question_type, num_questions)
        elif question_type == "qa":
            return handle_qa_generation(texts, num_questions)
        else:
            return render_template_string(MAIN_TEMPLATE, 
                                        model_loaded=True,
                                        extracted_texts=all_texts,
                                        output=f"❌ Unsupported question type: {question_type}")
            
    except Exception as e:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=True,
                                    extracted_texts=load_extracted_texts(),
                                    output=f"❌ Error: {str(e)}")

def handle_summarization(texts, num_summaries, summary_length):
    """Handle text summarization using T5 model with length option"""
    if len(texts) > 1:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=True,
                                    extracted_texts=texts,
                                    output="❌ Please select only ONE text for summarization. Multiple text summarization is not supported.")
    
    text_name, text_content = list(texts.items())[0]
    print(f"📝 Creating {summary_length} summary for {text_name}...")
    
    # Generate single summary with specified length
    summaries = quiz_generator.generate_questions_from_text(
        text_content, num_summaries, question_type="summarize", summary_length=summary_length
    )
    
    if not summaries:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=True,
                                    extracted_texts=texts,
                                    output="❌ No summary could be generated.")
    
    # Display summary in a nice format
    summary_output = f"📄 **{text_name}** ({summary_length} summary)\n\n{summaries[0]}"
    
    return render_template_string(SUMMARY_TEMPLATE,
                                summary_content=summary_output,
                                total_texts=1,
                                total_summaries=1)

def handle_quiz_generation(texts, question_type, num_questions):
    """Handle traditional quiz generation"""
    all_questions = []
    
    for text_name, text_content in texts.items():
        questions = quiz_generator.generate_questions_from_text(
            text_content, num_questions, question_type=question_type
        )
        
        if questions:
            for question in questions:
                # Parse different question formats
                parsed_question = _parse_question_format(question, question_type)
                if parsed_question:
                    parsed_question['source'] = text_name
                    all_questions.append(parsed_question)
    
    if not all_questions:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=True,
                                    extracted_texts=texts,
                                    output="❌ No valid questions could be generated.")
    
    # Shuffle questions and store in session
    import random
    random.shuffle(all_questions)
    session['quiz_questions'] = all_questions
    session['current_question'] = 0
    session['score'] = 0
    session['question_results'] = []
    
    # Clear any previous answer state to prevent auto-answer bug
    session.pop('question_answered', None)
    session.pop('user_answer', None)
    
    # Redirect to first question
    return redirect('/quiz/question')

def handle_qa_generation(texts, num_questions):
    """Handle question-answering generation"""
    all_questions = []
    
    for text_name, text_content in texts.items():
        questions = quiz_generator.generate_questions_from_text(
            text_content, num_questions, question_type="qa"
        )
        
        if questions:
            for question in questions:
                # Parse different question formats
                parsed_question = _parse_question_format(question, "qa")
                if parsed_question:
                    parsed_question['source'] = text_name
                    all_questions.append(parsed_question)
    
    if not all_questions:
        return render_template_string(MAIN_TEMPLATE, 
                                    model_loaded=True,
                                    extracted_texts=texts,
                                    output="❌ No valid questions could be generated.")
    
    # Shuffle questions and store in session
    import random
    random.shuffle(all_questions)
    session['quiz_questions'] = all_questions
    session['current_question'] = 0
    session['score'] = 0
    session['question_results'] = []
    
    # Clear any previous answer state to prevent auto-answer bug
    session.pop('question_answered', None)
    session.pop('user_answer', None)
    
    # Redirect to first question
    return redirect('/quiz/question')

def _parse_question_format(question: str, question_type: str) -> dict:
    """Parse question formats into a unified structure"""
    lines = question.strip().split('\n')
    
    if question_type == "qa":
        # Q&A format: ❓ **Fråga:** Question\n\n🤖 **T5 Svar:** Answer
        if "**Fråga:**" in question and "**T5 Svar:**" in question:
            # Split by the answer marker
            parts = question.split("🤖 **T5 Svar:**")
            if len(parts) == 2:
                question_part = parts[0].replace("❓ **Fråga:**", "").strip()
                answer_part = parts[1].strip()
                
                # Create a simple Q&A display format
                return {
                    'question': question_part,
                    'answer': answer_part,
                    'type': 'qa',
                    'alternatives': [],  # No alternatives for Q&A
                    'correct_index': -1  # No correct index needed
                }
    
    elif question_type == "multiple_choice":
        # Multiple choice format: Question + A) B) C) D) + Correct: A/B/C/D
        if len(lines) >= 6:
            question_text = lines[0]
            alternatives = []
            correct_answer = ""
            
            # Extract alternatives
            for line in lines[1:5]:
                if line.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                    alternatives.append(line.strip()[2:].strip())
            
            # Extract correct answer
            for line in lines:
                if line.strip().startswith('Correct:'):
                    correct_answer = line.strip().split(':')[1].strip()
                    break
            
            if len(alternatives) >= 3 and correct_answer:
                # Handle both 3 and 4 alternatives
                while len(alternatives) < 4:
                    alternatives.append("None of the above")
                    
                correct_index = ord(correct_answer) - ord('A')
                return {
                    'question': question_text,
                    'alternatives': alternatives,
                    'correct_index': correct_index,
                    'type': 'multiple_choice'
                }
    
    return None

@app.route('/quiz/question')
def quiz_question():
    if 'quiz_questions' not in session:
        return redirect('/')
    
    questions = session['quiz_questions']
    current_q = session.get('current_question', 0)
    
    if current_q >= len(questions):
        return redirect('/quiz/results')
    
    question_data = questions[current_q]
    
    # Handle Q&A format differently (no scoring, just display)
    if question_data.get('type') == 'qa':
        return render_template_string(QA_TEMPLATE,
                                    current_question=current_q,
                                    total_questions=len(questions),
                                    question_data=question_data)
    
    # Handle quiz questions (multiple choice, etc.)
    answered = session.get('question_answered', False)
    user_answer = session.get('user_answer', None)
    correct = False
    
    if answered and user_answer is not None:
        correct = (user_answer == question_data['correct_index'])
    
    return render_template_string(QUIZ_TEMPLATE,
                                current_question=current_q,
                                total_questions=len(questions),
                                score=session.get('score', 0),
                                question_data=question_data,
                                answered=answered,
                                user_answer=user_answer,
                                correct=correct)

@app.route('/quiz/answer', methods=['POST'])
def quiz_answer():
    if 'quiz_questions' not in session:
        return jsonify({'success': False, 'message': 'No active quiz'})
    
    questions = session['quiz_questions']
    current_q = session.get('current_question', 0)
    
    if current_q >= len(questions):
        return jsonify({'success': False, 'message': 'No more questions'})
    
    question_data = questions[current_q]
    
    # Skip scoring for Q&A type questions
    if question_data.get('type') == 'qa':
        return jsonify({'success': True, 'message': 'Q&A viewed'})
    
    data = request.get_json()
    answer = int(data['answer'])
    
    correct = (answer == question_data['correct_index'])
    
    # Update session
    session['question_answered'] = True
    session['user_answer'] = answer
    
    if correct:
        session['score'] = session.get('score', 0) + 1
    
    # Store result for final review
    result = {
        'question': question_data['question'],
        'user_answer': question_data['alternatives'][answer],
        'correct_answer': question_data['alternatives'][question_data['correct_index']],
        'correct': correct
    }
    
    if 'question_results' not in session:
        session['question_results'] = []
    session['question_results'].append(result)
    
    return jsonify({'success': True})

@app.route('/quiz/next')
def quiz_next():
    if 'quiz_questions' not in session:
        return redirect('/')
    
    session['current_question'] = session.get('current_question', 0) + 1
    
    # Properly clear answer state for next question
    session.pop('question_answered', None)
    session.pop('user_answer', None)
    
    return redirect('/quiz/question')

@app.route('/quiz/results')
def quiz_results():
    if 'quiz_questions' not in session:
        return redirect('/')
    
    questions = session['quiz_questions']
    total_questions = len(questions)
    
    # Check if this is a Q&A session (no scoring)
    if questions and questions[0].get('type') == 'qa':
        # Q&A session - show all Q&A pairs
        qa_pairs = []
        for i, q in enumerate(questions):
            qa_pairs.append({
                'number': i + 1,
                'question': q['question'],
                'answer': q['answer']
            })
        
        return render_template_string(QA_RESULTS_TEMPLATE,
                                    total_qa_pairs=total_questions,
                                    qa_pairs=qa_pairs)
    
    # Regular quiz with scoring
    score = session.get('score', 0)
    percentage = round((score / total_questions) * 100) if total_questions > 0 else 0
    question_results = session.get('question_results', [])
    
    # Determine performance
    if percentage >= 90:
        performance_class = 'excellent'
        performance_message = '🌟 Excellent! You performed exceptionally well!'
    elif percentage >= 70:
        performance_class = 'good'
        performance_message = '👍 Good job! You did well!'
    elif percentage >= 50:
        performance_class = 'average'
        performance_message = '📚 Not bad! Keep studying to improve!'
    else:
        performance_class = 'poor'
        performance_message = '📖 Keep practicing! You can do better!'
    
    return render_template_string(RESULTS_TEMPLATE,
                                score=score,
                                total_questions=total_questions,
                                percentage=percentage,
                                performance_class=performance_class,
                                performance_message=performance_message,
                                question_results=question_results)

@app.route('/process_smart_pdf', methods=['POST'])
def process_smart_pdf():
    """Process PDF with the smart hybrid approach"""
    try:
        pdf_path = "Attention_is_all_you_need_v7.pdf"
        
        if not Path(pdf_path).exists():
            return render_template_string(MAIN_TEMPLATE,
                                        model_loaded=bool(quiz_generator and quiz_generator.setup_success),
                                        extracted_texts=load_extracted_texts(),
                                        output="❌ Attention PDF not found in current directory")
        
        # Import smart processor
        from smart_pdf_processor import SmartPDFProcessor
        
        print(f"🔀 Processing {pdf_path} with Smart Hybrid approach...")
        processor = SmartPDFProcessor()
        
        # Process with hybrid approach
        cleaned_text = processor.process_pdf(pdf_path)
        
        return render_template_string(MAIN_TEMPLATE,
                                    model_loaded=bool(quiz_generator and quiz_generator.setup_success),
                                    extracted_texts=load_extracted_texts(),
                                    output=f"""✅ Successfully processed {pdf_path} with Smart Hybrid approach!
                                    
📊 Results:
• File: {pdf_path}
• Characters extracted: {len(cleaned_text):,}
• Method: PyMuPDF + OpenAI cleaning (when API works)
• Saved as: Attention_is_all_you_need_v7_SMART.txt

🎯 The cleaned text is now available for quiz generation!
You can now use 'AI Questions + T5 Answers (Hybrid)' with this high-quality text.""")
        
    except Exception as e:
        return render_template_string(MAIN_TEMPLATE,
                                    model_loaded=bool(quiz_generator and quiz_generator.setup_success),
                                    extracted_texts=load_extracted_texts(),
                                    output=f"❌ Error processing PDF: {str(e)}")

@app.route('/ask_t5', methods=['POST'])
def ask_t5():
    """Handle direct questions to T5 model (chatbot functionality)"""
    user_question = request.form.get('user_question', '').strip()
    
    if not quiz_generator or not quiz_generator.setup_success:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please load the model first.'
        })
    
    if not user_question:
        return jsonify({
            'success': False,
            'error': 'Please enter a question.'
        })
    
    try:
        # Use T5 to directly answer the user's question
        answer = quiz_generator.ask_direct_question(user_question)
        
        return jsonify({
            'success': True,
            'question': user_question,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating answer: {str(e)}'
        })

if __name__ == '__main__':
    print("🌐 Starting PDF-to-Quiz Web Interface...")
    print("📋 Access the interface at: http://localhost:5001")
    print("🔄 Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 