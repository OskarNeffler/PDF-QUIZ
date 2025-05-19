import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import threading
import json

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_processor import extract_text_from_pdf, get_document_metadata
from text_analyzer import analyze_content
from question_generator import generate_questions
from model_inference import get_model_instance


class PDFQuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF-to-Quiz Generator")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        self.pdf_path = None
        self.questions = []
        self.model_path = None
        self.model_loaded = False
        
        # Create main containers
        self.create_widgets()
        
        # Status variables
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Status bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_widgets(self):
        # Create top frame for file selection and controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # File selection
        file_label = tk.Label(top_frame, text="PDF File:")
        file_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(top_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_button = tk.Button(top_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5)
        
        # Create frame for model selection
        model_frame = tk.Frame(self.root)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model path
        model_label = tk.Label(model_frame, text="Model Path:")
        model_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.model_path_var = tk.StringVar()
        model_entry = tk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        model_browse_button = tk.Button(model_frame, text="Browse", command=self.browse_model)
        model_browse_button.pack(side=tk.LEFT, padx=5)
        
        load_model_button = tk.Button(model_frame, text="Load Model", command=self.load_model)
        load_model_button.pack(side=tk.LEFT, padx=5)
        
        # Create frame for options
        options_frame = tk.Frame(self.root)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Question types
        types_frame = tk.LabelFrame(options_frame, text="Question Types")
        types_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.question_type_vars = {}
        question_types = [
            ('Multiple Choice', 'multiple_choice'),
            ('True/False', 'true_false'),
            ('Fill in Blank', 'fill_in_blank'),
            ('Definition', 'definition'),
            ('Syntax Usage', 'syntax_usage')
        ]
        
        for i, (text, val) in enumerate(question_types):
            var = tk.BooleanVar(value=True)
            self.question_type_vars[val] = var
            cb = tk.Checkbutton(types_frame, text=text, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
        
        # Use model checkbox
        self.use_model_var = tk.BooleanVar(value=False)
        use_model_cb = tk.Checkbutton(types_frame, text="Use Trained Model", variable=self.use_model_var)
        use_model_cb.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Number of questions
        count_frame = tk.Frame(options_frame)
        count_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        count_label = tk.Label(count_frame, text="Number of questions:")
        count_label.pack(side=tk.LEFT, padx=5)
        
        self.count_var = tk.IntVar(value=10)
        count_spinner = tk.Spinbox(count_frame, from_=1, to=50, width=3, textvariable=self.count_var)
        count_spinner.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        generate_button = tk.Button(options_frame, text="Generate Questions", command=self.generate_questions)
        generate_button.pack(side=tk.LEFT, padx=10)
        
        # Export button
        export_button = tk.Button(options_frame, text="Export", command=self.export_questions)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Create output area
        output_frame = tk.LabelFrame(self.root, text="Generated Questions")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a Text widget with scrollbar
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
    
    def browse_file(self):
        filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path_var.set(filename)
            self.pdf_path = filename
            
            # Show PDF info
            try:
                metadata = get_document_metadata(filename)
                info = f"File: {os.path.basename(filename)}\n"
                info += f"Pages: {metadata.get('page_count', 'Unknown')}\n"
                if 'title' in metadata:
                    info += f"Title: {metadata['title']}\n"
                if 'author' in metadata:
                    info += f"Author: {metadata['author']}\n"
                
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, info)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read PDF metadata: {str(e)}")
    
    def browse_model(self):
        """Browse for trained model directory"""
        model_dir = filedialog.askdirectory(title="Select Trained Model Directory")
        if model_dir:
            self.model_path_var.set(model_dir)
            self.model_path = model_dir
    
    def load_model(self):
        """Load the trained model from the specified directory"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showwarning("Warning", "Please select a model directory first.")
            return
        
        self.status_var.set("Loading model...")
        self.progress["value"] = 10
        self.root.update_idletasks()
        
        try:
            # Disable buttons during loading
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Button):
                    widget["state"] = "disabled"
            
            # Load model in a separate thread to avoid freezing the UI
            threading.Thread(target=self._load_model, args=(model_path,)).start()
        
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
            # Re-enable buttons
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Button):
                    widget["state"] = "normal"
    
    def _load_model(self, model_path):
        """Background thread for model loading"""
        try:
            model = get_model_instance(model_path)
            self.model_loaded = model.is_model_loaded()
            
            if self.model_loaded:
                self.status_var.set(f"Model loaded successfully")
                self.progress["value"] = 100
                # Enable use model checkbox
                self.use_model_var.set(True)
            else:
                self.status_var.set(f"Failed to load model")
                messagebox.showerror("Error", "Failed to load model. Check the model path.")
                self.use_model_var.set(False)
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.use_model_var.set(False)
            self.model_loaded = False
        
        finally:
            # Re-enable buttons
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Button):
                    widget["state"] = "normal"
            
            self.root.update_idletasks()
    
    def get_selected_question_types(self):
        return [qt for qt, var in self.question_type_vars.items() if var.get()]
    
    def generate_questions(self):
        if not self.pdf_path:
            messagebox.showwarning("Warning", "Please select a PDF file first.")
            return
        
        use_model = self.use_model_var.get()
        if use_model and not self.model_loaded:
            messagebox.showwarning("Warning", "Model is not loaded yet. Please load a model first or uncheck 'Use Trained Model'.")
            return
        
        # Get options
        question_types = self.get_selected_question_types()
        if not question_types and not use_model:
            messagebox.showwarning("Warning", "Please select at least one question type.")
            return
        
        count = self.count_var.get()
        
        # Start processing in a separate thread
        self.progress["value"] = 0
        self.status_var.set("Processing PDF...")
        
        # Disable buttons during processing
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget["state"] = "disabled"
        
        threading.Thread(target=self._process_pdf, args=(question_types, count, use_model)).start()
    
    def _process_pdf(self, question_types, count, use_model):
        try:
            # Extract text
            self.status_var.set("Extracting text from PDF...")
            self.progress["value"] = 10
            self.root.update_idletasks()
            
            sections = extract_text_from_pdf(self.pdf_path)
            
            # Analyze content
            self.status_var.set("Analyzing content...")
            self.progress["value"] = 30
            self.root.update_idletasks()
            
            analyzed_sections = analyze_content(sections)
            
            # Generate questions
            self.status_var.set("Generating questions...")
            self.progress["value"] = 60
            self.root.update_idletasks()
            
            if use_model and self.model_loaded:
                # Use trained model
                model = get_model_instance()
                self.questions = model.generate_questions_from_sections(
                    analyzed_sections,
                    count=count
                )
            else:
                # Use rule-based generation
                self.questions = generate_questions(
                    analyzed_sections,
                    question_types=question_types,
                    count=count
                )
            
            # Display questions
            self.status_var.set(f"Generated {len(self.questions)} questions.")
            self.progress["value"] = 100
            self.root.update_idletasks()
            
            self.display_questions()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            # Re-enable buttons
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Button):
                    widget["state"] = "normal"
    
    def display_questions(self):
        if not self.questions:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "No questions could be generated. Try with a different PDF or different settings.")
            return
        
        self.output_text.delete(1.0, tk.END)
        
        for i, q in enumerate(self.questions, 1):
            self.output_text.insert(tk.END, f"Question {i}: {q['question']}\n")
            
            if 'options' in q:
                for j, option in enumerate(q['options']):
                    self.output_text.insert(tk.END, f"  {chr(65+j)}. {option}\n")
            
            self.output_text.insert(tk.END, f"Answer: {q['answer']}\n")
            
            if 'explanation' in q:
                self.output_text.insert(tk.END, f"Explanation: {q['explanation']}\n")
            
            self.output_text.insert(tk.END, "\n")
    
    def export_questions(self):
        if not self.questions:
            messagebox.showwarning("Warning", "No questions to export. Generate questions first.")
            return
        
        filetypes = [
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=filetypes
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.questions, f, indent=2, ensure_ascii=False)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    for i, q in enumerate(self.questions, 1):
                        f.write(f"Question {i}: {q['question']}\n")
                        if 'options' in q:
                            for j, option in enumerate(q['options']):
                                f.write(f"  {chr(65+j)}. {option}\n")
                        f.write(f"Answer: {q['answer']}\n")
                        if 'explanation' in q:
                            f.write(f"Explanation: {q['explanation']}\n")
                        f.write("\n")
            
            self.status_var.set(f"Questions exported to {os.path.basename(filename)}")
            messagebox.showinfo("Export Successful", f"Questions exported to {filename}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting questions: {str(e)}")


def run_ui():
    root = tk.Tk()
    app = PDFQuizApp(root)
    root.mainloop() 