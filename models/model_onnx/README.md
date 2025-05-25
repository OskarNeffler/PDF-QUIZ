# ONNX T5 Frågegenereringsmodell

Konverterad från T5-modell till ONNX-format för snabbare inferens.

## Installation
pip install onnx onnxruntime optimum transformers

## Användning
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5Tokenizer

model = ORTModelForSeq2SeqLM.from_pretrained("./")
tokenizer = T5Tokenizer.from_pretrained("./")

context = "Din text här"
input_text = "generate question: " + context
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(inputs["input_ids"], max_length=64)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(question)

Konverteringsdatum: 2025-05-22 14:30:42