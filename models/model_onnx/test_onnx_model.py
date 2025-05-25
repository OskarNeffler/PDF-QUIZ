from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5Tokenizer

ort_model = ORTModelForSeq2SeqLM.from_pretrained("./")
tokenizer = T5Tokenizer.from_pretrained("./")

context = "Python är ett högnivåspråk känt för sin läsbarhet."
input_text = "generate question: " + context
inputs = tokenizer(input_text, return_tensors="pt")

outputs = ort_model.generate(inputs["input_ids"], max_length=64)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Kontext: {context}")
print(f"Genererad fråga: {question}")