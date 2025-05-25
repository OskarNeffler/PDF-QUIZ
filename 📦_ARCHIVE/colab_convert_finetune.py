# === COLAB SKRIPT FÖR ONNX KONVERTERING OCH FINETUNING ===
# Kopiera detta till Google Colab och kör sektionerna en i taget

# === SEKTION 1: INSTALLATION AV NÖDVÄNDIGA PAKET ===
# %% [markdown]
"""
# T5 Modell Konvertering och Finetuning

Detta skript konverterar en T5-baserad frågegenereringsmodell till ONNX-format och finjusterar den med hyperparameter-tuning.

## 1. Installation av nödvändiga paket
"""

# %%
# Installera nödvändiga paket
!pip install -q onnx onnxruntime optimum transformers datasets torch optuna matplotlib

# Ladda in nödvändiga bibliotek
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime
import zipfile
from google.colab import files, drive

# Koppla till Google Drive för att spara modeller
drive.mount('/content/drive')

# === SEKTION 2: INSTÄLLNINGAR OCH SETUP ===
# %% [markdown]
"""
## 2. Inställningar och setup
"""

# %%
# Definiera dina inställningar här
MODEL_NAME = "t5-small"  # Base model name
MAX_TARGET_LENGTH = 64  # Maximum length for generated questions

# Ange sökvägen till din T5-modell på Google Drive
MODEL_PATH = "/content/drive/MyDrive/models/question_generation_model_final_CPU"  # Ändra denna sökväg

# Skapa utdatakatalog
OUTPUT_DIR = "/content/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SEKTION 3: ONNX KONVERTERING ===
# %% [markdown]
"""
## 3. Konvertera modellen till ONNX-format

ONNX (Open Neural Network Exchange) är ett öppet format som gör det möjligt att flytta modeller mellan olika ramverk. 
ONNX-modeller körs typiskt snabbare på CPU än vanliga PyTorch-modeller.
"""

# %%
def convert_to_onnx(model_dir, output_dir=None):
    """
    Konvertera modellen till ONNX-format för snabbare inferens
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    
    if output_dir is None:
        output_dir = f"{model_dir}_onnx"
    
    print(f"Konverterar modell från {model_dir} till ONNX-format...")
    
    try:
        # Ladda modell och tokenizer
        print("Laddar modell...")
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        
        # Konvertera till ONNX
        print("Konverterar till ONNX-format...")
        onnx_path = os.path.join(output_dir)
        os.makedirs(onnx_path, exist_ok=True)
        
        # Använd optimum för konvertering
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            model_dir, 
            from_transformers=True, 
            export=True
        )
        
        # Spara ONNX-modellen
        ort_model.save_pretrained(onnx_path)
        tokenizer.save_pretrained(onnx_path)
        
        # Skapa testskript
        with open(os.path.join(onnx_path, "test_onnx_model.py"), "w") as f:
            f.write("""
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5Tokenizer

# Ladda ONNX-modell och tokenizer
ort_model = ORTModelForSeq2SeqLM.from_pretrained("./")
tokenizer = T5Tokenizer.from_pretrained("./")

# Testdata
context = "Python är ett högnivåspråk känt för sin läsbarhet."
input_text = "generate question: " + context
inputs = tokenizer(input_text, return_tensors="pt")

# Generera med ONNX-modell
outputs = ort_model.generate(inputs["input_ids"], max_length=64)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Kontext: {context}")
print(f"Genererad fråga: {question}")
""")
        
        # Skapa README
        with open(os.path.join(onnx_path, "README.md"), "w") as f:
            f.write(f"""# ONNX T5 Frågegenereringsmodell

Denna modell har konverterats till ONNX-format för snabbare inferens.

## Installation
```bash
pip install onnx onnxruntime optimum transformers
```

## Användning
```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5Tokenizer

# Ladda modell och tokenizer
model = ORTModelForSeq2SeqLM.from_pretrained("./")
tokenizer = T5Tokenizer.from_pretrained("./")

# Förbered indata
context = "Din text här"
input_text = "generate question: " + context
inputs = tokenizer(input_text, return_tensors="pt")

# Generera en fråga (snabbare med ONNX)
outputs = model.generate(inputs["input_ids"])
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(question)
```

## Prestanda
- Snabbare inferens jämfört med vanlig PyTorch-modell
- Optimerad för CPU-användning
- Mindre minnesavtryck
- Bättre lämpad för produktion

## Modellinformation
- Originalmodell: {MODEL_NAME}
- Konverteringsdatum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        
        # Skapa zip-fil för nedladdning
        zip_path = f"{onnx_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(onnx_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(onnx_path))
                    zipf.write(file_path, arcname)
        
        print(f"""
✅ ONNX-konvertering klar!

📁 ONNX-modell sparad i: {onnx_path}
📦 Zip-fil skapad: {zip_path}

Ladda ner zip-filen genom att köra kommandot:
files.download('{zip_path}')
""")
        
        return onnx_path
    
    except Exception as e:
        print(f"ONNX-konvertering misslyckades: {str(e)}")
        return None

# Kör ONNX-konvertering
onnx_path = convert_to_onnx(MODEL_PATH, os.path.join(OUTPUT_DIR, "model_onnx"))

# Ladda ner ONNX-modellen
if onnx_path:
    zip_path = f"{onnx_path}.zip"
    if os.path.exists(zip_path):
        files.download(zip_path)

# === SEKTION 4: HYPERPARAMETER TUNING ===
# %% [markdown]
"""
## 4. Hyperparameter-tuning med Optuna

Nu när vi har en ONNX-modell, kan vi finjustera den med hyperparameter-tuning.
Detta hjälper oss att hitta optimala värden för lärhastighet, batch-storlek, etc.
"""

# %%
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset

# Ladda SQuAD dataset för finjustering
def load_and_preprocess_squad(tokenizer, max_length=512, max_target_length=64):
    """Ladda och förbehandla SQuAD-datasetet för frågegenereringsuppgiften"""
    print("Laddar SQuAD-dataset...")
    
    # Ladda SQuAD dataset
    squad = load_dataset("squad")
    
    # Förbehandla data
    def preprocess_function(examples):
        contexts = examples["context"]
        questions = examples["question"]
        
        # Indata: "generate question: " + context
        # Utdata: question
        inputs = ["generate question: " + context for context in contexts]
        targets = questions
        
        # Tokenisera indata och utdata
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tillämpa förbehandling
    processed_squad = squad.map(
        preprocess_function,
        batched=True,
        remove_columns=squad["train"].column_names,
        desc="Förbehandlar data"
    )
    
    print(f"Dataset förbehandlat. Träningsstorlek: {len(processed_squad['train'])}, Valideringsstorlek: {len(processed_squad['validation'])}")
    return processed_squad

# Beräkna utvärderingsmetriker
def compute_metrics(eval_pred):
    """Beräkna utvärderingsmetriker för träning"""
    from rouge_score import rouge_scorer
    import numpy as np
    
    predictions, labels = eval_pred
    
    # Avkoda prediktioner och faktiska etiketter
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    
    # Ta bort padding-token (vanligtvis -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Avkoda prediktioner och etiketter till text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE-metriker för textutvärdering
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Beräkna poäng
    rouge1, rouge2, rougeL = 0, 0, 0
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougeL += score['rougeL'].fmeasure
    
    # Genomsnittlig poäng
    rouge1 /= len(decoded_preds)
    rouge2 /= len(decoded_preds)
    rougeL /= len(decoded_preds)
    
    # Exakt matchning
    exact_match = sum([1 if pred.strip() == label.strip() else 0 for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
    
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rouge_l": rougeL,
        "exact_match": exact_match
    }

# Hyperparameter-tuning med Optuna
def run_hyperparameter_tuning(model_path, output_dir, num_trials=8):
    """Kör hyperparameter-tuning för T5-modellen med Optuna"""
    
    # Skapa tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Ladda dataset
    processed_dataset = load_and_preprocess_squad(tokenizer)
    
    def objective(trial):
        """Optuna-målfunktion för hyperparameter-optimering"""
        
        # Definiera hyperparametrar med breda intervall
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [4, 8, 16])
        wd = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
        warmup = trial.suggest_float("warmup_ratio", 0.0, 0.3)
        
        # Skapa utdatakatalog för denna körning
        trial_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        try:
            # Ladda modell
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            # Träningsinställningar
            train_dataset_size = len(processed_dataset["train"])
            steps_per_epoch = train_dataset_size // bs
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=trial_output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                weight_decay=wd,
                warmup_ratio=warmup,
                save_total_limit=1,
                num_train_epochs=3,
                predict_with_generate=True,
                generation_max_length=MAX_TARGET_LENGTH,
                generation_num_beams=4,
                load_best_model_at_end=True,
                metric_for_best_model="rouge_l",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),
                report_to="none"
            )
            
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True
            )
            
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            
            # Träna och utvärdera
            trainer.train()
            metrics = trainer.evaluate()
            
            # Spara metriker
            trial.set_user_attr("exact_match", metrics.get("eval_exact_match", 0))
            trial.set_user_attr("eval_loss", metrics.get("eval_loss", float('inf')))
            
            rouge_l = metrics.get("eval_rouge_l", 0)
            
            # Rensa upp om det behövs
            if os.path.exists(trial_output_dir):
                shutil.rmtree(trial_output_dir)
            
            return rouge_l
            
        except Exception as e:
            print(f"Trial {trial.number} misslyckades: {str(e)}")
            return 0
    
    # Skapa Optuna-studie
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        study_name="t5_question_generation"
    )
    
    print(f"Startar Optuna-optimering med {num_trials} körningar")
    
    try:
        study.optimize(objective, n_trials=num_trials)
        
        # Hämta resultat
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        
        print("\n" + "="*80)
        print("HYPERPARAMETER-OPTIMERINGSRESULTAT")
        print("="*80)
        print(f"Bästa ROUGE-L-poäng: {best_value:.6f}")
        print(f"Bästa Exact Match: {best_trial.user_attrs.get('exact_match', 'N/A'):.6f}")
        
        print("\nBästa hyperparametrar:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
        
        # Spara detaljerade resultat
        result_file = os.path.join(output_dir, "hyperparameter_comparison.txt")
        with open(result_file, "w") as f:
            f.write("T5 Frågegenereringsmodell Hyperparameter-optimeringsresultat\n")
            f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modell: {MODEL_NAME}\n")
            f.write(f"Antal körningar: {len(study.trials)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BÄSTA KONFIGURATION\n")
            f.write("="*80 + "\n")
            f.write(f"Körningsnummer: {best_trial.number}\n")
            f.write(f"ROUGE-L-poäng: {best_value:.6f}\n")
            
            for param, value in best_params.items():
                f.write(f"- {param}: {value}\n")
        
        print(f"\nResultat sparade i: {result_file}")
        
        # Skapa visualiseringar
        try:
            plot_dir = os.path.join(output_dir, "optimization_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Optimeringshistorik
            plt.figure(figsize=(12, 6))
            trial_numbers = [t.number for t in study.trials if t.value is not None]
            trial_values = [t.value for t in study.trials if t.value is not None]
            
            plt.plot(trial_numbers, trial_values, 'bo-')
            plt.xlabel('Körningsnummer')
            plt.ylabel('ROUGE-L-poäng')
            plt.title('Optimeringshistorik')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "optimization_history.png"))
            plt.close()
            
            print(f"Diagram sparade i: {plot_dir}")
            
        except Exception as e:
            print(f"Kunde inte skapa diagram: {e}")
        
        return best_params
        
    except Exception as e:
        print(f"Optuna-optimering misslyckades: {str(e)}")
        return None

# Kör hyperparameter-tuning
# Kör följande rader för att aktivera tuning (ta bort # för att aktivera)
# !pip install rouge_score  # Behövs för compute_metrics
# best_params = run_hyperparameter_tuning(onnx_path, os.path.join(OUTPUT_DIR, "tuning_results"), num_trials=8)

# === SEKTION 5: TRÄNA SLUTLIG MODELL ===
# %% [markdown]
"""
## 5. Träna slutlig modell med optimerade hyperparametrar

När vi har hittat de bästa hyperparametrarna, kan vi träna en slutlig modell.
"""

# %%
def train_final_model(model_path, output_dir, hyperparams=None):
    """Träna slutlig modell med optimerade hyperparametrar"""
    
    if hyperparams is None:
        # Standardvärden om inga optimerade parametrar anges
        hyperparams = {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1
        }
    
    print(f"Tränar slutlig modell med följande hyperparametrar:")
    for param, value in hyperparams.items():
        print(f"- {param}: {value}")
    
    # Skapa katalog
    os.makedirs(output_dir, exist_ok=True)
    
    # Ladda modell och tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Ladda dataset
    processed_dataset = load_and_preprocess_squad(tokenizer)
    
    # Träningsinställningar
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=hyperparams["learning_rate"],
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        weight_decay=hyperparams["weight_decay"],
        warmup_ratio=hyperparams["warmup_ratio"],
        save_total_limit=3,
        num_train_epochs=5,  # Fler epoker för slutlig träning
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard"
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Träna modellen
    print("Tränar slutlig modell...")
    trainer.train()
    
    # Utvärdera modellen
    print("Utvärderar modellen...")
    metrics = trainer.evaluate()
    
    print("\nUtvärderingsresultat:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
    
    # Spara modellen
    print(f"Sparar modellen till {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Skapa README
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"""# Finjusterad T5 Frågegenereringsmodell

## Modellinformation
- Basmodell: {MODEL_NAME}
- Träningsdatum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset: SQuAD

## Hyperparametrar
""")
        for param, value in hyperparams.items():
            f.write(f"- {param}: {value}\n")
        
        f.write("\n## Utvärderingsresultat\n")
        for key, value in metrics.items():
            f.write(f"- {key}: {value}\n")
        
        f.write("""
## Användning
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Ladda modell och tokenizer
model = T5ForConditionalGeneration.from_pretrained("./")
tokenizer = T5Tokenizer.from_pretrained("./")

# Förbered indata
context = "Din text här"
input_text = "generate question: " + context
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generera en fråga
outputs = model.generate(input_ids, max_length=64)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(question)
```
""")
    
    # Skapa zip-fil för nedladdning
    print("Skapar zip-fil för nedladdning...")
    zip_path = f"{output_dir}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)
    
    print(f"""
✅ Träning av slutlig modell klar!

📁 Modell sparad i: {output_dir}
📦 Zip-fil skapad: {zip_path}

Ladda ner zip-filen genom att köra kommandot:
files.download('{zip_path}')
""")
    
    return output_dir

# För att träna slutlig modell med de bästa hyperparametrarna, kör:
# final_model_path = train_final_model(onnx_path, os.path.join(OUTPUT_DIR, "final_model"), best_params)

# === SEKTION 6: LADDA NER MODELLER ===
# %% [markdown]
"""
## 6. Ladda ner modeller

Kör följande celler för att ladda ner dina modeller:
"""

# %%
# Ladda ner ONNX-modell
# files.download(f"{onnx_path}.zip")

# %%
# Ladda ner finjusterad modell
# files.download(f"{os.path.join(OUTPUT_DIR, 'final_model')}.zip") 