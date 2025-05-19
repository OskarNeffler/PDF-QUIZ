# PDF-to-Quiz

En applikation som automatiskt genererar quizfrågor från PDF-dokument med hjälp av NLP (Natural Language Processing) och en finetunad T5-modell.

## Funktioner

- Extrahera text från PDF-dokument
- Analysera innehåll för att identifiera viktiga koncept, fakta och definitioner
- Generera olika typer av frågor:
  - Flervalsalternativ
  - Sant/Falskt
  - Lucktext
  - Definitionsfrågor
  - Syntaxanvändningsfrågor
- Tillhandahålla förklaringar till svar
- Användarinterface via kommandoraden
- Exportera frågor i JSON- eller textformat

## Installation

1. Klona repositoriet:
   ```
   git clone https://github.com/OskarNeffler/PDF-QUIZ.git
   cd PDF-QUIZ
   ```

2. Skapa en virtuell miljö och installera de nödvändiga paketen:
   ```
   python -m venv venv
   source venv/bin/activate  # På Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ladda ner nödvändig spaCy-modell:
   ```
   python -m spacy download en_core_web_sm
   ```

4. Ladda ner nödvändig NLTK-data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Användning

### Via kommandoraden

För att generera frågor från en PDF via kommandoraden:

```
python generate_questions.py --pdf ditt_dokument.pdf --output fragor.txt --count 15
```

Valfria argument:
- `--pdf`: Sökväg till PDF-filen (krävs)
- `--output`: Utdatafil för genererade frågor (valfritt)
- `--count`: Antal frågor att generera (standard: 10)
- `--model_path`: Sökväg till modellkatalogen (standard: question_generation_model)

### Optimera modellen för CPU

För att optimera den finetunade modellen för CPU-inferens:

```
python optimize_for_cpu.py
```

Detta skapar en CPU-optimerad version av modellen i katalogen `question_generation_model_cpu`.

## Modellträning

Projektet använder en finetunad T5-modell tränad på SQuAD-datasetet för att generera frågor från text.

### Hur det fungerar

1. **PDF-bearbetning**: Applikationen extraherar text från PDF:er samtidigt som dokumentstrukturen bevaras.
2. **Innehållsanalys**: NLP-tekniker identifierar viktiga koncept, entiteter och definitioner i texten.
3. **Frågegenerering**: Baserat på det analyserade innehållet genererar applikationen lämpliga frågor av de valda typerna.
4. **Svarsförklaring**: Varje fråga kommer med en förklaring av varför svaret är korrekt.

## Projektarkitektur

- `start.py`: Huvudingångspunkt för applikationen
- `pdf_processor.py`: Hanterar PDF-textextraktion 
- `text_analyzer.py`: Utför NLP-analys på extraherad text
- `question_generator.py`: Genererar frågor baserat på analyserat innehåll
- `model_inference.py`: Hanterar inferens med den finetunade T5-modellen
- `optimize_for_cpu.py`: Optimerar modellen för CPU-inferens
- `generate_questions.py`: Skript för att generera frågor från en PDF

## License

MIT 