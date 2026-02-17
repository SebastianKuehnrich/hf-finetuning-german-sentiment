"""
Aufgabe 1.2: Fine-Tuning auf echten IMDB Daten
================================================
1.000 echte IMDB Reviews statt 20 handgeschriebene Saetze.
Mit compute_metrics (Accuracy + F1), Classification Report,
Confusion Matrix und Live-Test.

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging
import time

# --- Defensive Imports ---
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[FEHLER] pip install datasets")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        pipeline as hf_pipeline,
    )
except ImportError:
    sys.exit("[FEHLER] pip install transformers torch accelerate")

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
except ImportError:
    sys.exit("[FEHLER] pip install numpy scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[FEHLER] pip install matplotlib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================
# KONSTANTEN
# ============================================
MODEL_NAME = "distilbert-base-uncased"
TRAIN_SIZE = 500
TEST_SIZE = 500
MAX_LENGTH = 256
NUM_EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
LEARNING_RATE = 2e-5
SEED = 42
SAVE_PATH = "./mein_imdb_model"
CONFUSION_MATRIX_FILE = "confusion_matrix_imdb.png"


# ============================================
# SCHRITT 1: DATASET LADEN
# ============================================

def load_imdb_data():
    """Laedt IMDB und erstellt kleinere Subsets fuer CPU-Training."""
    print("=" * 60)
    print("SCHRITT 1: IMDB DATASET LADEN")
    print("=" * 60)

    dataset = load_dataset("stanfordnlp/imdb")

    # Nur 500 Training + 500 Test (CPU-freundlich)
    small_train = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
    small_test = dataset["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

    print(f"  Training:  {len(small_train)} Reviews")
    print(f"  Test:      {len(small_test)} Reviews")
    print(f"  Labels:    0=negativ, 1=positiv")

    # Label-Verteilung pruefen
    train_labels = small_train["label"]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"  Positiv:   {n_pos}")
    print(f"  Negativ:   {n_neg}")

    # Defensive: Warnung bei starkem Ungleichgewicht
    ratio = n_pos / n_neg if n_neg > 0 else float("inf")
    if ratio < 0.5 or ratio > 2.0:
        logger.warning("Dataset stark unbalanciert! Ratio pos/neg: %.2f", ratio)

    return small_train, small_test


# ============================================
# SCHRITT 2: TOKENIZER + TOKENISIEREN
# ============================================

def tokenize_data(train_data, test_data):
    """Tokenisiert die Datasets mit .map() und erstellt DataCollator."""
    print("\n" + "=" * 60)
    print("SCHRITT 2: TOKENISIERUNG")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize-Funktion fuer .map()
    # WICHTIG: Kein padding hier! DataCollator macht das pro Batch (effizienter)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)

    # DataCollator = Padding pro Batch (effizienter als alles auf max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"  Tokenizer: {MODEL_NAME}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  DataCollator: Dynamic Padding pro Batch")

    return tokenizer, tokenized_train, tokenized_test, data_collator


# ============================================
# SCHRITT 3: METRIKEN
# ============================================

def compute_metrics(eval_pred):
    """Berechnet Accuracy und F1-Score fuer den Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")

    return {"accuracy": acc, "f1": f1}


# ============================================
# SCHRITT 4: MODEL LADEN
# ============================================

def load_model():
    """Laedt DistilBERT mit Classification Head (2 Labels)."""
    print("\n" + "=" * 60)
    print("SCHRITT 4: MODEL LADEN")
    print("=" * 60)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model:      {MODEL_NAME}")
    print(f"  Parameter:  {total_params:,}")
    print(f"  Trainable:  {trainable:,}")

    return model


# ============================================
# SCHRITT 5: TRAINING
# ============================================

def train_model(model, tokenizer, tokenized_train, tokenized_test, data_collator):
    """Konfiguriert und startet das Fine-Tuning."""
    print("\n" + "=" * 60)
    print("SCHRITT 5: FINE-TUNING STARTET")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir="./imdb_finetuned",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    logger.info("Training abgeschlossen in %.1f Sekunden", elapsed)
    return trainer


# ============================================
# SCHRITT 6: EVALUATION
# ============================================

def evaluate_model(trainer, tokenized_test):
    """Evaluation mit Classification Report und Confusion Matrix."""
    print("\n" + "=" * 60)
    print("SCHRITT 6: EVALUATION AUF TEST-SET")
    print("=" * 60)

    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Defensive: Arrays muessen gleiche Laenge haben
    assert len(y_pred) == len(y_true), (
        f"Prediction/Label Mismatch: {len(y_pred)} vs {len(y_true)}"
    )

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]
    ))

    # Confusion Matrix plotten
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("IMDB Fine-Tuning - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    print(f"Confusion Matrix gespeichert: {CONFUSION_MATRIX_FILE}")

    return y_true, y_pred


# ============================================
# SCHRITT 7: MODEL SPEICHERN + LIVE-TEST
# ============================================

def save_and_test(model, tokenizer):
    """Speichert das Model und testet es mit der Pipeline API."""
    print("\n" + "=" * 60)
    print("SCHRITT 7: MODEL SPEICHERN + LIVE-TEST")
    print("=" * 60)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"  Model gespeichert: {SAVE_PATH}/")

    # Als Pipeline laden (Produktions-Modus)
    try:
        mein_classifier = hf_pipeline(
            "text-classification",
            model=SAVE_PATH,
            tokenizer=SAVE_PATH,
        )
    except Exception as exc:
        logger.error("Pipeline konnte nicht geladen werden: %s", exc)
        return

    test_saetze = [
        "This movie was absolutely incredible, best I've seen all year!",
        "Terrible waste of time, awful acting and boring plot.",
        "It was okay, nothing special but not terrible either.",
        "A masterpiece of cinema, truly groundbreaking filmmaking.",
        "I fell asleep halfway through, so predictable and dull.",
    ]

    print("\n" + "-" * 60)
    print("EUER MODEL - LIVE TEST")
    print("-" * 60)

    for satz in test_saetze:
        try:
            result = mein_classifier(satz)[0]
            bar = "#" * int(result["score"] * 20)
            print(f"  {result['label']:10s} [{bar:20s}] {result['score']:.1%}")
            print(f"  Text: {satz}\n")
        except Exception as exc:
            logger.warning("Fehler bei: %s -> %s", satz[:30], exc)


# ============================================
# MAIN
# ============================================

def main() -> None:
    """Hauptfunktion: Komplettes IMDB Fine-Tuning."""
    logger.info("IMDB Fine-Tuning Pipeline gestartet")
    pipeline_start = time.time()

    # 1. Daten laden
    train_data, test_data = load_imdb_data()

    # 2. Tokenisieren
    tokenizer, tok_train, tok_test, collator = tokenize_data(train_data, test_data)

    # 3. Model laden
    model = load_model()

    # 4. Training
    trainer = train_model(model, tokenizer, tok_train, tok_test, collator)

    # 5. Evaluation
    evaluate_model(trainer, tok_test)

    # 6. Speichern + Test
    save_and_test(model, tokenizer)

    elapsed = time.time() - pipeline_start
    logger.info("Gesamte Pipeline abgeschlossen in %.1f Sekunden", elapsed)


if __name__ == "__main__":
    main()
