
"""
Aufgabe 6 (SILVER): Fine-Tuning deepset/gbert-base auf deutschen Amazon Reviews
=================================================================================
Echte deutsche Amazon-Reviews (mteb/amazon_reviews_multi, "de") werden
binarisiert (negativ / positiv) und ein deutsches BERT-Modell wird
darauf feingetunt. Anschliessend: Classification Report, Confusion
Matrix und Live-Test mit der Pipeline API.

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging
import time
from typing import Any

# ---------------------------------------------------------------------------
# Defensive Imports
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, DatasetDict
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
        precision_score,
        recall_score,
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================
# KONSTANTEN
# ============================================
MODEL_NAME: str = "deepset/gbert-base"
DATASET_NAME: str = "mteb/amazon_reviews_multi"
DATASET_LANG: str = "de"

TRAIN_SIZE: int = 2000
TEST_SIZE: int = 500
MAX_LENGTH: int = 256
NUM_EPOCHS: int = 3
TRAIN_BATCH: int = 8
EVAL_BATCH: int = 16
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
LOGGING_STEPS: int = 50
SEED: int = 42

SAVE_PATH: str = "./mein_deutsches_model"
CONFUSION_MATRIX_FILE: str = "confusion_matrix_german.png"

LABEL_NAMES: list[str] = ["NEGATIV", "POSITIV"]
ID2LABEL: dict[int, str] = {0: "NEGATIV", 1: "POSITIV"}
LABEL2ID: dict[str, int] = {"NEGATIV": 0, "POSITIV": 1}


# ============================================
# SCHRITT 1: DATASET LADEN + BINARISIEREN
# ============================================

def binarize_labels(example: dict[str, Any]) -> dict[str, Any]:
    """Binarisiert die 5-Sterne-Skala (0-4) zu negativ/positiv.

    Mapping:
        0, 1   -> 0  (NEGATIV)   -- 1-2 Sterne
        3, 4   -> 1  (POSITIV)   -- 4-5 Sterne
        2      -> -1 (NEUTRAL)   -- wird danach rausgefiltert
    """
    star = example["label"]
    if star <= 1:
        example["label"] = 0
    elif star >= 3:
        example["label"] = 1
    else:
        example["label"] = -1
    return example


def load_amazon_data() -> tuple:
    """Laedt das deutsche Amazon-Reviews-Dataset und binarisiert die Labels."""
    print("=" * 60)
    print("SCHRITT 1: DEUTSCHES AMAZON DATASET LADEN")
    print("=" * 60)

    # Deutsche Parquet-Dateien direkt per URL laden.
    # Der Parquet-Branch hat Unterordner pro Sprache (de/, en/, ...),
    # aber keine 'language'-Spalte. Wir laden nur den de/-Ordner.
    BASE_URL = (
        "https://huggingface.co/datasets/mteb/amazon_reviews_multi"
        "/resolve/refs%2Fconvert%2Fparquet/de"
    )
    try:
        dataset = load_dataset(
            "parquet",
            data_files={
                "train":      f"{BASE_URL}/train/0000.parquet",
                "test":       f"{BASE_URL}/test/0000.parquet",
                "validation": f"{BASE_URL}/validation/0000.parquet",
            },
        )
    except Exception as exc:
        logger.error(
            "Amazon-DE Dataset konnte nicht geladen werden: %s", exc
        )
        sys.exit(1)

    logger.info(
        "Deutsches Dataset geladen: %d Train, %d Test",
        len(dataset["train"]),
        len(dataset["test"]),
    )

    # --- Label-Verteilung VOR Binarisierung ---
    raw_labels = dataset["train"]["label"]
    print("\n  Urspruengliche Sterne-Verteilung (Train DE):")
    for star in range(5):
        count = sum(1 for lbl in raw_labels if lbl == star)
        bar = "#" * (count // 500)
        print(f"    {star + 1} Stern(e): {count:>6d} {bar}")

    # --- Binarisieren ---
    dataset = dataset.map(binarize_labels)

    # Neutrale (Stern 3 = intern 2 -> jetzt -1) rausfiltern
    dataset = dataset.filter(lambda x: x["label"] != -1)

    logger.info(
        "Nach Binarisierung: %d Train, %d Test (neutral entfernt)",
        len(dataset["train"]),
        len(dataset["test"]),
    )

    # --- Subsets fuer CPU-freundliches Training ---
    small_train = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
    small_test = dataset["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

    # --- Label-Verteilung NACH Binarisierung ---
    train_labels = small_train["label"]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos

    print(f"\n  Training-Subset:  {len(small_train)} Reviews")
    print(f"  Test-Subset:      {len(small_test)} Reviews")
    print(f"  POSITIV:          {n_pos}")
    print(f"  NEGATIV:          {n_neg}")

    # Defensive: Warnung bei starkem Ungleichgewicht
    ratio = n_pos / n_neg if n_neg > 0 else float("inf")
    if ratio < 0.5 or ratio > 2.0:
        logger.warning("Dataset stark unbalanciert! Ratio pos/neg: %.2f", ratio)

    # Defensive: Sicherstellen dass nur 0 und 1 als Labels vorkommen
    unique_train = set(small_train["label"])
    unique_test = set(small_test["label"])
    assert unique_train.issubset({0, 1}), (
        f"Unerwartete Labels im Training: {unique_train}"
    )
    assert unique_test.issubset({0, 1}), (
        f"Unerwartete Labels im Test: {unique_test}"
    )

    return small_train, small_test


# ============================================
# SCHRITT 2: TOKENIZER + TOKENISIEREN
# ============================================

def tokenize_data(train_data, test_data):
    """Tokenisiert die Datasets mit .map(batched=True) und erstellt DataCollator.

    WICHTIG: Kein Padding hier! DataCollatorWithPadding macht das
    dynamisch pro Batch (effizienter als alles auf max_length zu padden).

    Returns:
        tokenizer, tokenized_train, tokenized_test, data_collator
    """
    print("\n" + "=" * 60)
    print("SCHRITT 2: TOKENISIERUNG")
    print("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        logger.error("Tokenizer konnte nicht geladen werden: %s", exc)
        sys.exit(1)

    def tokenize_function(examples: dict[str, list]) -> dict[str, list]:
        """Tokenisiert eine Batch von Texten (kein Padding!)."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)

    # DataCollator = dynamisches Padding pro Batch (effizienter)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"  Tokenizer:    {MODEL_NAME}")
    print(f"  Max Length:   {MAX_LENGTH}")
    print(f"  DataCollator: Dynamic Padding pro Batch")
    print(f"  Train Tokens: {len(tokenized_train)} Beispiele")
    print(f"  Test Tokens:  {len(tokenized_test)} Beispiele")

    return tokenizer, tokenized_train, tokenized_test, data_collator


# ============================================
# SCHRITT 3: METRIKEN
# ============================================

def compute_metrics(eval_pred) -> dict[str, float]:
    """Berechnet Accuracy, F1, Precision und Recall fuer den Trainer.

    Args:
        eval_pred: EvalPrediction mit (logits, label_ids)

    Returns:
        Dict mit accuracy, f1, precision, recall
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    prec = precision_score(labels, predictions, average="binary")
    rec = recall_score(labels, predictions, average="binary")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


# ============================================
# SCHRITT 4: MODEL LADEN
# ============================================

def load_model() -> Any:
    """Laedt deepset/gbert-base mit Classification Head (2 Labels).

    Returns:
        AutoModelForSequenceClassification Instanz
    """
    print("\n" + "=" * 60)
    print("SCHRITT 4: DEUTSCHES MODEL LADEN")
    print("=" * 60)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    except Exception as exc:
        logger.error("Model konnte nicht geladen werden: %s", exc)
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model:      {MODEL_NAME}")
    print(f"  Parameter:  {total_params:,}")
    print(f"  Trainable:  {trainable:,}")
    print(f"  Labels:     {ID2LABEL}")

    return model


# ============================================
# SCHRITT 5: TRAINING
# ============================================

def train_model(model, tokenizer, tokenized_train, tokenized_test, data_collator):
    """Konfiguriert und startet das Fine-Tuning.

    Args:
        model:           Das vortrainierte Model
        tokenizer:       Der Tokenizer
        tokenized_train: Tokenisiertes Trainings-Dataset
        tokenized_test:  Tokenisiertes Test-Dataset
        data_collator:   DataCollatorWithPadding

    Returns:
        Trainierter Trainer
    """
    print("\n" + "=" * 60)
    print("SCHRITT 5: FINE-TUNING STARTET")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir="./german_finetuned",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=LOGGING_STEPS,
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

    print(f"  Epochs:       {NUM_EPOCHS}")
    print(f"  Train Batch:  {TRAIN_BATCH}")
    print(f"  Eval Batch:   {EVAL_BATCH}")
    print(f"  LR:           {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Best Metric:  f1")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    logger.info("Training abgeschlossen in %.1f Sekunden", elapsed)
    return trainer


# ============================================
# SCHRITT 6: EVALUATION
# ============================================

def evaluate_model(trainer, tokenized_test):
    """Evaluation mit Classification Report und Confusion Matrix PNG.

    Args:
        trainer:        Der trainierte Trainer
        tokenized_test: Das tokenisierte Test-Dataset

    Returns:
        Tuple (y_true, y_pred)
    """
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

    # Classification Report (Accuracy, Precision, Recall, F1 pro Klasse)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # Confusion Matrix plotten und speichern
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=LABEL_NAMES
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("German Amazon Reviews - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close(fig)
    print(f"Confusion Matrix gespeichert: {CONFUSION_MATRIX_FILE}")

    return y_true, y_pred


# ============================================
# SCHRITT 7: MODEL SPEICHERN + LIVE-TEST
# ============================================

def save_and_test(model, tokenizer) -> None:
    """Speichert das Model und testet es mit der Pipeline API.

    Args:
        model:     Das feingetunte Model
        tokenizer: Der dazugehoerige Tokenizer
    """
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

    # 5 deutsche Test-Saetze
    test_saetze: list[str] = [
        "Absolut fantastisches Produkt, bin total begeistert und wuerde es sofort wieder kaufen!",
        "Totaler Schrott, nach zwei Tagen schon kaputt gegangen. Finger weg!",
        "Ganz solide Qualitaet fuer den Preis, erfuellt seinen Zweck ohne Probleme.",
        "Die Lieferung hat ewig gedauert und die Verpackung war komplett beschaedigt.",
        "Beste Kopfhoerer die ich je hatte, der Klang ist einfach unglaublich klar!",
    ]

    print("\n" + "-" * 60)
    print("EUER DEUTSCHES MODEL - LIVE TEST")
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
    """Hauptfunktion: Komplettes deutsches Fine-Tuning auf Amazon Reviews."""
    logger.info("German Fine-Tuning Pipeline gestartet")
    pipeline_start = time.time()

    # 1. Daten laden + binarisieren
    train_data, test_data = load_amazon_data()

    # 2. Tokenisieren
    tokenizer, tok_train, tok_test, collator = tokenize_data(train_data, test_data)

    # 3. Model laden
    model = load_model()

    # 4. Training
    trainer = train_model(model, tokenizer, tok_train, tok_test, collator)

    # 5. Evaluation
    evaluate_model(trainer, tok_test)

    # 6. Speichern + Live-Test
    save_and_test(model, tokenizer)

    elapsed = time.time() - pipeline_start
    logger.info("Gesamte Pipeline abgeschlossen in %.1f Sekunden", elapsed)

    print("\n" + "=" * 60)
    print("DEUTSCHES FINE-TUNING ABGESCHLOSSEN")
    print("=" * 60)


if __name__ == "__main__":
    main()
