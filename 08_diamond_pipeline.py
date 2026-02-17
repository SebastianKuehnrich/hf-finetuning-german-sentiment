"""
Aufgabe Diamond: Komplette KI-Pipeline mit allen Phasen
========================================================
Phase 1: Fine-Tuning auf echten IMDB Daten (1.000 Train + 500 Test)
Phase 2: Volle Evaluation (Classification Report + Confusion Matrix)
Phase 3: Training Curves (Loss, Accuracy, F1 ueber Epochs)
Phase 4: Vision Transformer (ViT) auf CIFAR-10
Phase 5: Dashboard mit 4 Plots (2x2 Grid)
Phase 6: Model speichern + laden + finaler Test

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging
import time

# ============================================
# DEFENSIVE IMPORTS
# ============================================
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
        ViTForImageClassification,
        ViTImageProcessor,
        pipeline as hf_pipeline,
    )
except ImportError:
    sys.exit("[FEHLER] pip install transformers torch accelerate")

try:
    import numpy as np
except ImportError:
    sys.exit("[FEHLER] pip install numpy")

try:
    import torch
except ImportError:
    sys.exit("[FEHLER] pip install torch")

try:
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
    sys.exit("[FEHLER] pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except ImportError:
    sys.exit("[FEHLER] pip install matplotlib")

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================
# KONSTANTEN
# ============================================
MODEL_NAME: str = "distilbert-base-uncased"
VIT_MODEL_NAME: str = "google/vit-base-patch16-224"
TRAIN_SIZE: int = 1000
TEST_SIZE: int = 500
MAX_LENGTH: int = 256
NUM_EPOCHS: int = 3
TRAIN_BATCH: int = 8
EVAL_BATCH: int = 16
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
SEED: int = 42
LOGGING_STEPS: int = 10
VIT_SAMPLE_COUNT: int = 8
OUTPUT_DIR: str = "./diamond_model"
SAVE_PATH: str = "./diamond_final_model"
DASHBOARD_FILE: str = "diamond_dashboard.png"
TARGET_NAMES: list[str] = ["NEGATIVE", "POSITIVE"]


# ============================================
# PHASE 1: FINE-TUNING
# ============================================

def phase1_finetuning() -> tuple:
    """Fine-Tuning von DistilBERT auf IMDB Reviews.

    Laedt 1.000 Train + 500 Test Reviews, tokenisiert mit .map(),
    trainiert 3 Epochs mit compute_metrics (Accuracy, F1, Precision, Recall).

    Returns:
        tuple: (trainer, tokenizer, tokenized_test, model) fuer weitere Phasen.
    """
    print("=" * 60)
    print("PHASE 1: FINE-TUNING")
    print("=" * 60)

    # --- Dataset laden ---
    try:
        dataset = load_dataset("stanfordnlp/imdb")
    except Exception as exc:
        logger.error("IMDB konnte nicht geladen werden: %s", exc)
        sys.exit(1)

    train_data = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
    test_data = dataset["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

    print(f"  Training:  {len(train_data)} Reviews")
    print(f"  Test:      {len(test_data)} Reviews")

    # Defensive: Label-Verteilung pruefen
    train_labels = train_data["label"]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"  Positiv:   {n_pos}")
    print(f"  Negativ:   {n_neg}")

    ratio = n_pos / n_neg if n_neg > 0 else float("inf")
    if ratio < 0.5 or ratio > 2.0:
        logger.warning("Dataset stark unbalanciert! Ratio pos/neg: %.2f", ratio)

    # --- Tokenizer + Tokenisierung ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples: dict) -> dict:
        """Tokenisiert Texte mit Truncation, ohne Padding (DataCollator macht das)."""
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_train = train_data.map(tokenize_fn, batched=True)
    tokenized_test = test_data.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"  Tokenizer: {MODEL_NAME}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  DataCollator: Dynamic Padding pro Batch")

    # --- Model laden ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameter:  {total_params:,}")
    print(f"  Trainable:  {trainable:,}")

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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

    print("\n  Fine-Tuning startet...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    logger.info("Phase 1 abgeschlossen in %.1f Sekunden", elapsed)

    return trainer, tokenizer, tokenized_test, model


def compute_metrics(eval_pred) -> dict[str, float]:
    """Berechnet Accuracy, F1, Precision und Recall fuer den Trainer.

    Args:
        eval_pred: Tuple aus (logits, labels) vom Trainer.

    Returns:
        dict: Metriken als Key-Value Paare.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }


# ============================================
# PHASE 2: EVALUATION
# ============================================

def phase2_evaluation(
    trainer: Trainer,
    tokenized_test,
) -> tuple[np.ndarray, np.ndarray]:
    """Volle Evaluation mit Classification Report.

    Args:
        trainer: Der trainierte Trainer.
        tokenized_test: Das tokenisierte Test-Dataset.

    Returns:
        tuple: (y_true, y_pred) fuer Confusion Matrix in Phase 5.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: EVALUATION")
    print("=" * 60)

    try:
        predictions = trainer.predict(tokenized_test)
    except Exception as exc:
        logger.error("Prediction fehlgeschlagen: %s", exc)
        sys.exit(1)

    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Defensive: Arrays muessen gleiche Laenge haben
    assert len(y_pred) == len(y_true), (
        f"Prediction/Label Mismatch: {len(y_pred)} vs {len(y_true)}"
    )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    return y_true, y_pred


# ============================================
# PHASE 3: TRAINING CURVES EXTRAHIEREN
# ============================================

def phase3_training_curves(trainer: Trainer) -> dict:
    """Extrahiert Training-Metriken aus dem Trainer fuer das Dashboard.

    Trennt log_history in:
    - train_loss_steps: Eintraege mit "loss" aber ohne "eval_loss"
    - eval_entries: Eintraege mit "eval_loss"
    - eval_accuracy und eval_f1 pro Epoch

    Args:
        trainer: Der trainierte Trainer mit log_history.

    Returns:
        dict: Alle extrahierten Metriken fuer das Dashboard.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING CURVES")
    print("=" * 60)

    history = trainer.state.log_history

    # Train Loss: Eintraege mit "loss" aber NICHT "eval_loss"
    train_loss_steps = [
        (entry["step"], entry["loss"])
        for entry in history
        if "loss" in entry and "eval_loss" not in entry
    ]

    # Eval Eintraege: Alles mit "eval_loss"
    eval_entries = [entry for entry in history if "eval_loss" in entry]

    # Entpacken (defensiv bei leeren Listen)
    if train_loss_steps:
        train_steps, train_losses = zip(*train_loss_steps)
        train_steps = list(train_steps)
        train_losses = list(train_losses)
    else:
        train_steps, train_losses = [], []
        logger.warning("Keine Train-Loss Eintraege gefunden!")

    eval_steps = [entry["step"] for entry in eval_entries]
    eval_losses = [entry["eval_loss"] for entry in eval_entries]
    eval_accs = [entry.get("eval_accuracy", 0.0) for entry in eval_entries]
    eval_f1s = [entry.get("eval_f1", 0.0) for entry in eval_entries]

    print(f"  Train Loss Eintraege: {len(train_loss_steps)}")
    print(f"  Eval Eintraege:       {len(eval_entries)}")

    for i, entry in enumerate(eval_entries):
        epoch = entry.get("epoch", i + 1)
        print(
            f"    Epoch {epoch:.0f}: "
            f"Loss={entry['eval_loss']:.4f}, "
            f"Acc={entry.get('eval_accuracy', 0.0):.4f}, "
            f"F1={entry.get('eval_f1', 0.0):.4f}"
        )

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
        "eval_accs": eval_accs,
        "eval_f1s": eval_f1s,
    }


# ============================================
# PHASE 4: ViT AUF CIFAR-10
# ============================================

def phase4_vit_cifar10() -> list[dict]:
    """Klassifiziert 8 zufaellige CIFAR-10 Bilder mit Vision Transformer.

    Laedt google/vit-base-patch16-224 und CIFAR-10 Dataset.
    Waehlt 8 Bilder mit RandomState(42), klassifiziert mit torch.no_grad().

    Returns:
        list[dict]: Ergebnisse pro Bild (image, true, predicted, confidence).
    """
    print("\n" + "=" * 60)
    print("PHASE 4: ViT AUF CIFAR-10")
    print("=" * 60)

    try:
        cifar = load_dataset("cifar10")
    except Exception as exc:
        logger.error("CIFAR-10 konnte nicht geladen werden: %s", exc)
        return []

    try:
        vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
        vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
    except Exception as exc:
        logger.error("ViT Model konnte nicht geladen werden: %s", exc)
        return []

    vit_model.eval()

    cifar_labels = cifar["test"].features["label"].names
    total_test = len(cifar["test"])
    sample_indices = np.random.RandomState(SEED).choice(
        total_test, VIT_SAMPLE_COUNT, replace=False
    )

    print(f"  Model:      {VIT_MODEL_NAME}")
    print(f"  Parameter:  {sum(p.numel() for p in vit_model.parameters()):,}")
    print(f"  CIFAR-10:   {total_test} Test-Bilder")
    print(f"  Samples:    {VIT_SAMPLE_COUNT} zufaellige Bilder")
    print(f"  Klassen:    {cifar_labels}")

    vit_results: list[dict] = []
    for idx in sample_indices:
        example = cifar["test"][int(idx)]
        image = example["img"]
        true_label = cifar_labels[example["label"]]

        inputs = vit_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)

        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        predicted_label = vit_model.config.id2label[predicted_class]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()

        vit_results.append({
            "image": image,
            "true": true_label,
            "predicted": predicted_label,
            "confidence": confidence,
            "correct": true_label.lower() in predicted_label.lower(),
        })

        status = "OK" if vit_results[-1]["correct"] else "FALSCH"
        print(
            f"    [{status:>5s}] True: {true_label:<10s} "
            f"-> ViT: {predicted_label:<25s} ({confidence:.1%})"
        )

    correct_count = sum(1 for r in vit_results if r["correct"])
    print(f"\n  ViT Accuracy: {correct_count}/{VIT_SAMPLE_COUNT}")

    return vit_results


# ============================================
# PHASE 5: DASHBOARD (2x2 PLOTS)
# ============================================

def phase5_dashboard(
    curves: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vit_results: list[dict],
) -> None:
    """Erstellt ein 2x2 Dashboard und speichert es als PNG.

    Plot 1 (oben links):  Train Loss + Eval Loss ueber Steps
    Plot 2 (oben rechts): Confusion Matrix (ConfusionMatrixDisplay)
    Plot 3 (unten links): Accuracy & F1 als Balkendiagramm pro Epoch
    Plot 4 (unten rechts): ViT CIFAR-10 Predictions (8 Mini-Bilder mit inset_axes)

    Args:
        curves: Dict mit train_steps, train_losses, eval_steps, eval_losses, eval_accs, eval_f1s.
        y_true: Wahre Labels aus Phase 2.
        y_pred: Vorhergesagte Labels aus Phase 2.
        vit_results: Liste der ViT-Ergebnisse aus Phase 4.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: DASHBOARD")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Diamond Pipeline - Complete Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    # --- Plot 1: Train Loss + Eval Loss ---
    ax1 = axes[0, 0]
    ax1.plot(
        curves["train_steps"], curves["train_losses"],
        "b-", alpha=0.5, label="Train Loss",
    )
    ax1.plot(
        curves["eval_steps"], curves["eval_losses"],
        "r-o", label="Eval Loss",
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Eval Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Confusion Matrix ---
    ax2 = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NEG", "POS"]
    )
    disp.plot(ax=ax2, cmap="Blues", values_format="d")
    ax2.set_title("Confusion Matrix")

    # --- Plot 3: Accuracy & F1 pro Epoch (Balkendiagramm) ---
    ax3 = axes[1, 0]
    num_epochs = len(curves["eval_accs"])
    if num_epochs > 0:
        epochs = list(range(1, num_epochs + 1))
        bar_width = 0.3
        x_acc = [e - bar_width / 2 for e in epochs]
        x_f1 = [e + bar_width / 2 for e in epochs]

        ax3.bar(x_acc, curves["eval_accs"], bar_width, label="Accuracy", color="steelblue")
        ax3.bar(x_f1, curves["eval_f1s"], bar_width, label="F1", color="coral")

        # Werte ueber den Balken anzeigen
        for i, (acc, f1) in enumerate(zip(curves["eval_accs"], curves["eval_f1s"])):
            ax3.text(x_acc[i], acc + 0.01, f"{acc:.2f}", ha="center", fontsize=8)
            ax3.text(x_f1[i], f1 + 0.01, f"{f1:.2f}", ha="center", fontsize=8)

        ax3.set_xticks(epochs)
        ax3.set_xticklabels([f"Epoch {e}" for e in epochs])
    else:
        ax3.text(0.5, 0.5, "Keine Eval-Daten", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.set_title("Accuracy & F1 pro Epoch")
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Plot 4: ViT CIFAR-10 Predictions (8 Mini-Bilder) ---
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("ViT - CIFAR-10 Predictions")

    if vit_results:
        num_images = min(len(vit_results), 8)
        cols = 4
        rows = 2
        for i in range(num_images):
            r = vit_results[i]
            row = i // cols
            col = i % cols

            # inset_axes: [x, y, width, height] relativ zum Parent-Axes
            x_pos = col * 0.25
            y_pos = (1 - row) * 0.45 + 0.05
            inset = inset_axes(
                ax4,
                width="20%",
                height="40%",
                loc="lower left",
                bbox_to_anchor=(x_pos, y_pos, 0.25, 0.45),
                bbox_transform=ax4.transAxes,
                borderpad=0,
            )
            inset.imshow(r["image"])
            color = "green" if r["correct"] else "red"
            # Kuerze das Label auf max 15 Zeichen
            short_label = r["predicted"][:15]
            inset.set_title(
                f"{short_label}\n({r['confidence']:.0%})",
                fontsize=7,
                color=color,
            )
            inset.axis("off")
    else:
        ax4.text(
            0.5, 0.5, "Keine ViT-Ergebnisse",
            ha="center", va="center", transform=ax4.transAxes,
        )

    plt.tight_layout()
    plt.savefig(DASHBOARD_FILE, dpi=150)
    print(f"  Dashboard gespeichert: {DASHBOARD_FILE}")
    logger.info("Dashboard erstellt: %s", DASHBOARD_FILE)


# ============================================
# PHASE 6: SAVE + LOAD + FINAL TEST
# ============================================

def phase6_save_and_test(model, tokenizer) -> None:
    """Speichert das Model, laedt es als Pipeline und testet 3 Saetze.

    Args:
        model: Das trainierte Model.
        tokenizer: Der Tokenizer.
    """
    print("\n" + "=" * 60)
    print("PHASE 6: SAVE & FINAL TEST")
    print("=" * 60)

    # --- Model + Tokenizer speichern ---
    try:
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  Model gespeichert: {SAVE_PATH}/")
    except Exception as exc:
        logger.error("Model konnte nicht gespeichert werden: %s", exc)
        return

    # --- Als Pipeline laden (Produktions-Modus) ---
    try:
        final_clf = hf_pipeline(
            "text-classification",
            model=SAVE_PATH,
            tokenizer=SAVE_PATH,
        )
    except Exception as exc:
        logger.error("Pipeline konnte nicht geladen werden: %s", exc)
        return

    # --- Finaler Test mit 3 Saetzen ---
    final_tests: list[str] = [
        "This movie was absolutely phenomenal, a true cinematic masterpiece!",
        "Terrible film, I want my money and two hours of my life back.",
        "Average movie with some good moments but overall forgettable.",
    ]

    print("\n" + "-" * 60)
    print("FINALER LIVE-TEST")
    print("-" * 60)

    for text in final_tests:
        try:
            result = final_clf(text)[0]
            # "#" fuer Confidence-Bars (nicht Unicode-Bloecke)
            bar_length = int(result["score"] * 20)
            bar = "#" * bar_length
            print(f"  {result['label']:10s} [{bar:20s}] {result['score']:.1%}")
            print(f"  Text: {text}\n")
        except Exception as exc:
            logger.warning("Fehler bei: %s -> %s", text[:30], exc)


# ============================================
# MAIN
# ============================================

def main() -> None:
    """Hauptfunktion: Komplette Diamond Pipeline mit 6 Phasen."""
    logger.info("Diamond Pipeline gestartet")
    pipeline_start = time.time()

    # Phase 1: Fine-Tuning
    trainer, tokenizer, tokenized_test, model = phase1_finetuning()

    # Phase 2: Evaluation
    y_true, y_pred = phase2_evaluation(trainer, tokenized_test)

    # Phase 3: Training Curves extrahieren
    curves = phase3_training_curves(trainer)

    # Phase 4: ViT auf CIFAR-10
    vit_results = phase4_vit_cifar10()

    # Phase 5: Dashboard (2x2 Plots)
    phase5_dashboard(curves, y_true, y_pred, vit_results)

    # Phase 6: Save + Load + Final Test
    phase6_save_and_test(model, tokenizer)

    # --- Zusammenfassung ---
    elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("DIAMOND PIPELINE KOMPLETT")
    print("=" * 60)
    print(f"  Gesamtzeit: {elapsed:.1f} Sekunden ({elapsed / 60:.1f} Minuten)")
    print(f"  Dashboard:  {DASHBOARD_FILE}")
    print(f"  Model:      {SAVE_PATH}/")
    logger.info("Diamond Pipeline abgeschlossen in %.1f Sekunden", elapsed)


if __name__ == "__main__":
    main()
