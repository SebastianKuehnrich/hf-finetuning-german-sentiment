"""
Aufgabe 7 GOLD: Vision Transformer (ViT) auf CIFAR-10
======================================================
Laedt ein vortrainiertes ViT-Model (google/vit-base-patch16-224)
und klassifiziert 20 zufaellige CIFAR-10 Testbilder.
Erstellt ein 4x5 Dashboard mit Ergebnissen.

Kurs: Morphos GmbH - KI & Python Modul
"""

from __future__ import annotations

import sys
import logging
import time

# --- Defensive Imports ---
try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "[FEHLER] 'datasets' ist nicht installiert.\n"
        "  Loesung: pip install datasets"
    )

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
except ImportError:
    sys.exit(
        "[FEHLER] 'transformers' ist nicht installiert.\n"
        "  Loesung: pip install transformers"
    )

try:
    import torch
except ImportError:
    sys.exit(
        "[FEHLER] 'torch' ist nicht installiert.\n"
        "  Loesung: pip install torch"
    )

try:
    import numpy as np
    from numpy.random import RandomState
except ImportError:
    sys.exit(
        "[FEHLER] 'numpy' ist nicht installiert.\n"
        "  Loesung: pip install numpy"
    )

try:
    import matplotlib
    matplotlib.use("Agg")  # Nicht-interaktives Backend (Server-sicher)
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(
        "[FEHLER] 'matplotlib' ist nicht installiert.\n"
        "  Loesung: pip install matplotlib"
    )

# Logging statt nur print - professioneller Standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# KONSTANTEN
# ============================================
MODEL_NAME: str = "google/vit-base-patch16-224"
DATASET_NAME: str = "cifar10"
NUM_SAMPLES: int = 20
RANDOM_SEED: int = 42
GRID_ROWS: int = 4
GRID_COLS: int = 5
OUTPUT_IMAGE: str = "vit_cifar10_results.png"


# ============================================
# SCHRITT 1: CIFAR-10 DATASET LADEN
# ============================================

def load_cifar10_dataset():
    """
    Laedt das CIFAR-10 Dataset ueber Hugging Face datasets.

    Returns:
        dataset: Das geladene CIFAR-10 Dataset.
        label_names: Liste der Klassennamen aus den Dataset-Features.
    """
    print("=" * 60)
    print("SCHRITT 1: CIFAR-10 DATASET LADEN")
    print("=" * 60)

    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as exc:
        logger.error("CIFAR-10 konnte nicht geladen werden: %s", exc)
        sys.exit(1)

    # Label-Namen aus den Dataset-Features extrahieren
    label_names = dataset["test"].features["label"].names

    print(f"  Dataset:     {DATASET_NAME}")
    print(f"  Splits:      {list(dataset.keys())}")
    print(f"  Test-Bilder: {len(dataset['test'])}")
    print(f"  Klassen:     {len(label_names)} -> {label_names}")
    print(f"  Bild-Groesse: 32x32 Pixel (RGB)")

    return dataset, label_names


# ============================================
# SCHRITT 2: ViT MODEL + PROCESSOR LADEN
# ============================================

def load_vit_model():
    """
    Laedt das vortrainierte ViT-Model und den zugehoerigen Image Processor.

    Returns:
        model: ViTForImageClassification (vortrainiert auf ImageNet-21k, fine-tuned auf ImageNet-1k).
        processor: ViTImageProcessor (Resize, Normalisierung, etc.).
    """
    print("\n" + "=" * 60)
    print("SCHRITT 2: ViT MODEL LADEN")
    print("=" * 60)

    try:
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    except Exception as exc:
        logger.error("ViT Model konnte nicht geladen werden: %s", exc)
        sys.exit(1)

    # Model in Evaluation-Modus setzen (kein Dropout, kein BatchNorm-Update)
    model.eval()

    # Model-Info ausgeben
    total_params = sum(p.numel() for p in model.parameters())
    num_labels = model.config.num_labels

    print(f"  Model:       {MODEL_NAME}")
    print(f"  Parameter:   {total_params:,} (~{total_params / 1e6:.0f}M)")
    print(f"  Num Labels:  {num_labels} (ImageNet-1k Klassen)")
    print(f"  Patch Size:  16x16 Pixel")
    print(f"  Input Size:  224x224 Pixel")

    return model, processor


# ============================================
# SCHRITT 3: 20 ZUFAELLIGE TESTBILDER WAEHLEN
# ============================================

def select_random_samples(dataset, label_names: list[str]):
    """
    Waehlt 20 zufaellige Testbilder aus CIFAR-10 aus.

    Args:
        dataset: Das CIFAR-10 Dataset.
        label_names: Liste der Klassennamen.

    Returns:
        samples: Liste von Dicts mit 'image', 'true_label', 'true_name'.
    """
    print("\n" + "=" * 60)
    print(f"SCHRITT 3: {NUM_SAMPLES} ZUFAELLIGE TESTBILDER WAEHLEN")
    print("=" * 60)

    test_set = dataset["test"]
    total_test = len(test_set)

    # RandomState(42) fuer Reproduzierbarkeit (wie in Aufgabe gefordert)
    rng = RandomState(RANDOM_SEED)
    indices = rng.choice(total_test, size=NUM_SAMPLES, replace=False)

    samples = []
    for idx in indices:
        example = test_set[int(idx)]
        image = example["img"]
        true_label_id = example["label"]

        # Defensive: Label-ID im gueltigen Bereich
        if true_label_id < 0 or true_label_id >= len(label_names):
            logger.warning("Ungueltiger Label-Index %d, ueberspringe.", true_label_id)
            continue

        true_name = label_names[true_label_id]
        samples.append({
            "image": image,
            "true_label": true_label_id,
            "true_name": true_name,
            "index": int(idx),
        })

    print(f"  Indices:  {list(indices)}")
    print(f"  Samples:  {len(samples)} Bilder ausgewaehlt")

    # Verteilung der ausgewaehlten Klassen anzeigen
    class_counts: dict[str, int] = {}
    for s in samples:
        name = s["true_name"]
        class_counts[name] = class_counts.get(name, 0) + 1
    for name, count in sorted(class_counts.items()):
        print(f"    {name:>12s}: {count}x")

    return samples


# ============================================
# SCHRITT 4: PREDICTIONS MIT ViT
# ============================================

def predict_with_vit(model, processor, samples: list[dict]) -> list[dict]:
    """
    Fuehrt Predictions auf allen Samples mit dem ViT-Model durch.

    Fuer jedes Bild:
    - Image Processor wandelt 32x32 CIFAR-10 Bild in 224x224 ViT-Input um
    - Model gibt Logits fuer 1000 ImageNet-Klassen zurueck
    - Softmax liefert Confidence
    - id2label liefert vorhergesagten Klassennamen

    Args:
        model: Das ViT-Model.
        processor: Der ViT Image Processor.
        samples: Liste von Sample-Dicts.

    Returns:
        results: Liste von Result-Dicts mit Prediction-Infos.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 4: PREDICTIONS MIT ViT")
    print("=" * 60)

    results = []

    for i, sample in enumerate(samples):
        image = sample["image"]
        true_name = sample["true_name"]

        try:
            # Processor: Resize 32x32 -> 224x224, Normalisierung, Tensor-Konvertierung
            inputs = processor(images=image, return_tensors="pt")

            # Prediction ohne Gradient-Berechnung (spart Speicher)
            with torch.no_grad():
                outputs = model(**inputs)

            # Logits -> Softmax -> Confidence
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = logits.argmax(-1).item()
            confidence = probabilities[0, predicted_class_id].item()

            # Predicted Label aus model.config.id2label
            predicted_label = model.config.id2label.get(
                predicted_class_id, f"class_{predicted_class_id}"
            )

            # Korrektheit pruefen: true_label.lower() in predicted_label.lower()
            is_correct = true_name.lower() in predicted_label.lower()

            result = {
                "image": image,
                "true_name": true_name,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": is_correct,
                "index": sample["index"],
            }
            results.append(result)

            # Ergebnis ausgeben
            status = "OK" if is_correct else "XX"
            print(
                f"  {status} [{i + 1:2d}/{NUM_SAMPLES}] "
                f"True: {true_name:>10s} | "
                f"ViT: {predicted_label[:30]:>30s} | "
                f"Conf: {confidence:.1%}"
            )

        except Exception as exc:
            logger.warning(
                "Fehler bei Bild %d (idx=%d): %s",
                i + 1, sample["index"], exc,
            )
            # Fehlerhaftes Bild trotzdem in Ergebnisse aufnehmen
            results.append({
                "image": image,
                "true_name": true_name,
                "predicted_label": "FEHLER",
                "confidence": 0.0,
                "is_correct": False,
                "index": sample["index"],
            })

    return results


# ============================================
# SCHRITT 5: ACCURACY BERECHNEN + AUSGEBEN
# ============================================

def compute_accuracy(results: list[dict]) -> float:
    """
    Berechnet und gibt die Accuracy auf den 20 Testbildern aus.

    Args:
        results: Liste von Result-Dicts mit 'is_correct' Feld.

    Returns:
        accuracy: Accuracy als Prozentwert.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 5: ACCURACY")
    print("=" * 60)

    total = len(results)
    # Defensive: Division durch Null verhindern
    if total == 0:
        logger.warning("Keine Ergebnisse vorhanden, Accuracy = 0%")
        return 0.0

    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total * 100

    print(f"  Korrekt:    {correct}/{total}")
    print(f"  Accuracy:   {accuracy:.1f}%")

    # Aufschluesselung nach Klasse
    class_stats: dict[str, dict[str, int]] = {}
    for r in results:
        name = r["true_name"]
        if name not in class_stats:
            class_stats[name] = {"correct": 0, "total": 0}
        class_stats[name]["total"] += 1
        if r["is_correct"]:
            class_stats[name]["correct"] += 1

    print("\n  Pro Klasse:")
    for name, stats in sorted(class_stats.items()):
        cls_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
        print(f"    {name:>12s}: {stats['correct']}/{stats['total']} = {cls_acc:.0f}%")

    return accuracy


# ============================================
# SCHRITT 6: 4x5 DASHBOARD ERSTELLEN
# ============================================

def create_dashboard(results: list[dict], accuracy: float) -> None:
    """
    Erstellt ein 4x5 Grid-Dashboard mit allen 20 Bildern und Predictions.

    Jedes Subplot zeigt:
    - Das CIFAR-10 Bild
    - Titel: 'True: X / ViT: Y (confidence%)'
    - Gruen wenn korrekt, Rot wenn falsch

    Args:
        results: Liste von Result-Dicts.
        accuracy: Gesamte Accuracy als Prozentwert.
    """
    print("\n" + "=" * 60)
    print("SCHRITT 6: DASHBOARD ERSTELLEN")
    print("=" * 60)

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(20, 16),
    )
    fig.suptitle(
        f"ViT (google/vit-base-patch16-224) auf CIFAR-10 "
        f"| Accuracy: {accuracy:.1f}% ({sum(1 for r in results if r['is_correct'])}/{len(results)})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(results):
            ax.axis("off")
            continue

        result = results[i]
        image = result["image"]
        true_name = result["true_name"]
        predicted_label = result["predicted_label"]
        confidence = result["confidence"]
        is_correct = result["is_correct"]

        # Bild anzeigen
        ax.imshow(image)
        ax.axis("off")

        # Predicted Label kuerzen falls zu lang (ImageNet-Labels sind oft lang)
        short_pred = predicted_label
        if len(short_pred) > 25:
            short_pred = short_pred[:22] + "..."

        # Titel: True: X / ViT: Y (confidence%)
        title = f"True: {true_name}\nViT: {short_pred} ({confidence:.1%})"
        title_color = "green" if is_correct else "red"

        ax.set_title(
            title,
            fontsize=9,
            fontweight="bold",
            color=title_color,
            pad=4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"  Dashboard gespeichert: {OUTPUT_IMAGE}")


# ============================================
# MAIN
# ============================================

def main() -> None:
    """Hauptfunktion: Komplette ViT CIFAR-10 Pipeline."""
    logger.info("ViT CIFAR-10 Pipeline gestartet")
    pipeline_start = time.time()

    # 1. Dataset laden
    dataset, label_names = load_cifar10_dataset()

    # 2. Model + Processor laden
    model, processor = load_vit_model()

    # 3. Zufaellige Testbilder waehlen
    samples = select_random_samples(dataset, label_names)

    # 4. Predictions
    results = predict_with_vit(model, processor, samples)

    # 5. Accuracy
    accuracy = compute_accuracy(results)

    # 6. Dashboard
    create_dashboard(results, accuracy)

    # Gesamtzeit
    elapsed = time.time() - pipeline_start
    logger.info("Pipeline abgeschlossen in %.1f Sekunden", elapsed)

    print(f"\n{'=' * 60}")
    print("  ViT CIFAR-10 ANALYSE ABGESCHLOSSEN")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Dashboard: {OUTPUT_IMAGE}")
    print(f"  Gesamtzeit: {elapsed:.1f} Sekunden")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


# ============================================
# VERSTAENDNIS-FRAGEN
# ============================================

# FRAGE 1: Wie viele Patches entstehen aus einem 224x224 Bild bei 16x16 Patch-Groesse?
#
#    224 / 16 = 14 Patches pro Achse (horizontal und vertikal)
#    14 x 14 = 196 Patches insgesamt
#
#    Jeder Patch ist 16x16x3 = 768 Werte (RGB). Diese 768 Werte werden als
#    Embedding-Vektor in den Transformer gefuettert. Dazu kommt noch ein
#    spezieller [CLS]-Token am Anfang -> also 197 Tokens insgesamt.
#    Das ist analog zum [CLS]-Token bei BERT fuer Text.

# FRAGE 2: CIFAR-10 Bilder sind 32x32, ViT erwartet 224x224.
#           Was macht der Processor? Gibt es Qualitaetsprobleme?
#
#    Der ViTImageProcessor skaliert (resized) das 32x32 Bild auf 224x224
#    mittels Interpolation (bilinear). Danach normalisiert er die Pixelwerte
#    auf den Bereich den das Model erwartet (ImageNet Mean/Std).
#
#    QUALITAETSPROBLEM: Ja! Ein 32x32 Bild auf 224x224 hochzuskalieren
#    bedeutet 7x Vergroesserung. Das Bild wird unscharf/pixelig weil
#    keine echten Details hinzugefuegt werden koennen (Information geht
#    nicht verloren, aber es wird auch keine neue gewonnen).
#    Trotzdem funktioniert es erstaunlich gut, weil ViT auf die
#    SEMANTISCHEN Merkmale achtet (Form, Farbe, Kontext), nicht auf
#    Pixel-Details. Die groben Strukturen eines Autos oder Hundes sind
#    auch bei 32x32 erkennbar.

# FRAGE 3: ViT hat ~86M Parameter, ein typisches CNN fuer CIFAR-10 hat ~500K.
#           Wann ist welches besser?
#
#    ViT (86M Parameter) ist besser wenn:
#    - VIELE Trainingsdaten vorhanden (>1M Bilder, z.B. ImageNet-21k)
#    - Komplexe Aufgaben mit vielen Klassen
#    - Transfer Learning / vortrainierte Models genutzt werden
#    - GPU/TPU mit genuegend Speicher verfuegbar
#    - Globale Zusammenhaenge im Bild wichtig sind (Attention sieht alles)
#
#    CNN (500K Parameter) ist besser wenn:
#    - WENIGE Trainingsdaten (< 10.000 Bilder)
#    - Einfachere Aufgaben (10 Klassen wie CIFAR-10)
#    - Begrenzte Rechenressourcen (CPU, Edge-Devices, Smartphone)
#    - Schnelle Inferenz noetig (Echtzeit-Anwendungen)
#    - Lokale Muster wichtiger als globale (Texturen, Kanten)
#
#    GRUNDREGEL: ViT braucht MEHR Daten um gut zu werden, aber mit
#    genug Daten (oder Pretraining) uebertrifft es CNNs deutlich.
#    CNN hat einen "inductive bias" (Lokalitaet, Translation Invariance)
#    der bei wenig Daten hilft. ViT muss das alles aus Daten lernen.

# FRAGE 4: ViT ist auf ImageNet (1000 Klassen) trainiert, CIFAR-10 hat nur 10 Klassen.
#           Funktioniert das trotzdem? Warum?
#
#    JA, es funktioniert erstaunlich gut! Gruende:
#
#    1. UEBERLAPPENDE KLASSEN: Viele CIFAR-10 Klassen existieren auch
#       in ImageNet. "automobile" -> ImageNet hat "sports car", "minivan",
#       "pickup truck" etc. "dog" -> ImageNet hat 120+ Hunderassen.
#       Die Pruefung true_label.lower() in predicted_label.lower()
#       findet diese Uebereinstimmungen.
#
#    2. TRANSFER LEARNING: Das Model hat gelernt WAS ein Auto, Hund,
#       Flugzeug etc. IST. Diese Features (Raeder, Fell, Fluegel)
#       sind universell und uebertragbar.
#
#    3. HIERARCHISCHE FEATURES: Die fruehen Layers erkennen Kanten,
#       Texturen, Formen. Die spaeteren Layers erkennen Objekte.
#       Diese Feature-Hierarchie ist fuer viele Aufgaben nuetzlich.
#
#    4. EINSCHRAENKUNG: Es funktioniert NICHT perfekt weil:
#       - ImageNet hat FEINERE Klassen (120 Hunderassen vs 1 "dog")
#       - Das Model sagt z.B. "tabby cat" statt "cat" -> Match klappt
#         nur wenn wir .lower() + "in" Pruefung machen
#       - Manche CIFAR-10 Klassen wie "automobile" heissen in ImageNet
#         anders ("sports car") -> kann zu Fehlern fuehren
#       - Fuer optimale Ergebnisse muesste man ViT auf CIFAR-10 fine-tunen
