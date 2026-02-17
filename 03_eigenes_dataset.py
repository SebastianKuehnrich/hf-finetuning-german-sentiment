"""
Aufgabe 1.3: Eigenes deutsches Dataset + deutsches Model
=========================================================
20+ selbstgeschriebene deutsche Texte (Restaurant-Reviews).
Fine-Tuning mit deepset/gbert-base (deutsches BERT).

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    from sklearn.metrics import classification_report
except ImportError as exc:
    sys.exit(f"[FEHLER] Fehlende Abhaengigkeit: {exc}\n  pip install transformers torch scikit-learn numpy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================
# EUER DATASET - 24 deutsche Restaurant-Reviews
# ============================================
# Regeln:
# 1. DEUTSCH geschrieben
# 2. Verschiedene Laengen (kurz + lang)
# 3. Verschiedene Intensitaeten (sehr positiv bis leicht positiv)
# 4. Schwierige Faelle eingebaut ("nicht schlecht", "geht so")
# 5. SELBST geschrieben

train_texts = [
    # === POSITIV (Label 1) ===
    "Das Essen war hervorragend, besonders die hausgemachte Pasta war ein Gedicht.",
    "Tolles Ambiente und super freundliches Personal, wir kommen definitiv wieder!",
    "Beste Pizza in der ganzen Stadt, knuspriger Teig und frische Zutaten.",
    "Der Service war schnell und aufmerksam, alles hat perfekt gepasst.",
    "Fantastisches Preis-Leistungs-Verhaeltnis, fuer die Qualitaet ein Schnaeppchen.",
    "Das Dessert war unglaublich lecker, die Schokoladentorte ist ein Muss!",
    "Sehr gemuetliche Atmosphaere, perfekt fuer ein romantisches Abendessen.",
    "Die Weinkarte ist beeindruckend, tolle Empfehlung vom Sommelier.",
    "Frischer Fisch, perfekt zubereitet, man schmeckt die Qualitaet sofort.",
    "Nicht schlecht, das Essen hat mich positiv ueberrascht.",
    "Der Koch hat sich persoenlich vorgestellt, das zeigt echte Leidenschaft.",
    "Wunderbare Vorspeisen und grosszuegige Portionen, sehr empfehlenswert.",
    # === NEGATIV (Label 0) ===
    "Wir haben ueber eine Stunde auf das Hauptgericht gewartet, inakzeptabel.",
    "Das Steak war komplett verkocht und trocken, fuer 35 Euro eine Frechheit.",
    "Unhoefliches Personal, der Kellner hat uns komplett ignoriert.",
    "Die Portion war winzig, dafuer aber voellig ueberteuert.",
    "Haare im Salat gefunden, einfach nur ekelhaft und unprofessionell.",
    "Das Restaurant war viel zu laut, man konnte sich kaum unterhalten.",
    "Reservierung wurde vergessen, mussten 40 Minuten an der Bar warten.",
    "Das Essen war lauwarm und geschmacklos, wie aus der Mikrowelle.",
    "Furchtbarer Geruch in der Naehe der Toiletten, appetitanregend war das nicht.",
    "Die Rechnung enthielt Posten die wir nie bestellt haben, sehr unserioes.",
    "Geht so, war insgesamt eher enttaeuschend fuer den Preis.",
    "Schlimmste Erfahrung seit langem, werde dieses Restaurant nie wieder besuchen.",
]

train_labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

# Defensive Pruefung
assert len(train_texts) == len(train_labels), (
    f"Mismatch! {len(train_texts)} Texte vs {len(train_labels)} Labels"
)

# Evaluation Set (NICHT im Training!)
eval_texts = [
    "Hervorragende Kueche, jeder Gang war ein Highlight.",
    "Katastrophaler Abend, alles was schiefgehen konnte ging schief.",
    "Solide Hausmannskost, nichts Ausgefallenes aber gut gemacht.",
    "Voellig ueberteuert und dazu noch unfreundlicher Service.",
    "Die Nachspeise allein war den Besuch wert, traumhaft!",
    "Kaltes Essen, dreckiges Besteck, nie wieder.",
]
eval_labels = [1, 0, 1, 0, 1, 0]

assert len(eval_texts) == len(eval_labels), (
    f"Mismatch! {len(eval_texts)} Texte vs {len(eval_labels)} Labels"
)

n_pos_train = sum(train_labels)
n_neg_train = len(train_labels) - n_pos_train
print(f"Training:   {len(train_texts)} Texte ({n_pos_train} pos, {n_neg_train} neg)")
print(f"Evaluation: {len(eval_texts)} Texte ({sum(eval_labels)} pos, {len(eval_labels) - sum(eval_labels)} neg)")


# ============================================
# DATASET-KLASSE (aus der Vorlesung, korrigiert)
# ============================================

class ReviewDataset(Dataset):
    """PyTorch Dataset fuer tokenisierte Text-Reviews."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) und labels ({len(labels)}) unterschiedlich lang!"
            )
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ============================================
# DEUTSCHES MODEL LADEN
# ============================================
# Warum deepset/gbert-base?
# - Speziell fuer Deutsch trainiert (auf 163GB deutschen Texten)
# - Bessere Performance auf deutschen Tasks als multilingual BERT
# - 110M Parameter, Standard-Groesse
# - Von deepset.ai, die auch Haystack (RAG Framework) bauen

model_name = "deepset/gbert-base"
print(f"\nModel: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIV", 1: "POSITIV"},
    label2id={"NEGATIV": 0, "POSITIV": 1},
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter: {total_params:,}")

# Datasets erstellen
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
eval_dataset = ReviewDataset(eval_texts, eval_labels, tokenizer)

print(f"Train Dataset: {len(train_dataset)} Beispiele")
print(f"Eval Dataset:  {len(eval_dataset)} Beispiele")


# ============================================
# TRAINING
# ============================================

training_args = TrainingArguments(
    output_dir="./deutsches_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("\nFine-Tuning startet...")
trainer.train()


# ============================================
# LIVE-TEST
# ============================================

model.eval()

test_saetze = [
    "Absolut fantastisches Essen, komme definitiv wieder!",
    "Katastrophaler Service, nie wieder dieses Restaurant.",
    "War ganz okay, nichts besonderes.",
    "Die beste Lasagne die ich je gegessen habe!",
    "Finger weg, totale Abzocke und schlechtes Essen.",
]

print("\n" + "=" * 60)
print("EUER DEUTSCHES MODEL - LIVE TEST")
print("=" * 60)

for text in test_saetze:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    predicted = torch.argmax(probs).item()
    confidence = probs[0][predicted].item()

    label = "POSITIV" if predicted == 1 else "NEGATIV"
    bar = "#" * int(confidence * 20)

    print(f"  {label:8s} [{bar:20s}] ({confidence:.1%}): {text}")

print("\n" + "=" * 60)
print("DEUTSCHES FINE-TUNING ABGESCHLOSSEN")
print("=" * 60)
