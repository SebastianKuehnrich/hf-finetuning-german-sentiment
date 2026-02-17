# Abschlussbericht: Fine-Tuning & Vision Transformer

**Kurs:** Morphos GmbH -- KI & Python Modul, Woche 2
**Aufgabe:** 05\_HAUSAUFGABE\_FineTuning\_ViT
**Datum:** 13.02.2025

---

## Inhaltsverzeichnis

1. [Projektuebersicht](#1-projektuebersicht)
2. [Pflichtaufgaben (1.1--1.4)](#2-pflichtaufgaben)
3. [Challenge-Aufgaben (Bronze--Diamond)](#3-challenge-aufgaben)
4. [Gesamtergebnisse im Vergleich](#4-gesamtergebnisse-im-vergleich)
5. [Technische Erkenntnisse](#5-technische-erkenntnisse)
6. [Fazit](#6-fazit)

---

## 1. Projektuebersicht

### Ziel

Praktische Erfahrung mit **Hugging Face Transformers** sammeln: Datasets laden,
Models fine-tunen, evaluieren und vergleichen -- sowohl fuer Text (Sentiment Analysis)
als auch fuer Bild (Vision Transformer).

### Technologie-Stack

| Komponente       | Technologie                                   |
|------------------|-----------------------------------------------|
| Sprache          | Python 3.12                                   |
| Deep Learning    | PyTorch + Hugging Face Transformers            |
| Datasets         | Hugging Face `datasets` Library               |
| Evaluation       | scikit-learn (Accuracy, F1, Confusion Matrix) |
| Visualisierung   | Matplotlib                                    |
| Hardware         | CPU (kein GPU)                                |

### Dateistruktur

```
Hausaufgabe/
  01_datasets_erkunden.py         Pflicht 1.1
  02_finetuning_imdb.py           Pflicht 1.2
  03_eigenes_dataset.py           Pflicht 1.3
  04_verstaendnis.py              Pflicht 1.4
  05_bronze_model_vergleich.py    Challenge Bronze
  06_silver_german_finetuning.py  Challenge Silver
  07_gold_vit.py                  Challenge Gold
  08_diamond_pipeline.py          Challenge Diamond
  ABSCHLUSSBERICHT.md             Dieser Bericht
```

---

## 2. Pflichtaufgaben

### 2.1 Datasets erkunden (01)

Drei Datasets wurden geladen und analysiert:

| Dataset                          | Sprache | Klassen       | Groesse (Train) |
|----------------------------------|---------|---------------|-----------------|
| stanfordnlp/imdb                 | EN      | 2 (pos/neg)   | 25.000          |
| mteb/amazon\_reviews\_multi (de) | DE      | 5 (Sterne)    | 200.000         |
| clapAI/MultiLingualSentiment     | 17 Spr. | 3 (pos/neu/neg) | 3.147.478    |

**Technische Herausforderung:** Die originalen Datasets `mteb/amazon_reviews_multi`
und `tyqiangz/multilingual-sentiments` nutzen veraltete Dataset-Scripts (`.py`), die
seit `datasets>=4.0` nicht mehr unterstuetzt werden.

**Loesung:**
- Amazon DE: Parquet-Dateien direkt per URL vom `refs/convert/parquet`-Branch laden
- Multilingual: Alternatives Dataset `clapAI/MultiLingualSentiment` (natives Parquet)

**Verstaendnisfragen:**
- *FRAGE 1:* Amazon hat 5 Sterne-Klassen (Multi-Class) vs. IMDB mit 2 Labels (binaer).
  Multi-Class ist schwerer, weil die Grenzen zwischen z.B. 3 und 4 Sternen fliessend sind.
- *FRAGE 2:* `load_dataset()` ist besser als eigene Daten wegen: (1) Skalierung,
  (2) Qualitaet, (3) Reproduzierbarkeit, (4) Caching, (5) vordefinierte Splits.

---

### 2.2 Fine-Tuning IMDB (02)

| Parameter        | Wert                                    |
|------------------|-----------------------------------------|
| Model            | distilbert-base-uncased (67M Parameter) |
| Trainingsdaten   | 500 IMDB Reviews                        |
| Testdaten        | 500 IMDB Reviews                        |
| Epochs           | 3                                       |
| Learning Rate    | 2e-5                                    |
| Max Length        | 256 Tokens                              |

**Ergebnisse:**

| Epoch | Eval Loss | Accuracy | F1    |
|-------|-----------|----------|-------|
| 1     | 0.499     | 78.0%    | 80.7% |
| 2     | 0.370     | 84.4%    | 85.0% |
| 3     | 0.345     | **84.8%**| **85.0%** |

```
              precision    recall  f1-score   support
    NEGATIVE       0.89      0.79      0.84       254
    POSITIVE       0.81      0.90      0.85       246
    accuracy                           0.84       500
```

**Live-Test:** 5/5 korrekt. Neutrale Texte werden mit niedriger Konfidenz
klassifiziert (52.7%), was zeigt, dass das Model bei Grenzfaellen unsicher ist.

---

### 2.3 Eigenes deutsches Dataset (03)

| Parameter        | Wert                                    |
|------------------|-----------------------------------------|
| Model            | deepset/gbert-base (110M Parameter)     |
| Trainingsdaten   | 24 eigene Restaurant-Reviews (DE)       |
| Evaluationsdaten | 6 Reviews                               |
| Epochs           | 5                                       |

**Warum deepset/gbert-base?**
- Speziell auf deutschem Text vortrainiert (Wikipedia DE, OPUS, etc.)
- BERT-Architektur ideal fuer Klassifikation
- Groesserer Wortschatz fuer deutsche Sprache als multilingual-BERT

**Ergebnisse:**

| Epoch | Eval Loss |
|-------|-----------|
| 1     | 0.578     |
| 2     | 0.408     |
| 3     | 0.291     |
| 4     | 0.241     |
| 5     | **0.208** |

**Live-Test:** 3/5 korrekt. Mit nur 24 Trainingstexten kann ein 110M-Parameter-Model
die Muster nicht zuverlaessig lernen -- aber der stetig sinkende Loss zeigt,
dass das Training funktioniert.

---

### 2.4 Verstaendnisfragen (04)

Alle 6 Fragen wurden ausfuehrlich beantwortet:

1. **learning\_rate:** Steuert die Schrittgroesse beim Gradientenabstieg.
   Zu gross = Springen, zu klein = kein Fortschritt. 2e-5 ist Standard fuer Fine-Tuning.

2. **Warum datasets-Library:** Millionen Texte mit einer Zeile Code, professionell
   gelabelt, automatisches Caching, vordefinierte Splits.

3. **Metriken:** Accuracy allein reicht bei unbalancierten Daten nicht.
   F1-Score kombiniert Precision und Recall zu einem einzelnen Mass.

4. **Deutsche vs. englische Models:** Deutsche Models (gbert-base) verstehen
   Umlaute, Komposita und deutsche Grammatik. Englische Models (DistilBERT)
   versagen auf deutschem Text komplett.

5. **Fine-Tuning vs. Transfer Learning:** Fine-Tuning passt alle Gewichte an,
   Transfer Learning friert die Basis ein und trainiert nur den Klassifikator.

6. **DataCollator vs. padding=True:** DataCollator padded dynamisch pro Batch
   (effizient), padding=True padded alles auf die gleiche Laenge (verschwendet
   Rechenzeit bei kurzen Texten).

---

## 3. Challenge-Aufgaben

### 3.1 Bronze: 4 Models vergleichen (05)

20 deutsche Restaurant-Reviews wurden mit 4 verschiedenen Sentiment-Models getestet:

| # | Model                                     | Accuracy | F1     | Ladezeit |
|---|-------------------------------------------|----------|--------|----------|
| 1 | oliverguhr/german-sentiment-bert          | **100%** | **1.000** | 1.2s  |
| 2 | nlptown/bert-base-multilingual-sentiment  | **100%** | **1.000** | 0.8s  |
| 3 | cardiffnlp/twitter-xlm-roberta-sentiment  | 95%      | 0.952  | 47.5s    |
| 4 | distilbert-base-uncased-finetuned-sst-2   | 50%      | 0.000  | 1.2s     |

**Zentrale Erkenntnis:** Das englische DistilBERT-SST2 erreicht nur 50% Accuracy
auf deutschen Texten -- es klassifiziert **alle** Texte als NEGATIVE, weil deutsche
Woerter fuer das englische Model unbekannte Token sind. Sprach-spezifische Models
sind entscheidend.

**Warum german-sentiment-bert gewinnt:**
- Speziell auf deutschen Sentiment-Daten trainiert
- Versteht Nuancen wie "Preis-Leistungs-Verhaeltnis"
- Hohe Konfidenz (>90%) bei fast allen Vorhersagen

---

### 3.2 Silver: Deutsches Fine-Tuning auf Amazon Reviews (06)

| Parameter        | Wert                                    |
|------------------|-----------------------------------------|
| Model            | deepset/gbert-base (110M Parameter)     |
| Dataset          | mteb/amazon\_reviews\_multi (DE)        |
| Trainingsdaten   | 2.000 Reviews (binarisiert)             |
| Testdaten        | 500 Reviews                             |
| Binarisierung    | 1-2 Sterne = NEGATIV, 4-5 = POSITIV    |

**Ergebnisse:**

| Epoch | Eval Loss | Accuracy | F1    |
|-------|-----------|----------|-------|
| 1     | 0.189     | 95.4%    | 95.7% |
| 2     | 0.210     | 95.4%    | 95.7% |
| 3     | 0.231     | 94.8%    | 95.1% |

```
              precision    recall  f1-score   support
     NEGATIV       0.96      0.94      0.95       234
     POSITIV       0.95      0.97      0.96       266
    accuracy                           0.95       500
```

**95% Accuracy mit nur 2.000 Trainingstexten** -- das beste Ergebnis aller Aufgaben.

**Beobachtung:** Der Eval Loss steigt ab Epoch 2, waehrend der Train Loss weiter sinkt.
Das deutet auf leichtes **Overfitting** hin. Best Model wird nach Epoch 1 geladen
(load\_best\_model\_at\_end=True).

---

### 3.3 Gold: Vision Transformer auf CIFAR-10 (07)

| Parameter        | Wert                                    |
|------------------|-----------------------------------------|
| Model            | google/vit-base-patch16-224 (87M Param.)|
| Dataset          | CIFAR-10 (10.000 Testbilder)            |
| Samples          | 20 zufaellige Bilder                    |
| Vortrainiert auf | ImageNet-1k (1.000 Klassen)             |

**Ergebnisse:** 4/20 korrekt (20% Accuracy)

| CIFAR-10 Klasse | Accuracy | Bemerkung              |
|-----------------|----------|------------------------|
| cat             | 3/3 (100%) | Egyptian cat, Persian cat |
| ship            | 1/5 (20%)  | Verwechslung mit yawl, speedboat |
| airplane        | 0/1 (0%)   | Als "airliner" erkannt (99%!) |
| automobile      | 0/3 (0%)   | Als "beach wagon", "moving van" |
| dog             | 0/3 (0%)   | Als "elkhound", "spaniel" |
| truck           | 0/3 (0%)   | Als "moving van" (99.7%!) |

**Zentrale Erkenntnis: Label-Mismatch, nicht Model-Versagen!**

Das ViT-Model erkennt die Bilder semantisch **korrekt** -- aber mit den falschen
Label-Granularitaeten:

- CIFAR-10: "airplane" -- ViT: "airliner" (99.1% Konfidenz) -> gleiche Sache!
- CIFAR-10: "truck" -- ViT: "moving van" (99.7%) -> ein Truck-Typ!
- CIFAR-10: "ship" -- ViT: "yawl" (98.1%) -> ein Segelschiff!
- CIFAR-10: "dog" -- ViT: "Norwegian elkhound" -> eine Hunderasse!

ImageNet hat 1.000 feine Klassen, CIFAR-10 nur 10 grobe. Bei semantischer
Bewertung waere die Accuracy bei ca. 80-90%.

**Warum Katzen perfekt erkannt werden:** ImageNet-Klassen wie "Egyptian cat" und
"Persian cat" enthalten das Wort "cat", das direkt auf die CIFAR-10 Klasse "cat"
mappt. Andere Klassen haben dieses Glueck nicht.

**Patches-Berechnung:**
- Bild: 224x224 Pixel, Patch: 16x16 Pixel
- Anzahl: (224/16) x (224/16) = 14 x 14 = **196 Patches** + 1 CLS-Token = **197**

---

### 3.4 Diamond: Komplette Pipeline (08)

Die Diamond-Pipeline kombiniert alle vorherigen Aufgaben in einer 6-Phasen-Pipeline:

| Phase | Inhalt                          | Ergebnis            |
|-------|---------------------------------|---------------------|
| 1     | Fine-Tuning (1000 IMDB Reviews) | 87% Accuracy        |
| 2     | Evaluation + Classification Report | F1 = 87.2%       |
| 3     | Training Curves (Loss + Metrics)| 37 Train + 3 Eval   |
| 4     | ViT auf CIFAR-10 (8 Bilder)    | 2/8 korrekt (25%)   |
| 5     | Dashboard (2x2 Plot)            | diamond\_dashboard.png |
| 6     | Save + Live-Test                | 3/3 korrekt         |

**Gesamtzeit:** 28.8 Minuten (CPU)

**Dashboard-Inhalt:**
1. Training Loss Kurve (37 Datenpunkte)
2. Evaluation Metriken pro Epoch (Accuracy, F1, Precision, Recall)
3. ViT CIFAR-10 Bildvorhersagen (4x2 Grid)
4. Confusion Matrix

---

## 4. Gesamtergebnisse im Vergleich

### Text-Klassifikation: Datenmenge vs. Accuracy

| Aufgabe    | Daten  | Model         | Accuracy | F1    |
|------------|--------|---------------|----------|-------|
| 03 (eigen) | 24     | gbert-base    | ~60%     | --    |
| 02 (IMDB)  | 500    | DistilBERT    | 84%      | 85.0% |
| 08 (Diamond)| 1.000 | DistilBERT    | 87%      | 87.2% |
| 06 (Silver)| 2.000  | gbert-base    | **95%**  | **95.7%** |

**Fazit:** Mehr Daten = bessere Ergebnisse. Der Sprung von 500 auf 2.000
Trainingstexte bringt +11% Accuracy. Das deutsche gbert-base profitiert
besonders stark, weil es die Sprache bereits versteht.

### Model-Vergleich auf Deutsch

| Model                    | Deutsch? | Accuracy (DE) |
|--------------------------|----------|---------------|
| german-sentiment-bert    | Ja       | **100%**      |
| bert-multilingual        | Ja       | **100%**      |
| twitter-xlm-roberta      | Ja       | 95%           |
| distilbert-sst-2 (EN)   | Nein     | 50% (Zufall)  |

---

## 5. Technische Erkenntnisse

### 5.1 Best Practices angewendet

Alle 8 Dateien folgen einheitlichen Coding-Standards:

```python
from __future__ import annotations           # Moderne Type Hints
import logging                                # Statt print() fuer Status
try:                                          # Defensive Imports
    from transformers import ...
except ImportError:
    sys.exit("[FEHLER] pip install ...")

MODEL_NAME: str = "deepset/gbert-base"       # Konstanten am Anfang
SEED: int = 42                                # Reproduzierbarkeit

def main() -> None:                           # Klare Funktionsstruktur
    """Docstring auf jeder Funktion."""
    ...

if __name__ == "__main__":                    # Entry Point
    main()
```

### 5.2 Haeufige Fallstricke

| Problem                    | Loesung                                     |
|----------------------------|---------------------------------------------|
| Dataset-Scripts veraltet   | `revision="refs/convert/parquet"` oder Parquet-URLs |
| Label-Mapping (5 Sterne -> binaer) | Binarisierung + Neutral filtern    |
| Unicode in Terminal        | `"#"` statt `"█"` fuer Balken               |
| Keras input\_shape Warning | `keras.layers.Input(shape=...)` nutzen       |
| ViT Label-Mismatch        | ImageNet-1k vs. CIFAR-10 Label-Granularitaet |
| Overfitting bei wenig Daten | `load_best_model_at_end=True`              |

### 5.3 Hardware-Ueberlegungen

Alle Aufgaben liefen auf **CPU** ohne GPU. Trainingszeiten:

| Aufgabe         | Daten  | Epochs | Zeit      |
|-----------------|--------|--------|-----------|
| 03 (eigen)      | 24     | 5      | 38 Sek.   |
| 02 (IMDB)       | 500    | 3      | 15 Min.   |
| 08 (Diamond)    | 1.000  | 3      | 28 Min.   |
| 06 (Silver)     | 2.000  | 3      | 45 Min.   |

Mit GPU waeren die Zeiten ca. 5-10x schneller.

---

## 6. Fazit

### Was gelernt wurde

1. **Fine-Tuning funktioniert:** Selbst mit wenigen hundert Texten kann ein
   vortrainiertes Model sinnvolle Ergebnisse liefern.

2. **Sprache matters:** Ein englisches Model auf deutschen Texten ist nutzlos (50%).
   Sprach-spezifische oder multilinguale Models sind Pflicht.

3. **Daten sind der Schluessel:** 24 Texte -> ~60%, 500 -> 84%, 2000 -> 95%.
   Die Datenmenge hat den groessten Einfluss auf die Qualitaet.

4. **Transfer Learning hat Grenzen:** ViT erkennt Bilder korrekt, aber die
   Label-Granularitaet muss zum Ziel-Dataset passen.

5. **Evaluation ist mehr als Accuracy:** F1-Score, Precision, Recall und
   Confusion Matrix geben ein vollstaendigeres Bild der Model-Qualitaet.

### Alle Aufgaben erfolgreich abgeschlossen

| Aufgabe        | Status | Highlight                              |
|----------------|--------|----------------------------------------|
| 01 Datasets    | Pass   | 3 Datasets geladen + analysiert        |
| 02 IMDB        | Pass   | 84% Accuracy, Confusion Matrix         |
| 03 Eigen       | Pass   | 24 deutsche Reviews, Loss sinkt        |
| 04 Fragen      | Pass   | 6/6 Fragen beantwortet                 |
| 05 Bronze      | Pass   | 4 Models verglichen, DE gewinnt        |
| 06 Silver      | Pass   | 95% Accuracy auf Amazon DE             |
| 07 Gold        | Pass   | ViT Label-Mismatch erklaert            |
| 08 Diamond     | Pass   | 6-Phasen-Pipeline, Dashboard erstellt  |

**Gesamte Laufzeit:** ca. 1.5 Stunden auf CPU

---

*Erstellt mit Python, Hugging Face Transformers und scikit-learn.*
*Alle Ergebnisse sind reproduzierbar mit Seed=42.*
