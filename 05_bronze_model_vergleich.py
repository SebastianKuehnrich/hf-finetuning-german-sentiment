"""
Aufgabe BRONZE: 4 Sentiment-Models auf deutschem Test-Dataset vergleichen
=========================================================================
20 deutsche Restaurant-Reviews (10 positiv, 10 negativ) werden durch
4 verschiedene Sentiment-Models geschickt. Jedes Model liefert ein
anderes Label-Format -> alle werden auf binaer 0/1 gemappt.
Ergebnis: Accuracy, F1, Lade-/Inferenzzeit pro Model + Vergleichstabelle.

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging
import time
from typing import Final

# ============================================
# DEFENSIVE IMPORTS
# ============================================

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    sys.exit(
        "[FEHLER] 'transformers' ist nicht installiert.\n"
        "  Loesung: pip install transformers torch"
    )

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    sys.exit(
        "[FEHLER] 'scikit-learn' ist nicht installiert.\n"
        "  Loesung: pip install scikit-learn"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# KONSTANTEN
# ============================================

POSITIVE_LABEL: Final[int] = 1
NEGATIVE_LABEL: Final[int] = 0
NEUTRAL_LABEL: Final[int] = -1   # Wird bei Evaluation herausgefiltert

# Die 4 Models mit ihren unterschiedlichen Label-Formaten
MODELS: Final[list[dict[str, str]]] = [
    {
        "name": "oliverguhr/german-sentiment-bert",
        "format": "positive / negative / neutral",
    },
    {
        "name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "format": "1-5 Sterne",
    },
    {
        "name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "format": "positive / negative / neutral",
    },
    {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "format": "POSITIVE / NEGATIVE",
    },
]


# ============================================
# DEUTSCHES TEST-DATASET (20 Texte)
# ============================================
# 10 positiv (label=1), 10 negativ (label=0)
# Verschiedene Laengen, Intensitaeten, Schwierigkeiten

TEST_TEXTS: Final[list[str]] = [
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
    "Der Koch hat sich persoenlich vorgestellt, das zeigt echte Leidenschaft.",
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
]

TEST_LABELS: Final[list[int]] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # 10x positiv
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # 10x negativ
]

# Defensive Pruefung: Texte und Labels muessen gleich lang sein
assert len(TEST_TEXTS) == len(TEST_LABELS) == 20, (
    f"Dataset-Fehler! {len(TEST_TEXTS)} Texte vs {len(TEST_LABELS)} Labels (erwartet: 20)"
)
assert sum(TEST_LABELS) == 10, (
    f"Balance-Fehler! {sum(TEST_LABELS)} positive Labels (erwartet: 10)"
)


# ============================================
# LABEL-MAPPING: Verschiedene Formate -> binaer 0/1
# ============================================

def map_label_to_binary(raw_label: str, model_name: str) -> int:
    """
    Mappt verschiedene Label-Formate auf binaer 0/1.

    Jedes Model liefert ein anderes Format:
    - german-sentiment-bert:    positive / negative / neutral
    - nlptown (Sterne):         1 star / 2 stars / 3 stars / 4 stars / 5 stars
    - cardiffnlp (XLM-R):      positive / negative / neutral
    - distilbert-sst2:          POSITIVE / NEGATIVE

    Neutral -> -1 (wird spaeter aus der Evaluation gefiltert)

    Returns:
        1 (positiv), 0 (negativ), -1 (neutral/unklar)
    """
    label = raw_label.strip().lower()

    # --- Model 1 & 3: positive / negative / neutral ---
    if label in ("positive", "pos"):
        return POSITIVE_LABEL
    if label in ("negative", "neg"):
        return NEGATIVE_LABEL
    if label in ("neutral", "neu"):
        return NEUTRAL_LABEL

    # --- Model 2: nlptown Sterne (1-5) ---
    if "star" in label:
        # Format: "1 star", "2 stars", "3 stars", "4 stars", "5 stars"
        try:
            stars = int(label[0])
        except (ValueError, IndexError):
            logger.warning("Konnte Sterne nicht parsen: '%s'", raw_label)
            return NEUTRAL_LABEL

        if stars >= 4:
            return POSITIVE_LABEL    # 4-5 Sterne -> positiv
        if stars <= 2:
            return NEGATIVE_LABEL    # 1-2 Sterne -> negativ
        return NEUTRAL_LABEL         # 3 Sterne  -> neutral (-1)

    # --- Fallback: Grossbuchstaben-Varianten (distilbert-sst2) ---
    # Bereits durch .lower() abgedeckt, aber sicherheitshalber:
    logger.warning("Unbekanntes Label '%s' von Model '%s' -> neutral", raw_label, model_name)
    return NEUTRAL_LABEL


# ============================================
# EIN MODEL LADEN + KLASSIFIZIEREN
# ============================================

def evaluate_single_model(
    model_name: str,
    texts: list[str],
    true_labels: list[int],
) -> dict | None:
    """
    Laedt ein Model, klassifiziert alle Texte, berechnet Metriken.

    Returns:
        Dict mit accuracy, f1, load_time, inference_time, details
        oder None bei Fehler.
    """
    short_name = model_name.split("/")[-1]

    # --- Model laden (mit Zeitmessung) ---
    logger.info("Lade Model: %s ...", model_name)
    load_start = time.time()

    try:
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,   # CPU erzwingen -> kein CUDA-Fehler moeglich
        )
    except Exception as exc:
        logger.error("Model '%s' konnte nicht geladen werden: %s", model_name, exc)
        return None

    load_time = time.time() - load_start

    # --- Alle Texte klassifizieren (mit Zeitmessung) ---
    y_true_filtered: list[int] = []
    y_pred_filtered: list[int] = []
    details: list[dict] = []
    n_neutral = 0

    inference_start = time.time()

    for text, true_label in zip(texts, true_labels):
        try:
            result = classifier(text)
            if not result:
                logger.warning("Leeres Ergebnis fuer: '%s'", text[:30])
                continue

            raw_label = result[0]["label"]
            confidence = result[0]["score"]
            predicted = map_label_to_binary(raw_label, model_name)

            if predicted == NEUTRAL_LABEL:
                # Neutral -> aus der Evaluation filtern
                n_neutral += 1
                is_correct = None
                symbol = "[~]"
            else:
                is_correct = predicted == true_label
                symbol = "[+]" if is_correct else "[X]"
                y_true_filtered.append(true_label)
                y_pred_filtered.append(predicted)

            details.append({
                "text": text,
                "true": true_label,
                "pred": predicted,
                "raw_label": raw_label,
                "confidence": confidence,
                "symbol": symbol,
            })

        except Exception as exc:
            logger.warning("Fehler bei '%s...': %s", text[:20], exc)
            details.append({
                "text": text,
                "true": true_label,
                "pred": None,
                "raw_label": "ERROR",
                "confidence": 0.0,
                "symbol": "[!]",
            })

    inference_time = time.time() - inference_start

    # --- Metriken berechnen (nur nicht-neutrale Predictions) ---
    if len(y_true_filtered) == 0:
        logger.warning("Model '%s': Keine auswertbaren Predictions!", model_name)
        accuracy = 0.0
        f1 = 0.0
    else:
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        f1 = f1_score(y_true_filtered, y_pred_filtered, average="binary")

    return {
        "model_name": model_name,
        "short_name": short_name,
        "accuracy": accuracy,
        "f1": f1,
        "load_time": load_time,
        "inference_time": inference_time,
        "n_evaluated": len(y_true_filtered),
        "n_neutral": n_neutral,
        "details": details,
    }


# ============================================
# ERGEBNISSE EINES MODELS ANZEIGEN
# ============================================

def print_model_results(result: dict) -> None:
    """Zeigt die Detail-Ergebnisse eines Models an."""
    print(f"\n{'=' * 70}")
    print(f"  MODEL: {result['model_name']}")
    print(f"{'=' * 70}")
    print(f"  Ladezeit:     {result['load_time']:.1f}s")
    print(f"  Inferenzzeit: {result['inference_time']:.1f}s")
    print(f"  Ausgewertet:  {result['n_evaluated']}/{len(TEST_TEXTS)} "
          f"({result['n_neutral']} neutral gefiltert)")
    print(f"  Accuracy:     {result['accuracy']:.1%}")
    print(f"  F1-Score:     {result['f1']:.4f}")
    print(f"{'-' * 70}")

    for d in result["details"]:
        true_str = "POS" if d["true"] == 1 else "NEG"
        if d["pred"] == NEUTRAL_LABEL:
            pred_str = "NEU"
        elif d["pred"] == POSITIVE_LABEL:
            pred_str = "POS"
        elif d["pred"] == NEGATIVE_LABEL:
            pred_str = "NEG"
        else:
            pred_str = "ERR"

        print(
            f"  {d['symbol']} true={true_str} pred={pred_str} "
            f"({d['raw_label']:>12s} {d['confidence']:.3f}) "
            f"{d['text'][:55]}..."
        )


# ============================================
# VERGLEICHSTABELLE (sortiert nach F1)
# ============================================

def print_comparison_table(results: list[dict]) -> None:
    """Gibt die finale Vergleichstabelle sortiert nach F1-Score aus."""
    # Nach F1 absteigend sortieren
    sorted_results = sorted(results, key=lambda r: r["f1"], reverse=True)

    print(f"\n{'#' * 70}")
    print(f"  VERGLEICHSTABELLE: 4 SENTIMENT-MODELS (sortiert nach F1)")
    print(f"{'#' * 70}")
    print(
        f"  {'#':>2s}  {'Model':<45s} {'Acc':>6s} {'F1':>6s} "
        f"{'Load':>6s} {'Infer':>6s} {'Eval':>4s}"
    )
    print(f"  {'-' * 66}")

    for rank, r in enumerate(sorted_results, start=1):
        short = r["short_name"][:44]
        print(
            f"  {rank:>2d}  {short:<45s} {r['accuracy']:>5.1%} "
            f"{r['f1']:>6.4f} {r['load_time']:>5.1f}s "
            f"{r['inference_time']:>5.1f}s {r['n_evaluated']:>3d}/{len(TEST_TEXTS)}"
        )

    print(f"  {'-' * 66}")

    # Bestes Model hervorheben
    best = sorted_results[0]
    print(f"\n  BESTES MODEL (F1): {best['model_name']}")
    print(f"  -> F1: {best['f1']:.4f} | Accuracy: {best['accuracy']:.1%}")


# ============================================
# VERSTAENDNIS-FRAGEN
# ============================================

# FRAGE 1: Warum liefern die 4 Models so unterschiedliche Label-Formate?
#
#    Jedes Model wurde auf einem ANDEREN Dataset mit ANDEREM Label-Schema
#    trainiert:
#    - german-sentiment-bert: Auf deutschen Texten mit 3 Klassen
#      (positive/negative/neutral) trainiert.
#    - nlptown: Auf Produkt-Reviews mit 1-5 Sternen trainiert ->
#      Multi-Class Regression, 5 Label-Klassen.
#    - cardiffnlp: Auf Twitter-Daten in 100+ Sprachen mit 3 Sentiment-
#      Klassen trainiert (positiv/negativ/neutral).
#    - distilbert-sst2: Auf englischem SST-2 Benchmark mit binaerer
#      Klassifikation (POSITIVE/NEGATIVE) trainiert.
#
#    Das Label-Format spiegelt die Trainingsaufgabe wider. Deshalb
#    brauchen wir das Mapping auf binaer 0/1 um sie vergleichbar zu machen.

# FRAGE 2: Warum filtern wir "neutral" aus der Evaluation?
#
#    Unser Gold-Standard ist BINAER (0=negativ, 1=positiv). Es gibt
#    kein "neutral" in unseren True Labels. Wenn ein Model "neutral"
#    vorhersagt, koennen wir nicht sagen ob das richtig oder falsch ist,
#    weil der Text in Wirklichkeit positiv oder negativ gelabelt ist.
#    Neutral als falsch zu zaehlen waere unfair gegenueber Models die
#    3 Klassen haben. Deshalb filtern wir neutral heraus und berechnen
#    Accuracy/F1 nur auf den Texten wo das Model sich entschieden hat.
#    Die Anzahl der neutralen Predictions zeigen wir trotzdem -> ein
#    Model das alles als neutral einstuft hat zwar keine Fehler, aber
#    auch keine Aussagekraft (n_evaluated sehr niedrig).

# FRAGE 3: Welches Model funktioniert am besten auf deutschen Texten und warum?
#
#    Erwartet: oliverguhr/german-sentiment-bert ist am besten weil es
#    SPEZIELL fuer deutsche Sentiment-Analyse trainiert wurde. Es kennt
#    deutsche Grammatik, Woerter und Ausdruecke. Das cardiffnlp-Model
#    (XLM-RoBERTa) ist ebenfalls multilingual und sollte Deutsch
#    einigermassen gut koennen, aber es wurde auf Twitter-Daten trainiert
#    (kurze, informelle Texte) -> Restaurant-Reviews sind ein anderer
#    Stil. Das nlptown-Model ist multilingual aber auf Produktbewertungen
#    trainiert. distilbert-sst2 ist rein englisch und wird auf deutschen
#    Texten am schlechtesten abschneiden, weil es deutsche Woerter nicht
#    kennt und der Tokenizer sie in unsinnige Sub-Tokens zerlegt.


# ============================================
# MAIN
# ============================================

def main() -> None:
    """Hauptfunktion: Alle 4 Models laden, testen, vergleichen."""
    logger.info("Bronze Model-Vergleich gestartet")
    total_start = time.time()

    n_pos = sum(TEST_LABELS)
    n_neg = len(TEST_LABELS) - n_pos
    print("=" * 70)
    print("  BRONZE CHALLENGE: 4 SENTIMENT-MODELS VERGLEICHEN")
    print("=" * 70)
    print(f"  Test-Dataset: {len(TEST_TEXTS)} deutsche Restaurant-Reviews")
    print(f"  Positiv: {n_pos} | Negativ: {n_neg}")
    print(f"  Models:  {len(MODELS)}")
    for i, m in enumerate(MODELS, start=1):
        print(f"    {i}. {m['name']}")
        print(f"       Format: {m['format']}")

    # --- Alle Models auswerten ---
    all_results: list[dict] = []

    for model_info in MODELS:
        result = evaluate_single_model(
            model_name=model_info["name"],
            texts=TEST_TEXTS,
            true_labels=TEST_LABELS,
        )

        if result is not None:
            print_model_results(result)
            all_results.append(result)
        else:
            print(f"\n  [UEBERSPRUNGEN] {model_info['name']} konnte nicht geladen werden")

    # --- Vergleichstabelle ---
    if all_results:
        print_comparison_table(all_results)
    else:
        logger.error("Kein einziges Model konnte geladen werden!")

    # --- Zusammenfassung ---
    elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  BRONZE CHALLENGE ABGESCHLOSSEN")
    print(f"  {len(all_results)}/{len(MODELS)} Models erfolgreich getestet")
    print(f"  Gesamtzeit: {elapsed:.1f} Sekunden")
    print(f"{'=' * 70}")
    logger.info("Bronze Model-Vergleich abgeschlossen in %.1f Sekunden", elapsed)


if __name__ == "__main__":
    main()
