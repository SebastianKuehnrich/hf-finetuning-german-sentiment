"""
Aufgabe 1.1: Echte Datasets laden und erkunden
================================================
Die 'datasets' Library ist wie pip fuer Daten.
Statt Daten selbst zu schreiben, ladet ihr echte
Produktionsdaten mit einer Zeile Code.

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

from __future__ import annotations

import sys
import logging

# --- Defensive Imports ---
try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "[FEHLER] 'datasets' ist nicht installiert.\n"
        "  Loesung: pip install datasets"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# DATASET 1: IMDB Film-Reviews (Englisch)
# ============================================
# 25.000 Training + 25.000 Test Reviews
# Labels: 0 = negativ, 1 = positiv

def explore_imdb() -> None:
    """Laedt und zeigt das IMDB Dataset."""
    print("=" * 60)
    print("DATASET 1: IMDB Movie Reviews")
    print("=" * 60)

    try:
        imdb = load_dataset("stanfordnlp/imdb")
    except Exception as exc:
        logger.error("IMDB konnte nicht geladen werden: %s", exc)
        return

    print(f"  Splits:    {list(imdb.keys())}")
    print(f"  Training:  {len(imdb['train'])} Reviews")
    print(f"  Test:      {len(imdb['test'])} Reviews")
    print(f"  Spalten:   {imdb['train'].column_names}")
    print(f"  Beispiel:  {imdb['train'][0]['text'][:100]}...")
    print(f"  Label:     {imdb['train'][0]['label']}")

    # Defensive: Label-Verteilung pruefen
    train_labels = imdb["train"]["label"]
    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)
    print(f"\n  Label-Verteilung (Train):")
    print(f"    Positiv: {n_pos}")
    print(f"    Negativ: {n_neg}")
    print(f"    Balance: {'Balanciert' if abs(n_pos - n_neg) < 100 else 'Unbalanciert!'}")


# ============================================
# DATASET 2: Amazon Reviews (Deutsch!)
# ============================================
# 200.000 Training + 5.000 Test Reviews pro Sprache
# Labels: 0-4 (Sterne 1-5)

def explore_amazon_de() -> None:
    """Laedt und zeigt das deutsche Amazon Reviews Dataset.

    Loesung: Das originale 'mteb/amazon_reviews_multi' nutzt ein
    veraltetes Dataset-Script (.py). Der Parquet-Branch hat die
    Sprachen in Unterordnern (de/, en/, ...) aber ohne 'language'-
    Spalte. Wir laden die deutschen Parquet-Dateien direkt per URL.
    """
    print("\n" + "=" * 60)
    print("DATASET 2: Amazon Reviews (DEUTSCH)")
    print("=" * 60)

    # --- Deutsche Parquet-Dateien direkt laden ---
    # Auf dem Parquet-Branch liegen die Dateien unter:
    #   de/train/0000.parquet, de/test/0000.parquet, de/validation/0000.parquet
    BASE = (
        "https://huggingface.co/datasets/mteb/amazon_reviews_multi"
        "/resolve/refs%2Fconvert%2Fparquet/de"
    )
    try:
        amazon_de = load_dataset(
            "parquet",
            data_files={
                "train":      f"{BASE}/train/0000.parquet",
                "test":       f"{BASE}/test/0000.parquet",
                "validation": f"{BASE}/validation/0000.parquet",
            },
        )
        source = "mteb/amazon_reviews_multi (de, parquet-branch)"
    except Exception as exc:
        logger.error("Amazon DE konnte nicht geladen werden: %s", exc)
        print("  [!] Dataset nicht verfuegbar.")
        print(f"      Fehler: {exc}")
        return

    print(f"  Quelle:    {source}")
    print(f"  Splits:    {list(amazon_de.keys())}")
    print(f"  Training:  {len(amazon_de['train'])} Reviews")
    print(f"  Test:      {len(amazon_de['test'])} Reviews")
    print(f"  Spalten:   {amazon_de['train'].column_names}")
    print(f"  Beispiel:  {amazon_de['train'][0]['text'][:100]}...")
    print(f"  Label:     {amazon_de['train'][0]['label']}")

    # Label-Verteilung (5 Sterne)
    train_labels = amazon_de["train"]["label"]
    unique_labels = sorted(set(train_labels))
    n_stars = len(unique_labels)
    print(f"\n  Label-Verteilung (Train DE) - {n_stars} Klassen:")
    for star in unique_labels:
        count = sum(1 for l in train_labels if l == star)
        divisor = max(100, len(train_labels) // 60)
        bar = "#" * (count // divisor)
        print(f"    {star + 1} Stern(e): {count:>6d} {bar}")


# ============================================
# DATASET 3: Multilingual Sentiments (12 Sprachen!)
# ============================================

def explore_multilingual() -> None:
    """Laedt und zeigt das multilingual Sentiments Dataset.

    Loesung: Das originale 'tyqiangz/multilingual-sentiments' nutzt
    ein veraltetes Dataset-Script. Wir verwenden stattdessen
    'clapAI/MultiLingualSentiment' (natives Parquet, kein Script).
    Dieses hat 17 Sprachen inkl. Deutsch, 3 Klassen, 212k DE-Texte.
    """
    print("\n" + "=" * 60)
    print("DATASET 3: Multilingual Sentiments")
    print("=" * 60)

    # --- clapAI/MultiLingualSentiment: Natives Parquet ---
    try:
        multi_full = load_dataset("clapAI/MultiLingualSentiment")
        source = "clapAI/MultiLingualSentiment"
    except Exception as exc:
        logger.error(
            "Multilingual Sentiments konnte nicht geladen werden: %s",
            exc,
        )
        print("  [!] Dataset nicht verfuegbar.")
        print(f"      Fehler: {exc}")
        return

    print(f"  Quelle:    {source}")
    print(f"  Splits:    {list(multi_full.keys())}")

    first_split = list(multi_full.keys())[0]
    print(f"  Gesamt ({first_split}): {len(multi_full[first_split])} Eintraege")
    print(f"  Spalten:   {multi_full[first_split].column_names}")

    # --- Nur deutsche Texte filtern ---
    german = multi_full[first_split].filter(
        lambda x: x["language"] == "de"
    )
    n_german = len(german)
    print(f"\n  Deutsche Texte:  {n_german}")

    if n_german == 0:
        logger.warning("Keine deutschen Texte gefunden!")
        return

    print(f"  Beispiel:  {german[0]['text'][:100]}...")
    print(f"  Label:     {german[0]['label']}")

    # Label-Verteilung (String-Labels: Positive/Negative/Neutral)
    train_labels = german["label"]
    unique_labels = sorted(set(train_labels))
    print(f"\n  Label-Verteilung (Deutsch) - {len(unique_labels)} Klassen:")
    for label_name in unique_labels:
        count = sum(1 for l in train_labels if l == label_name)
        divisor = max(1, n_german // 60)
        bar = "#" * (count // divisor)
        print(f"    {label_name:>10s}: {count:>6d} {bar}")


# ============================================
# VERSTAENDNIS-FRAGEN
# ============================================

# FRAGE 1: Wie viele Sterne-Bewertungen hat das Amazon-Dataset?
#           Was ist der Unterschied zu IMDB (nur 2 Labels)?
#
# Antwort:  Amazon hat 5 Sterne (0-4 intern, also 1-5 Sterne).
#           IMDB hat nur 2 Labels (positiv/negativ) -> binaere Klassifikation.
#           Amazon erlaubt feinere Abstufungen (Multi-Class), aber ist schwerer
#           zu klassifizieren weil der Unterschied zwischen 3 und 4 Sternen
#           oft unklar ist. IMDB ist einfacher weil die Grenzen klarer sind.

# FRAGE 2: Warum ist load_dataset() besser als Daten selbst zu schreiben?
#           Nennt mindestens 3 Gruende.
#
# Antwort:  1. SKALIERUNG: load_dataset laed Tausende/Millionen Texte,
#              manuell wuerde das Tage dauern.
#           2. QUALITAET: Professionell gelabelte Daten, peer-reviewed,
#              weniger Fehler als selbstgeschriebene Listen.
#           3. REPRODUZIERBARKEIT: Jeder kann das gleiche Dataset laden
#              und Ergebnisse vergleichen. Standard-Benchmarks.
#           4. CACHING: Automatisches Caching, Memory-Mapping fuer
#              grosse Datasets die nicht in RAM passen.
#           5. SPLITS: Train/Test Splits sind vordefiniert -> kein
#              versehentliches Data Leakage.


# ============================================
# MAIN
# ============================================

def main() -> None:
    """Hauptfunktion: Alle Datasets erkunden."""
    logger.info("Datasets-Exploration gestartet")

    explore_imdb()
    explore_amazon_de()
    explore_multilingual()

    print("\n" + "=" * 60)
    print("ALLE DATASETS GELADEN")
    print("=" * 60)
    logger.info("Datasets-Exploration abgeschlossen")


if __name__ == "__main__":
    main()
