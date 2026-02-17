"""
Aufgabe 1.4: Verstaendnis-Fragen (PFLICHT)
============================================
Alle 6 Fragen in eigenen Worten beantwortet.

Kurs: Morphos GmbH - KI & Python Modul, Woche 2
"""

# ============================================
# FRAGE 1: learning_rate
# ============================================
# In der Vorlesung war learning_rate=2e-5 (also 0.00002).
# Euer CNN von Woche 1 hatte learning_rate=0.001.
#
# a) Warum ist die Learning Rate beim Fine-Tuning 50x kleiner?
#
#    Beim Fine-Tuning hat das Model (DistilBERT) bereits 67 Millionen
#    Parameter die auf riesigen Textmengen vortrainiert wurden.
#    Dieses Vorwissen (Sprachverstaendnis, Grammatik, Semantik) ist
#    extrem wertvoll. Eine hohe LR wuerde diese gelernten Gewichte
#    zu stark veraendern und das Vorwissen zerstoeren.
#    Beim CNN von Woche 1 starteten alle Gewichte ZUFAELLIG -> da
#    muss man schnell lernen, also groessere LR.
#
# b) Was passiert wenn ihr learning_rate=0.1 benutzt?
#
#    Das Model wuerde "katastrophales Vergessen" (Catastrophic Forgetting)
#    erleiden: Die vortrainierten Sprachfaehigkeiten gehen verloren,
#    der Loss explodiert oder oszilliert wild, und das Model lernt
#    effektiv gar nichts Sinnvolles. Es springt ueber das Minimum hinaus.
#
# c) Wie heisst das Problem wenn die LR zu gross ist? (Englischer Fachbegriff)
#
#    "Catastrophic Forgetting" (bei Fine-Tuning vortrainierter Models)
#    bzw. "Gradient Explosion" / "Divergence" (allgemein bei zu hoher LR).


# ============================================
# FRAGE 2: Datasets
# ============================================
# In Aufgabe 1.2 habt ihr load_dataset("stanfordnlp/imdb") benutzt.
# In Aufgabe 1.3 habt ihr eigene Listen geschrieben.
#
# a) Was ist der Vorteil von load_dataset() gegenueber eigenen Listen?
#
#    1. SKALIERUNG: Tausende/Millionen Texte mit einer Zeile Code
#    2. QUALITAET: Professionell gelabelt, peer-reviewed, Benchmark-Standard
#    3. REPRODUZIERBARKEIT: Jeder kann exakt das gleiche Dataset laden
#    4. EFFIZIENZ: Automatisches Caching, Memory-Mapping, Streaming
#    5. VORDEFINIERTE SPLITS: Train/Test/Validation -> kein Data Leakage
#
# b) Warum haben wir nur 500 Reviews genommen statt alle 25.000?
#
#    Weil wir auf CPU trainieren. 25.000 Reviews x 3 Epochs x 256 Tokens
#    wuerde Stunden dauern. 500 Reviews reichen um das Prinzip zu lernen
#    und liefern in ca. 10-20 Minuten ein Ergebnis. In Produktion mit
#    GPU wuerde man natuerlich alle 25.000 (oder mehr) nehmen.
#
# c) Was wuerde passieren wenn wir ALLE 25.000 auf CPU nehmen?
#
#    Das Training wuerde ca. 3-6 Stunden dauern (statt 10-20 Minuten).
#    Es wuerde funktionieren, aber die Wartezeit waere fuer den Kurs
#    nicht praktikabel. Der RAM-Verbrauch steigt auch stark, besonders
#    beim Tokenisieren und beim Gradient-Speicher.


# ============================================
# FRAGE 3: Metriken
# ============================================
# In der Vorlesung hatten wir nur eval_loss.
# In Aufgabe 1.2 habt ihr compute_metrics mit Accuracy und F1 benutzt.
#
# a) Warum reicht Accuracy NICHT aus? Gebt ein konkretes Beispiel.
#
#    Beispiel: Ein Spam-Detektor hat 95% normale Emails und 5% Spam.
#    Wenn das Model IMMER "kein Spam" sagt, hat es 95% Accuracy!
#    Aber es erkennt keinen einzigen Spam -> Recall = 0%.
#    Accuracy luegt bei unbalancierten Daten.
#    F1 kombiniert Precision und Recall und zeigt das wahre Bild.
#
# b) Was ist der Unterschied zwischen Precision und Recall?
#
#    PRECISION: Von allen Texten die das Model als POSITIV vorhersagt,
#    wie viele sind tatsaechlich positiv? -> "Wie praezise sind die Vorhersagen?"
#    Formel: TP / (TP + FP)
#
#    RECALL: Von allen tatsaechlich POSITIVEN Texten, wie viele hat das
#    Model gefunden? -> "Wie vollstaendig findet das Model alle Positiven?"
#    Formel: TP / (TP + FN)
#
#    Beispiel: Krebsdiagnose -> hoher Recall wichtig (keinen Fall uebersehen!)
#    Beispiel: Spam-Filter -> hohe Precision wichtig (keine echte Mail loeschen!)
#
# c) Was zeigt die Confusion Matrix was Accuracy alleine nicht zeigt?
#
#    Die Confusion Matrix zeigt die FEHLERVERTEILUNG:
#    - Wie viele False Positives? (Negative faelschlich als Positiv)
#    - Wie viele False Negatives? (Positive faelschlich als Negativ)
#    - Wo genau macht das Model systematische Fehler?
#    Accuracy sagt nur "87% richtig" aber nicht WO die 13% Fehler liegen.


# ============================================
# FRAGE 4: Deutsche vs Englische Models
# ============================================
#
# a) Warum funktioniert "distilbert-base-uncased" schlecht auf deutschen Texten?
#
#    distilbert-base-uncased wurde NUR auf englischen Texten trainiert.
#    Es kennt keine deutschen Woerter, keine deutsche Grammatik, keine
#    Umlaute. Sein Tokenizer zerlegt deutsche Woerter in unsinnige
#    Sub-Tokens. "fantastisch" wird z.B. zu ["fan", "##tas", "##tisch"]
#    statt als bekanntes Wort erkannt zu werden.
#
# b) Was bedeutet "cased" vs "uncased"?
#
#    UNCASED: Alle Buchstaben werden zu Kleinschreibung konvertiert
#    BEFORE Tokenisierung. "Berlin" -> "berlin". Verliert Gross/Klein-Info.
#
#    CASED: Gross/Kleinschreibung wird beibehalten. "Berlin" bleibt "Berlin".
#    Wichtig fuer Deutsch weil: Nomen werden gross geschrieben!
#    "Er machte die Tur auf" vs "Er machte die Tur AUF" (Partikel).
#
# c) Welches deutsche Model habt ihr gewaehlt und warum?
#
#    deepset/gbert-base - Gruende:
#    1. Speziell auf 163GB deutschen Texten trainiert
#    2. Von deepset.ai, einem deutschen AI-Unternehmen (Haystack/RAG)
#    3. Bessere Performance auf deutschen NLP-Benchmarks als mBERT
#    4. CASED -> erkennt deutsche Gross/Kleinschreibung korrekt


# ============================================
# FRAGE 5: Fine-Tuning vs Transfer Learning
# ============================================
#
# a) Ist Fine-Tuning das gleiche wie Transfer Learning? Erklaert.
#
#    Fine-Tuning ist eine FORM von Transfer Learning, aber nicht dasselbe.
#
#    TRANSFER LEARNING (allgemein): Vorwissen von einer Aufgabe auf eine
#    andere uebertragen. Kann verschiedene Formen haben:
#    - Feature Extraction (Layers einfrieren, nur neuen Head trainieren)
#    - Fine-Tuning (alle oder manche Layers weiter trainieren)
#
#    FINE-TUNING (spezifisch): Das gesamte vortrainierte Model wird
#    auf der neuen Aufgabe WEITER trainiert (alle Gewichte angepasst).
#
#    -> Fine-Tuning ist IMMER Transfer Learning,
#       aber Transfer Learning ist NICHT immer Fine-Tuning.
#
# b) Bei VGG16 haben wir Layers eingefroren (base_model.trainable=False).
#    Beim DistilBERT waren ALLE 66M Parameter trainable.
#    Was ist der Unterschied in der Strategie?
#
#    VGG16-Strategie: "Feature Extraction"
#    -> Die CNN-Basis (gelernte Bild-Features) bleibt fix
#    -> Nur der neue Dense-Head wird trainiert (~wenige 1000 Parameter)
#    -> Schnell, wenig Daten noetig, wenig Overfitting-Risiko
#
#    DistilBERT-Strategie: "Full Fine-Tuning"
#    -> ALLE 66M Parameter werden angepasst (sehr kleine LR!)
#    -> Das gesamte Sprachverstaendnis wird an die Aufgabe angepasst
#    -> Langsamer, braucht mehr Daten, bessere Ergebnisse
#
# c) Wann friert man Layers ein, wann nicht?
#
#    EINFRIEREN (Feature Extraction):
#    - Wenige Trainingsdaten (< 1000)
#    - Aufgabe aehnlich zu Pretraining-Aufgabe
#    - Wenig GPU/Zeit verfuegbar
#    - Overfitting-Risiko hoch
#
#    NICHT EINFRIEREN (Full Fine-Tuning):
#    - Genuegend Trainingsdaten (> 1000-5000)
#    - Aufgabe unterscheidet sich von Pretraining
#    - GPU verfuegbar
#    - Maximale Performance gewuenscht


# ============================================
# FRAGE 6: DataCollator vs padding=True
# ============================================
#
# a) Was ist der Unterschied?
#
#    padding=True: ALLE Texte werden auf die Laenge des laengsten Texts
#    im GESAMTEN Dataset gepadded. Einmal am Anfang, fuer alle gleich.
#
#    DataCollatorWithPadding: Texte werden pro BATCH auf die Laenge des
#    laengsten Texts IN DIESEM BATCH gepadded. Jeder Batch kann anders lang sein.
#
# b) Warum ist DataCollator effizienter?
#
#    Weil kurze Texte nicht unnoetig viele Padding-Tokens bekommen.
#    Weniger Padding = weniger Berechnung = schnelleres Training + weniger RAM.
#
# c) Gebt ein Beispiel mit konkreten Zahlen.
#    (z.B. 3 Texte mit Laenge 10, 50, 200)
#
#    3 Texte: Laenge 10, 50, 200 Tokens
#
#    MIT padding=True (alles auf max_length):
#      Text 1: 10 echte + 190 Padding = 200 Tokens  (95% verschwendet!)
#      Text 2: 50 echte + 150 Padding = 200 Tokens  (75% verschwendet!)
#      Text 3: 200 echte + 0 Padding  = 200 Tokens
#      TOTAL: 600 Tokens verarbeitet, davon 340 Padding (57% Verschwendung)
#
#    MIT DataCollator (Padding pro Batch):
#      Batch 1 (Text 1 + Text 2): Pad auf 50 -> 50 + 50 = 100 Tokens
#      Batch 2 (Text 3): 200 Tokens
#      TOTAL: 300 Tokens verarbeitet, davon nur 40 Padding (13% Verschwendung)
#
#    -> DataCollator verarbeitet HALB so viele Tokens!
#       Bei 1000 Texten ist der Unterschied noch groesser.
