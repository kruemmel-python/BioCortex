<p align="center">
  <img src="docs/images/biocortex-brain.jpg" width="600"/>
  <h2>BioCortex — Ein lebendiges, tensorfreies Lernsystem</h2>
  <p><strong>Lokales Lernen. Neuromodulation. Replay. Erklärbarkeit.</strong></p>
  <a href="https://github.com/kruemmel-python/BioCortex/stargazers"><img src="https://img.shields.io/github/stars/kruemmel-python/BioCortex?style=social" /></a>
  <a href="https://github.com/kruemmel-python/BioCortex/forks"><img src="https://img.shields.io/github/forks/kruemmel-python/BioCortex?style=social" /></a>
  <a href="https://github.com/kruemmel-python/BioCortex/issues"><img src="https://img.shields.io/github/issues/kruemmel-python/BioCortex" /></a>
  <a href="https://github.com/kruemmel-python/BioCortex/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kruemmel-python/BioCortex" /></a>
</p>



> **Erklärbare KI ohne Backprop.**
> BioCortex lernt mit lokalen Regeln (STDP), Neuromodulation, Pheromon-Dynamik und Hippocampus-Replay – auf einer robusten **Kneser-Ney**-Sprachmodellbasis. Transparent. Reproduzierbar. Forschbar.

<img width="150" height="150" alt="logo" src="https://github.com/user-attachments/assets/3d153770-667d-42a2-bf79-ee0dd4426c2b" />

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit-red.svg)](#biocortex-studio-streamlit-ui)

---

## Inhaltsverzeichnis

* [Motivation](#motivation)
* [Kernideen](#kernideen)
* [Schnellstart](#schnellstart)
* [Installation](#installation)
* [Benutzung](#benutzung)

  * [CLI](#cli)
  * [Python-API](#python-api)
  * [BioCortex Studio (Streamlit UI)](#biocortex-studio-streamlit-ui)
* [Logging, Metriken & Persistenz](#logging-metriken--persistenz)
* [Projektstruktur](#projektstruktur)
* [Konfiguration](#konfiguration)
* [Roadmap](#roadmap)
* [Beitrag & Support](#beitrag--support)
* [Lizenz](#lizenz)
* [Zitation](#zitation)

---

## Motivation

Die meisten modernen Sprachmodelle sind Black Boxes: riesige Tensoren, versteckte Gradienten, schwer zu erklären. **BioCortex** geht den anderen Weg:

* **Lernen als Beziehung** statt reine Berechnung
* **Lokale Regeln** statt globaler Backpropagation
* **Erklärbarkeit & Transparenz** als erste Bürger

Das zugehörige Essay: *“Manifest des Lebendigen Denkens – KI ohne Tensoren”* (siehe `manifest.md`).

---

## Kernideen

* **BioBPE-Tokenizer** – erklärbare Subword-Tokenisierung (häufige Paarungen → Fusion).
* **Kneser-Ney N-Gramm** – statistische, tensorfreie Basis für Next-Token-Wahrscheinlichkeiten.
* **Synapsen-Graph** – gerichtete Übergänge über Token mit **STDP**-ähnlichen Updates.
* **Neuromodulation** – Dopamin/Überraschung/Bindung skalieren Plastizität.
* **Pheromon-Dynamik** – Pfadnutzung verstärkt/verblasst wie bei Ameisen.
* **Hippocampus-Replay** – stichprobenartige Reaktivierung zur Gedächtniskonsolidierung.
* **Erzeugung** – KN-Wahrscheinlichkeiten × Bio-Bias → Top-p (Nucleus) Sampling.
* **Erklärbarkeit** – Graph-Visualisierung, Modulator-Verläufe, Replay-Logs.

---

## Schnellstart

```bash
# 1) Projekt klonen
git clone https://github.com/kruemmel-python/BioCortex.git
cd <REPO>

# 2) Umgebung anlegen (empfohlen)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Abhängigkeiten
pip install -r requirements.txt

# 4) Erstes Training (CLI)
python biollm.py train --data data/sample.txt --out models/biocortex_demo

# 5) Generierung
python biollm.py generate --model models/biocortex_demo --prompt "Hallo BioCortex," --max_new 80 --top_p 0.9
```

---

## Installation

**Voraussetzungen**

* Python **3.10+**
* Plattform: Linux, macOS, Windows
* Keine GPU/Frameworks erforderlich (rein `numpy`-basiert)

**Paketabhängigkeiten**

```
numpy
streamlit
matplotlib
networkx
pandas
```

> Installiere alles bequem via `pip install -r requirements.txt`.

---

## Benutzung

### CLI

Trainiere ein neues Modell:

```bash
python biollm.py train --data data/*.txt --out models/my_model
```

Feintuning (inkrementell) eines bestehenden Modells:

```bash
python biollm.py finetune --model models/my_model --data new_texts/*.txt
```

Textgenerierung:

```bash
python biollm.py generate --model models/my_model \
  --prompt "Die Natur lehrt uns," --max_new 128 --top_p 0.9
```

**Hinweis zu Pfaden:** `--out/--model` sind **Basispfade**; der Code legt/erwartet mehrere Dateien mit Endungen wie `.cfg.json`, `.bpe.json`, `.kn.json`, `.graph.json`, `.meta.json`.

---

### Python-API

```python
from biollm import BioCortex, BioLLMConfig, BioBPEConfig, KNConfig

# Konfiguration (Defaults sind bewusst konservativ)
cfg = BioLLMConfig(
    bpe=BioBPEConfig(vocab_size=4000, min_pair_freq=2, lowercase=True),
    kn=KNConfig(order=5, discount=0.75)
)

model = BioCortex(cfg)
model.fit(["Das ist ein Beispieltext.", "Noch ein kurzer Text."])

text = model.generate("BioCortex lernt", max_new_tokens=80, top_p=0.9)
print(text)

model.save("models/biocortex_example")
model2 = BioCortex.load("models/biocortex_example")
```

---

### BioCortex Studio (Streamlit UI)

Interaktive Oberfläche für Training, Feintuning, Generierung, Graph-Visualisierung, Modulator- und Replay-Plots.

```bash
streamlit run streamlit_app.py
```

Features im Studio:

* Parameter-Tuning (BPE/KN/Plastizität/Replay/Sampling)
* Fortschritt & Events während Training/Replay
* **Synapsen-Graph** (Gewichte/Pheromone, Dichte, Clustering)
* **Neuromodulatoren-Verläufe** und **Replay-Dynamik**
* Modelle **speichern/laden**
---
<img width="500" height="500" alt="gui" src="https://github.com/user-attachments/assets/590d66f4-549c-44e7-8fca-4996f9fc1f1f" />


<img width="500" height="500" alt="bio-synapsen-500-kanten" src="https://github.com/user-attachments/assets/fb220673-6097-4e0b-b23f-dba4303163ec" />

---

## Logging, Metriken & Persistenz

* **Logging** (`biollm_logging.py`) schreibt mit Zeitstempel nach `logs/biocortex-train_YYYY-MM-DD_HH-MM-SS.log`.
* **Laufende Metriken** in `logs/metrics.csv` (u. a. mean_pher, mean_weight, dopamine, surprise, bond, ltp/ltd, replay_flag).
* **Persistenzformate** (unter Basispfad `models/<name>`):

  * `*.cfg.json` – Komplettkonfiguration
  * `*.bpe.json` – Tokenizer-Vokabular
  * `*.kn.json` – Kneser-Ney-Zähler/Successors
  * `*.graph.json` – Synapsengewichte + Pheromone
  * `*.meta.json` – Trainingsmetadaten inkl. Replay-Aktivität

---

## Projektstruktur

```
.
├─ biollm.py               # Kernsystem (BioBPE, KN-LM, BioGraph, Replay, CLI)
├─ biollm_logging.py       # Zentrales Logging-Setup
├─ streamlit_app.py        # "BioCortex Studio" – interaktive UI
├─ data/                   # Beispieltexte (optional)
├─ models/                 # Gespeicherte Modelle (output)
├─ logs/                   # Logs & CSV-Metriken (auto)
├─ docs/
│  ├─ images/
│  │  ├─ biocortex-brain.jpg
│  │  └─ studio_screenshot.png
│  └─ manifest.md          # Manifest des Lebendigen Denkens
└─ requirements.txt
```

---

## Konfiguration

Wichtige Parameter (Auszug):

| Bereich         | Parameter                                      | Bedeutung                            |
| --------------- | ---------------------------------------------- | ------------------------------------ |
| **BPE**         | `vocab_size`, `min_pair_freq`, `lowercase`     | Größe/Granularität der Subwords      |
| **KN-LM**       | `order`, `discount`                            | N-Gramm-Ordnung, Kneser-Ney-Discount |
| **Plastizität** | `a_plus`, `a_minus`, `tau`                     | STDP-Fenster & LTP/LTD               |
|                 | `dopamine_gain`, `surprise_gain`, `bond_gain`  | Modulator-Skalierungen               |
|                 | `pher_evaporation`, `pher_reinforce`           | Pheromon-Dynamik                     |
| **Replay**      | `buffer_size`, `sample_len`, `nightly_samples` | Gedächtniskonsolidierung             |
| **Generierung** | `max_gen_len`, `top_p`, `gamma_bias`           | Sampling & Bio-Bias-Faktor           |

**Adaptives Verhalten:** während Training/Generierung werden `top_p` und `discount` dynamisch an Dopamin- und Pheromon-Niveaus angepasst.

---

## Roadmap

* [ ] Diskrete **Themen-/Kontext-Mycel**-Edges (semantische Nähe)
* [ ] Export/Import von **Graph-Schnappschüssen** für Vergleichsstudien
* [ ] Evaluations-Notebooks (Perplexity, Divergenz, Stabilitätsmetriken)
* [ ] Optionaler **Audio/Text-Hybrid** (experimentell)
* [ ] Dockerfile + reproducible env

Vorschläge willkommen! → Issues/Discussions

---

## Beitrag & Support

Beiträge sind sehr willkommen ❤️

1. Issue eröffnen (Bug/Feature/Design)
2. Fork → Feature-Branch → PR
3. Bitte **Tests/Beispiele** beilegen (wo sinnvoll) und Stil beibehalten.

> Ethik-Hinweis: Dieses Projekt steht für **Transparenz, Kontextbewusstsein und Demut**. Anwendungen, die Menschen schaden, sind nicht erwünscht.

---

## Lizenz

**MIT License** – siehe `LICENSE`.

---

## Zitation

Wenn du BioCortex wissenschaftlich verwendest, zitiere bitte so:

```
Ralf Krümmel (2025). BioCortex: Ein lebendiges, tensorfreies Lernsystem.
https://github.com/<USER>/<REPO>
```

---

### Danksagung

An alle System-Denker:innen, die Tiefe über Größe stellen – und an die Natur als Lehrerin.
