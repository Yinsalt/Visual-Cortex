# Visual Cortex Simulation Toolkit

Dieses Repository enthält ein Jupyter Notebook (`Vision.ipynb`) sowie begleitende Python-Skripte und Notizen zur geometrischen und funktionalen Modellierung des visuellen Cortex. Ziel ist die Entwicklung und Visualisierung räumlicher neuronaler Strukturen für Simulations- und Forschungszwecke, insbesondere im Kontext von spikenden neuronalen Netzwerken (SNNs).

## Features

- **Geometrische Generatoren:**  
  Erzeuge Kreise, Kegel, Blobs und gestapelte 2D-Grids in 3D-Raum mit frei wählbaren Parametern (Position, Orientierung, Größe). Ideal zur Visualisierung und für die Platzierung neuronaler Populationen.

- **Wave Function Collapse für Neuronen-Labels:**  
  Prozedurale Zuordnung von Zelltypen im Raum — mit kontrollierbaren Wahrscheinlichkeiten und Clustering-Effekten. Ermöglicht realistischere, aber flexibel-beeinflussbare neuronale Architekturen.

- **Visualisierung:**  
  Interaktive 3D-Plots mit matplotlib für alle generierten Strukturen.

- **Helper Functions:**  
  Werkzeuge zur Extraktion, Analyse und Visualisierung von Netzwerkverbindungen (z.B. für NEST-Simulationen).

- **Literatur- und Link-Übersicht:**  
  Sammlung zentraler Papers und Ressourcen zum Thema visueller Cortex und SNN-Modelle.

## Getting Started

### Voraussetzungen

- Python 3.8+
- Empfohlene Libraries:
    - `numpy`
    - `matplotlib`
    - `pandas`
    - `networkx`
    - `nest` (für Simulationen, optional)
    - `jupyter` (für interaktive Notebooks)

### Installation

1. Repository klonen:
    ```
    git clone https://github.com/Yinsalt/Visual-Cortex.git
    ```
2. Abhängigkeiten installieren (z.B. via pip):
    ```
    pip install numpy matplotlib pandas networkx jupyter
    ```
   Für NEST siehe [NEST Doku](https://www.nest-simulator.org/).

3. Notebook öffnen:
    ```
    jupyter notebook Vision.ipynb
    ```

## Anwendung

- Starte das Notebook und führe Zellen schrittweise aus.
- Passe Parameter in den Geometrie-Funktionen (Kreise, Kegel, Grids, Blobs) an, um verschiedene neuronale Populationen zu erzeugen.
- Nutze die Wave-Collapse-Funktionen, um Zelltypen im Raum zu verteilen.
- Die Visualisierungen helfen bei der Verifikation und Exploration der Strukturen.

## Hinweise/Status

- Das Projekt ist work-in-progress und ständig in Entwicklung.
- Viele Funktionen sind modular und können einfach für eigene Experimente angepasst werden.
- Für Feedback, Fragen oder Beiträge gern Issues oder Pull Requests einreichen!

## Literatur & Links

Im Notebook findest du eine kommentierte Sammlung relevanter Papers und Ressourcen zu Modellen des visuellen Cortex und SNNs.

---

**Autor:** [Yinsalt](https://github.com/Yinsalt)
