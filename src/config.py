"""
Zentrale Konfiguration.
WICHTIG: CAPTURE_REGION MUSST du an deine Bildschirmauflösung und
Fensterposition anpassen. Nutze src/pick_region.py dafür (siehe README).
"""

# Bereich des Balkens mit dem Fisch (x, y, breite, höhe) in Bildschirm-Pixeln
CAPTURE_REGION = {
    "left": 501,
    "top": 909,
    "width": 917,
    "height": 63,
}

# Wie oft pro Sekunde gescannt wird
CAPTURE_FPS = 15

# Bildgröße, auf die jeder Ausschnitt für das Modell skaliert wird
MODEL_INPUT_SIZE = (64, 64)  # (Breite, Höhe) - klein halten = schnell

# Klassen in fester Reihenfolge (Index = Modell-Output-Index)
# Beschreibt die Position des Fisches relativ zum steuerbaren Balken:
#   fisch_rechts_von_balken -> Balken muss nach rechts (halten)
#   fisch_links_von_balken -> Balken muss nach links (loslassen)
#   fisch_auf_balken -> Position halten, nichts aendern
#   balken_weg -> kein Balken sichtbar, einmal klicken (neu starten)
CLASSES = [
    "balken_weg",
    "fisch_auf_balken",
    "fisch_links_von_balken",
    "fisch_rechts_von_balken",
]

# Wie viele gleiche Klassifikationen in Folge nötig sind, bevor die Aktion
# ausgeführt wird (gegen Flackern/Fehlklassifikationen)
DEBOUNCE_FRAMES = 2

# Pfad zum trainierten Modell
MODEL_PATH = "models/fishbot_cnn.pt"
