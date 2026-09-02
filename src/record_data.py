"""
Sammelt Screenshots des Balken-Bereichs waehrend du selbst spielst/angelst.
Die Bilder landen zunaechst ungesortiert in data/_unsorted/ - danach musst du
sie von Hand in die vier Klassen-Ordner einsortieren (data/fisch_links usw.).

Ausfuehren: python src/record_data.py
Stoppen: Strg+C im Terminal
"""

import os
import sys
import time
from datetime import datetime

import mss
import numpy as np
from PIL import Image

# Erlaubt Ausführung sowohl aus src/ als auch aus dem Projekt-Root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CAPTURE_REGION, CAPTURE_FPS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "_unsorted")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    interval = 1.0 / CAPTURE_FPS

    print(f"Nehme Bereich auf: {CAPTURE_REGION}")
    print(f"Speichere nach: {OUT_DIR}")
    print("Strg+C zum Stoppen.\n")

    count = 0
    with mss.mss() as sct:
        try:
            while True:
                start = time.time()
                shot = sct.grab(CAPTURE_REGION)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img.save(os.path.join(OUT_DIR, f"{ts}.png"))
                count += 1

                if count % 50 == 0:
                    print(f"{count} Bilder aufgenommen...")

                elapsed = time.time() - start
                time.sleep(max(0, interval - elapsed))
        except KeyboardInterrupt:
            print(f"\nFertig. {count} Bilder in {OUT_DIR} gespeichert.")
            print("Jetzt von Hand in data/fisch_links, data/fisch_rechts, "
                  "data/balken_weg, data/dialog_neuer_fisch einsortieren.")


if __name__ == "__main__":
    main()
