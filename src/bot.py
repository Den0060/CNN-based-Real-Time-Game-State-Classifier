"""
Hauptskript: scannt den Bildschirm live, klassifiziert den Zustand mit dem
trainierten Modell und steuert die Maus entsprechend.

Zustaende -> Aktionen:
  fisch_rechts_von_balken -> Linksklick HALTEN (Balken soll nach rechts wandern)
  fisch_links_von_balken  -> Linksklick LOSLASSEN (Balken soll nach links wandern)
  fisch_auf_balken         -> nichts tun (aktuellen Zustand halten, Position ist gut)
  balken_weg               -> einmal Linksklick (kurzer Klick, startet/erneuert das Angeln)

Ausfuehren: python src/bot.py
Stoppen: Strg+C im Terminal (oder Failsafe: Maus in die Bildschirmecke)
"""

import os
import sys
import time
from collections import deque

import mss
import numpy as np
import pyautogui
import torch
from PIL import Image
from torchvision import transforms

# Erlaubt Ausführung sowohl aus src/ als auch aus dem Projekt-Root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAPTURE_REGION,
    CAPTURE_FPS,
    MODEL_INPUT_SIZE,
    MODEL_PATH,
    DEBOUNCE_FRAMES,
)
from model import FishBotCNN

# Modellpfad relativ zum Projekt-Root, unabhängig vom Arbeitsverzeichnis
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)

pyautogui.FAILSAFE = True  # Maus in Bildschirmecke = Not-Stopp

transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])),
    transforms.ToTensor(),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    model = FishBotCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def grab_and_classify(sct, model, classes, region):
    shot = sct.grab(region)
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()
    return classes[pred_idx]


def main():
    model, classes = load_model()
    print(f"Modell geladen. Klassen: {classes}")
    print("Bot laeuft. Strg+C zum Stoppen, oder Maus in Bildschirmecke (Failsafe).\n")

    mouse_held = False
    already_clicked = False  # verhindert Mehrfachklick solange balken_weg anhält
    recent_states = deque(maxlen=DEBOUNCE_FRAMES)
    interval = 1.0 / CAPTURE_FPS

    with mss.MSS() as sct:
        try:
            while True:
                start = time.time()

                state = grab_and_classify(sct, model, classes, CAPTURE_REGION)
                recent_states.append(state)

                # Nur handeln, wenn die letzten N Frames uebereinstimmen (Debounce)
                stable_state = None
                if len(recent_states) == DEBOUNCE_FRAMES and len(set(recent_states)) == 1:
                    stable_state = recent_states[0]

                if stable_state == "fisch_rechts_von_balken":
                    already_clicked = False
                    if not mouse_held:
                        pyautogui.mouseDown(button="left")
                        mouse_held = True
                        print("Fisch rechts von Balken -> Maus HALTEN")

                elif stable_state == "fisch_links_von_balken":
                    already_clicked = False
                    if mouse_held:
                        pyautogui.mouseUp(button="left")
                        mouse_held = False
                        print("Fisch links von Balken -> Maus LOSLASSEN")

                elif stable_state == "fisch_auf_balken":
                    already_clicked = False
                    # nichts tun: aktuellen Zustand (gehalten oder losgelassen) beibehalten
                    # Gut für Links wo man sowieso loslässt, nicht aber für Rechts - paar Bilder ganz rechts einfügen
                    # auch wenn der Fisch drin ist, um es zu halten

                elif stable_state == "balken_weg":
                    if mouse_held:
                        pyautogui.mouseUp(button="left")
                        mouse_held = False
                    if not already_clicked:
                        pyautogui.click(button="left")
                        already_clicked = True
                        print("Balken weg -> einmal klicken (Angeln starten)")
                        recent_states.clear()

                elapsed = time.time() - start
                time.sleep(max(0, interval - elapsed))

        except KeyboardInterrupt:
            if mouse_held:
                pyautogui.mouseUp(button="left")
            print("\nBot gestoppt.")


if __name__ == "__main__":
    main()
