"""
Hilfstool: Screenshot machen und per Klick+Ziehen den Bereich auswaehlen,
den du in config.py als CAPTURE_REGION / DIALOG_REGION eintragen sollst.

Ausfuehren: python src/pick_region.py
Dann: Rechteck mit der Maus aufziehen, Fenster schliessen (q oder Enter).
Die Konsole gibt dir das fertige dict fuer config.py aus.
"""

import cv2
import mss
import numpy as np

drawing = False
ix, iy = -1, -1
rect = None


def mouse_cb(event, x, y, flags, param):
    global ix, iy, drawing, rect, img_show, img_orig
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_show = img_orig.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        img_show = img_orig.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)


def main():
    global img_show, img_orig
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # Hauptmonitor; ggf. Index anpassen
        shot = sct.grab(monitor)
        img_orig = np.array(shot)[:, :, :3]  # BGRA -> BGR

    img_show = img_orig.copy()
    cv2.namedWindow("Bereich auswaehlen - ziehen, dann 'q'")
    cv2.setMouseCallback("Bereich auswaehlen - ziehen, dann 'q'", mouse_cb)

    while True:
        cv2.imshow("Bereich auswaehlen - ziehen, dann 'q'", img_show)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 13):  # q oder Enter
            break

    cv2.destroyAllWindows()

    if rect:
        left, top, width, height = rect
        print("\nFuege das hier in config.py ein:\n")
        print(f'{{\n    "left": {left},\n    "top": {top},\n    "width": {width},\n    "height": {height},\n}}')
    else:
        print("Kein Bereich ausgewaehlt.")


if __name__ == "__main__":
    main()
