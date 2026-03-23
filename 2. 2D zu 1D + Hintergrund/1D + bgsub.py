#!/usr/bin/env python3
"""
Automatische Ausrichtung + Cropping + 1D-Extraktion mit optionaler Hintergrundsubtraktion
"""

import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from astropy.io import fits
from scipy.ndimage import rotate as imrotate, gaussian_filter1d

# NEU: Datei-Find-Logik aus TXT

VALID_PATTERN = re.compile(r"^(.*)_s\d+(s|med|mea)\.fits$", re.IGNORECASE)

def get_folder_from_txt():
    """Liest automatisch den Ordnerpfad aus der fest definierten TXT-Datei."""
    txt = r"C:\Users\benne\Desktop\Visual Studio Code\Astro\1. stack + align\ordner.txt"

    if not os.path.isfile(txt):
        raise FileNotFoundError(f"TXT nicht gefunden: {txt}")

    with open(txt, "r", encoding="utf-8") as f:
        folder = f.read().strip()

    folder = folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Ordner aus TXT existiert nicht: {folder}")

    print(f"→ Ordner aus TXT geladen: {folder}")
    return folder



def find_valid_stack_file(folder):
    """
    Sucht im Ordner nach Dateien wie:
        <name>_s<number>(s|med|mea).fits
    Gibt entweder eine Datei zurück oder lässt den Nutzer wählen.
    """

    files = os.listdir(folder)
    candidates = []

    for f in files:
        if f.lower().endswith(".fits") and VALID_PATTERN.match(f):
            candidates.append(f)

    if not candidates:
        raise FileNotFoundError("Keine gültige Stack-Datei im Ordner gefunden.")

    # nur 1 Treffer → gut
    if len(candidates) == 1:
        print(f"→ Gefundene Datei: {candidates[0]}")
        return os.path.join(folder, candidates[0])

    # mehrere → Auswahl
    print("\nMehrere gültige Dateien gefunden:")
    for i, f in enumerate(candidates):
        print(f"   [{i}] {f}")

    while True:
        try:
            idx = int(input("Welche Datei soll verwendet werden? Index eingeben: "))
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                print(f"→ Gewählt: {chosen}")
                return os.path.join(folder, chosen)
        except ValueError:
            pass
        print("Ungültige Eingabe.\n")


# --------------------------------------------------------------
# Vorhandene Hilfsfunktionen (unverändert)
# --------------------------------------------------------------

def percentile_scale(img, p_low=1.0, p_high=99.5):
    a = img[np.isfinite(img)]
    if a.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(a, [p_low, p_high])
    return float(vmin), float(vmax)


def projection_metric(img):
    proj = np.nansum(img, axis=1)
    m = proj.mean() + 1e-12
    return float(proj.max() / m)


def find_best_angle_via_projection(img, amin=-5.0, amax=5.0, step=0.05):
    best_angle, best_score = 0.0, -np.inf
    for a in np.arange(amin, amax + step*0.5, step):
        rot = imrotate(img, angle=-a, reshape=False, order=1, mode='constant', cval=0.0, prefilter=True)
        score = projection_metric(rot)
        if score > best_score:
            best_score, best_angle = score, a
    return float(best_angle), float(best_score)


def rotate_image(img, angle_deg):
    return imrotate(img, angle=-angle_deg, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)


def select_crop_rectangle(img):
    vmin, vmax = percentile_scale(img)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title('Ziehe ein Rechteck und drücke ENTER')
    bounds = {'x0': None, 'x1': None, 'y0': None, 'y1': None}

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        bx0, bx1 = sorted([int(round(x0)), int(round(x1))])
        by0, by1 = sorted([int(round(y0)), int(round(y1))])
        bounds['x0'], bounds['x1'], bounds['y0'], bounds['y1'] = bx0, bx1, by0, by1

        ax.lines = []
        ax.axhline(by0, ls='--')
        ax.axhline(by1, ls='--')
        ax.axvline(bx0, ls='--')
        ax.axvline(bx1, ls='--')
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, onselect, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=True
    )

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if None in bounds.values():
        raise SystemExit("Kein Rechteck gewählt")

    return slice(bounds['y0'], bounds['y1']), slice(bounds['x0'], bounds['x1'])


def click_hlines(img, title, n):
    vmin, vmax = percentile_scale(img)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ys = []
    for i in range(n):
        pts = plt.ginput(1, timeout=-1)
        if not pts:
            plt.close(fig)
            raise SystemExit("Kein Klick registriert.")
        ys.append(pts[0][1])
        ax.axhline(ys[-1], ls='--')
        fig.canvas.draw()
    plt.close(fig)
    return np.sort(ys)


def band_from_lines(y1, y2, ny):
    lo, hi = sorted((y1, y2))
    return slice(max(0, int(np.floor(lo))), min(ny, int(np.ceil(hi))))


def extract_1d_spectrum(img, y_ex, y_bg_top, y_bg_bot, apply_bgsub=True, smooth_sigma=3.0):
    ny, nx = img.shape
    sl_ex = band_from_lines(y_ex[0], y_ex[1], ny)
    sl_t = band_from_lines(y_bg_top[0], y_bg_top[1], ny)
    sl_b = band_from_lines(y_bg_bot[0], y_bg_bot[1], ny)

    prof_t = np.nanmedian(img[sl_t, :], axis=0)
    prof_b = np.nanmedian(img[sl_b, :], axis=0)
    bg_prof = 0.5 * (prof_t + prof_b)

    if smooth_sigma > 0:
        bg_prof = gaussian_filter1d(bg_prof, sigma=smooth_sigma)

    if apply_bgsub:
        img_corr = img - bg_prof[None, :]
        spec1d = np.nansum(img_corr[sl_ex, :], axis=0)
    else:
        spec1d = np.nansum(img[sl_ex, :], axis=0)

    return spec1d, bg_prof

# MAIN

def main():
    print("→ Lade Ordner aus TXT…")
    folder = get_folder_from_txt()

    print("→ Suche nach gültigen Stack-Dateien…")
    fits_path = find_valid_stack_file(folder)

    header = None
    try:
        header = fits.getheader(fits_path)
    except Exception:
        header = None

    img = fits.getdata(fits_path).astype(float)

    print("→ Ermittle besten Drehwinkel…")
    best_angle, score = find_best_angle_via_projection(img)
    print(f"   Beste Rotation: {best_angle:.3f}° (Score={score:.3f})")

    rot = rotate_image(img, best_angle)

    print("→ Wähle Crop-Bereich…")
    sY, sX = select_crop_rectangle(rot)
    img_crop = rot[sY, sX]

    print("→ Wähle Spektrumslinien…")
    y_ex = click_hlines(img_crop, "Klicke 2 Linien (oben/unten) für Spektrum", 2)
    print("→ Wähle Hintergrund oben…")
    y_bg_top = click_hlines(img_crop, "Hintergrund oben: 2 Linien", 2)
    print("→ Wähle Hintergrund unten…")
    y_bg_bot = click_hlines(img_crop, "Hintergrund unten: 2 Linien", 2)

    do_bg = input("Hintergrund abziehen? (J/n) [J]: ").strip().lower() or "j"
    apply_bgsub = do_bg.startswith("j") or do_bg.startswith("y")

    spec1d, bg_prof = extract_1d_spectrum(img_crop, y_ex, y_bg_top, y_bg_bot, apply_bgsub)


    # Speichern
    base = os.path.splitext(os.path.basename(fits_path))[0]
    suffix = "1dbgsub" if apply_bgsub else "1draw"

    out_csv = os.path.join(folder, base + suffix + ".csv")
    np.savetxt(out_csv, np.column_stack([np.arange(spec1d.size), spec1d]),
               delimiter=",", header="pixel,flux", comments="")
    print(f"→ CSV gespeichert: {out_csv}")

    print("Fertig.")

    
    # Anzeige
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    vmin,vmax = percentile_scale(img_crop)
    plt.imshow(img_crop, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.axhline(y_ex[0], color='r', ls='--'); plt.axhline(y_ex[1], color='r', ls='--')
    plt.axhline(y_bg_top[0], color='b', ls=':'); plt.axhline(y_bg_top[1], color='b', ls=':')
    plt.axhline(y_bg_bot[0], color='b', ls=':'); plt.axhline(y_bg_bot[1], color='b', ls=':')
    plt.title('Auswahl')

    plt.subplot(1,2,2)
    plt.plot(np.arange(spec1d.size), spec1d)
    plt.title('1D-Spektrum')
    plt.tight_layout()

    #  Plot automatisch speichern

    out_png = os.path.join(folder, base + suffix + ".png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"→ Plot gespeichert: {out_png}")
    plt.show()




if __name__ == "__main__":
    main()
 