"""
SA100/ASI1600MM Pro – Align & Stack mit wählbarer Stack-Methode (Median, Mean, Sigma-Clipping)
automatischem Sternnamen im Dateinamen
"""

import os
import glob
import sys
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage

# --- Konstanten ---
SEARCH_WINDOW = 61
CENTROID_WINDOW = 21
SHIFT_ORDER = 3

# Ordnerpfad aus TXT lesen

TXT_PATH = r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt"

def read_folder_from_txt() -> Tuple[str, str]:
    """Liest den Ordnerpfad aus einer Textdatei. Kein Filter."""
    if not os.path.exists(TXT_PATH):
        sys.exit(f" TXT-Datei nicht gefunden:\n{TXT_PATH}")

    with open(TXT_PATH, "r", encoding="utf-8") as f:
        folder = f.read().strip().strip('"').strip("'")

    if not os.path.isdir(folder):
        sys.exit(f" Ungültiger Ordnerpfad in TXT:\n{folder}")

    return folder, ""   # kein Filter

# Hilfsfunktionen

def collect_fits(folder: str, name_filter: str) -> List[str]:
    """Sammelt alle FITS-Dateien im Ordner, optional mit Namensfilter."""
    files = sorted(glob.glob(os.path.join(folder, "*.fits")))
    if name_filter:
        files = [f for f in files if name_filter in os.path.basename(f)]
    if not files:
        sys.exit(" Keine passenden FITS-Dateien gefunden.")
    return files


def percentile_scale(img: np.ndarray, p_low: float = 1, p_high: float = 99.5) -> Tuple[float, float]:
    return tuple(np.percentile(img, [p_low, p_high]))


def show_image_and_get_click(data: np.ndarray, title: str) -> Tuple[float, float]:
    vmin, vmax = percentile_scale(data)
    plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    pts = plt.ginput(1, timeout=-1)
    plt.close()
    if not pts:
        sys.exit(" Kein Klick auf das Bild.")
    return pts[0]


def cutout(img: np.ndarray, xc: float, yc: float, size: int) -> Tuple[np.ndarray, int, int]:
    half = size // 2
    x0, y0 = int(round(xc)) - half, int(round(yc)) - half
    x1, y1 = x0 + size, y0 + size
    return img[max(0, y0):y1, max(0, x0):x1], max(0, x0), max(0, y0)


def brightest_pixel_position(cut: np.ndarray) -> Tuple[int, int]:
    idx = np.nanargmax(cut)
    return np.unravel_index(idx, cut.shape)[::-1]


def centroid_2d(cut: np.ndarray) -> Tuple[float, float]:
    yy, xx = np.indices(cut.shape)
    m = cut.sum()
    if m <= 0:
        return (cut.shape[1]-1)/2.0, (cut.shape[0]-1)/2.0
    cx = (cut * xx).sum() / m
    cy = (cut * yy).sum() / m
    return cx, cy


def find_zero_order(img: np.ndarray, guess_x: float, guess_y: float) -> Tuple[float, float]:
    big, bx0, by0 = cutout(img, guess_x, guess_y, SEARCH_WINDOW)
    bx, by = brightest_pixel_position(big)
    cx_guess, cy_guess = bx0 + bx, by0 + by
    small, sx0, sy0 = cutout(img, cx_guess, cy_guess, CENTROID_WINDOW)
    cx_small, cy_small = centroid_2d(small)
    return sx0 + cx_small, sy0 + cy_small


def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return ndimage.shift(img, shift=(dy, dx), order=SHIFT_ORDER, mode='constant', cval=0.0)


def stack_images(aligned: List[np.ndarray], method: str = 'median') -> np.ndarray:
    method = method.lower()
    if method == 'mean':
        return np.mean(aligned, axis=0)
    elif method == 'sigma':
        clipped = sigma_clip(aligned, sigma=3, axis=0)
        return np.mean(clipped, axis=0)
    else:
        return np.median(aligned, axis=0)


def display_image(img: np.ndarray, title: str):
    vmin, vmax = percentile_scale(img)
    plt.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()


def extract_star_name(filename: str) -> str:
    base = os.path.basename(filename)
    name = base.split('.')[0]
    for sep in ['-', '_']:
        if sep in name:
            name = name.split(sep)[0]
            break
    return name.strip().lower()

# Hauptprogramm

def main():
    # Ordner aus TXT lesen
    folder, filt = read_folder_from_txt()
    print(f" Images-Ordner aus TXT geladen:\n{folder}")

    files = collect_fits(folder, filt)
    print(f"Gefundene Dateien: {len(files)}")

    star_name = extract_star_name(files[0])
    print(f"Stern erkannt: {star_name}")

    data0 = fits.getdata(files[0]).astype(float)
    x_click, y_click = show_image_and_get_click(data0, "Klicke auf die 0. Ordnung (heller Punkt)")
    ref_x, ref_y = find_zero_order(data0, x_click, y_click)

    aligned = []
    for idx, path in enumerate(files, 1):
        data = fits.getdata(path).astype(float)
        cx, cy = find_zero_order(data, ref_x, ref_y)
        dx, dy = ref_x - cx, ref_y - cy
        aligned_img = shift_image(data, dy=dy, dx=dx)
        aligned.append(aligned_img)
        print(f"[{idx}/{len(files)}] {os.path.basename(path)}  → Shift dx={dx:+.3f}, dy={dy:+.3f}")

    # Stack-Methode
    method = input("Stack-Methode wählen (median/mean/sigma) [median]: ").strip().lower() or "median"
    stacked = stack_images(aligned, method=method)

    # Methoden-Kürzel bestimmen
    if method.startswith("med"):
        mcode = "med"
    elif method.startswith("mea"):
        mcode = "mea"
    else:
        mcode = "s"

    # Neuer Dateiname: stern_s(anzahl)(mcode)
    outname = f"{star_name}_s{len(files)}{mcode}.fits"
    outpath = os.path.join(folder, outname)

    hdr = fits.getheader(files[0])
    hdr.add_history(f"Aligned & stacked using {method} method")
    fits.writeto(outpath, stacked.astype(np.float32), hdr, overwrite=True)
    print(f" Gestacktes Bild gespeichert: {outpath}")

    display_image(stacked, f"{method.title()}-Stack ({len(files)} Bilder)")


if __name__ == "__main__":
    main()
