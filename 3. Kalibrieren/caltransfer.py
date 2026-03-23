#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalibration übertragen – nur Verschiebung der 0. Ordnung
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import re

# Ordner aus TXT laden 
TXT_PATH = r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt"
CALIBRATION_CSV = r"C:\Users\joche\Desktop\jufo2025-26\Messungencropped\VegaNeu\calibration.csv"
VALID_CSV_PATTERN = re.compile(r".*\.csv$", re.IGNORECASE)

def get_folder_from_txt():
    if not os.path.isfile(TXT_PATH):
        raise FileNotFoundError(f"TXT nicht gefunden: {TXT_PATH}")
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        folder = f.read().strip().strip('"')
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Ordner aus TXT existiert nicht: {folder}")
    print(f"→ Ordner aus TXT geladen: {folder}")
    return folder

def find_csv_files(folder):
    files = [f for f in os.listdir(folder) if VALID_CSV_PATTERN.match(f)]
    if not files:
        raise FileNotFoundError("Keine CSV-Dateien im Ordner gefunden.")
    return files

def choose_file(files, prompt):
    print(f"{prompt}:")
    for i,f in enumerate(files,1):
        print(f"{i}. {f}")
    k = int(input(f"Nummer wählen [1-{len(files)}]: ") or 1)
    return files[max(1,min(k,len(files)))-1]

# Datei laden 

def load_1d(path):
    base, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() in ('.fits', '.fit', '.fts'):
        y = np.asarray(fits.getdata(path), dtype=float).ravel()
        x = np.arange(y.size, dtype=float)
    else:
        # Prüfen, ob die erste Zeile ein Header ist
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            skiprows = 1 if any(c.isalpha() for c in first_line) else 0

        arr = np.loadtxt(path, delimiter=',', comments='#', skiprows=skiprows)
        if arr.ndim == 1:
            y = arr.astype(float)
            x = np.arange(y.size, dtype=float)
        else:
            x = np.asarray(arr[:,0], dtype=float)
            y = np.asarray(arr[:,1], dtype=float)
    return x, y, base


#  Kalibration laden 

def load_calibration(csv_path):
    if not os.path.isfile(csv_path):
        sys.exit(f"Kalibrationsdatei nicht gefunden: {csv_path}")
    arr = np.loadtxt(csv_path, delimiter=',', comments='#', skiprows=1)
    lam, pix = arr[:,0], arr[:,1]
    return pix, lam

#  Verschiebung anwenden 

def shift_calibration(pix_old, new_pixel_zero):
    old_zero_pixel = pix_old[0]
    shift = new_pixel_zero - old_zero_pixel
    pix_new = pix_old + shift
    return pix_new

#  Wellenlängen interpolieren 

def interpolate_wavelengths(pix_new, lam_cal, n_pixels):
    return np.interp(np.arange(n_pixels), pix_new, lam_cal)

# Speichern 

def save_spectrum(base, lam, y, folder):
    mask = (lam >= 3900) & (lam <= 7000)
    lam_save = lam[mask]
    y_save = y[mask]

    base_clean = os.path.splitext(base)[0]
    fits_path = os.path.join(folder, f"{base_clean}tc.fits")
    csv_path  = os.path.join(folder, f"{base_clean}tc.csv")
    png_path  = os.path.join(folder, f"{base_clean}tc.png")

    # FITS
    hdu = fits.PrimaryHDU(y_save)
    hdr = hdu.header
    hdr['CUNIT1'] = 'Angstrom'
    hdu.writeto(fits_path, overwrite=True)

    # CSV
    np.savetxt(csv_path, np.column_stack([lam_save, y_save]),
               delimiter=',', header='lambda_A,flux', comments='', fmt='%.6f')

    # PNG
    plt.figure(figsize=(10,5))
    plt.plot(lam_save, y_save)
    plt.xlabel("Wellenlänge [A]")
    plt.ylabel("Flux")
    plt.title(f"{base_clean} – kalibriert")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"\n Gespeichert direkt im Ordner:\n- {fits_path}\n- {csv_path}\n- {png_path}\n")

# Hauptprogramm 

def main():
    folder = get_folder_from_txt()
    os.chdir(folder)
    print(f"Arbeitsverzeichnis gesetzt: {os.getcwd()}")

    # CSV-Dateien finden
    csv_files = find_csv_files(folder)
    if not csv_files:
        sys.exit("Keine CSV-Dateien im Ordner gefunden.")

    # 1D-Spektrum auswählen
    spec_path = choose_file(csv_files, "Wähle das 1D-Spektrum")
    x, y, base = load_1d(spec_path)

    # Kalibration fest setzen
    pix_old, lam_cal = load_calibration(CALIBRATION_CSV)

    # Plot zur Auswahl der neuen 0. Ordnung
    print("Bitte auf die neue 0. Ordnung klicken im Plot...")
    plt.figure(figsize=(12,5))
    plt.plot(x, y)
    plt.xlabel("Pixel")
    plt.ylabel("Flux")
    plt.title("Klicke ungefähr auf die neue 0. Ordnung (Peak wird automatisch verfeinert)")
    plt.grid(True)
    click_x = plt.ginput(1)[0][0]
    plt.close()

    # Subpixel-Verfeinerung
    window = 10
    i0 = int(round(click_x))
    lo = max(0, i0 - window)
    hi = min(len(y), i0 + window + 1)
    local_max_index = lo + np.argmax(y[lo:hi])
    new_zero_pixel = float(local_max_index)
    print(f"Neue 0. Ordnung Pixel (Maximum im Fenster): {new_zero_pixel:.2f}")

    # Pixel verschieben

    pix_new = shift_calibration(pix_old, new_zero_pixel)

    # (exakt wie bei einer normalen quad. Kalibration) # Quadratische Kalibration erneut anwenden

    # quadratischen Fit aus den (verschobenen) Kalibrationspunkten bestimmen
    coeff = np.polyfit(pix_new, lam_cal, 2)   # a, b, c

    # Wellenlängen für ALLE Pixel berechnen
    p = np.arange(len(y), dtype=float)
    lam_full = coeff[0]*p**2 + coeff[1]*p + coeff[2]

    print("Quadratische Kalibration:")
    print(f"λ(p) = {coeff[0]:.3e}·p² + {coeff[1]:.3e}·p + {coeff[2]:.3f}")
    print(f"λ-Bereich: {lam_full.min():.1f} – {lam_full.max():.1f} Å")


    # Plot kalibriertes Spektrum
    plt.figure(figsize=(12,5))
    plt.plot(lam_full, y)
    plt.xlabel("Wellenlänge [A]")
    plt.ylabel("Flux")
    plt.title(f"{base} – kalibriert")
    plt.grid(True)
    plt.show()

    # Speichern direkt im Input-Ordner
    save_spectrum(base, lam_full, y, folder)

if __name__ == "__main__":
    main()
