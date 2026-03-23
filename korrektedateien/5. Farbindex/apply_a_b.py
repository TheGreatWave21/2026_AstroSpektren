#!/usr/bin/env python3
"""
colorindex_teff.py
Bestimmung der effektiven Temperatur eines Sterns aus einem atmosphärisch korrigierten Spektrum (CSV)
"""

import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import re

# Pfad-Datei: hier steht der Ordnerpfad nicht datei!!!!!
PATH_FILE = r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt"

# Parameter (Fenster und Kalibration)
B_WINDOW = (390, 490)   # nm
V_WINDOW = (500, 600)   # nm

CAL_A = 1.29826
CAL_B = -0.06840


# Hilfsfunktionen

def read_folder_path(path_file):
    """Liest die Textdatei, bereinigt den Pfad und prüft, ob er existiert."""
    with open(path_file, "r", encoding="utf-8") as f:
        folder = f.readline()
    folder = re.sub(r'[\x00-\x1f\x7f]', '', folder)
    folder = folder.strip()
    folder = os.path.normpath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Ordner existiert nicht: {folder}\n"
                                "Bitte prüfen: Leerzeichen, Tippfehler, existierender Ordner?")
    return folder

def find_csv_files(folder):
    """Findet alle Dateien, die auf '_atmosphere_corrected.csv' enden."""
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith("_atmosphere_corrected.csv")]
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien gefunden in {folder}")
    return sorted(files)

def read_spectrum_csv(path):
    df = pd.read_csv(path, comment="#", encoding='latin1')
    df["wl_nm"] = df["lambda_A"] * 0.1
    df["flux"] = df["S_obs_atm_corrected"]
    return df[["wl_nm", "flux"]]

def sum_flux_in_window(df, win):
    low, high = win
    sel = df[(df["wl_nm"] >= low) & (df["wl_nm"] <= high)]
    return sel["flux"].sum()

def compute_CI(F_B, F_V):
    eps = 1e-30
    return -2.5 * math.log10(max(F_B, eps) / max(F_V, eps))

def ci_to_bv(ci, a=CAL_A, b=CAL_B):
    return a * ci + b

def ballesteros_teff(bv):
    d1 = 0.92 * bv + 1.7
    d2 = 0.92 * bv + 0.62
    if d1 == 0 or d2 == 0:
        return float("nan")
    return 4600 * (1/d1 + 1/d2)

def save_results_txt(results, out_path):
    with open(out_path, "w") as f:
        f.write("=== Ergebnis ===\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

def plot_spectrum(df, results, out_path):
    plt.figure(figsize=(10,6))
    plt.plot(df["wl_nm"], df["flux"], label="Spektrum", color='blue')
    plt.axvspan(*B_WINDOW, color='cyan', alpha=0.3, label='B Fenster')
    plt.axvspan(*V_WINDOW, color='yellow', alpha=0.3, label='V Fenster')
    txt = f"B-V = {results['B_V']:.4f}\nT_eff = {results['Teff_K']:.1f} K"
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Flux")
    plt.title(f"Spektrum: {results['file']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def process_file(path):
    df = read_spectrum_csv(path)
    F_B = sum_flux_in_window(df, B_WINDOW)
    F_V = sum_flux_in_window(df, V_WINDOW)
    CI = compute_CI(F_B, F_V)
    BV = ci_to_bv(CI)
    Teff = ballesteros_teff(BV)

    results = {
        "file": os.path.basename(path),
        "F_B": F_B,
        "F_V": F_V,
        "CI": CI,
        "B_V": BV,
        "Teff_K": Teff
    }

    print("\n=== Ergebnis ===")
    for k,v in results.items():
        print(f"{k}: {v}")

    folder = os.path.dirname(path)
    # Sternennamen aus Dateiname extrahieren (z.B. '250810_20.09_vega_atmosphere_corrected.csv')
    match = re.search(r'_(\w+)_atmosphere_corrected\.csv$', os.path.basename(path))
    if match:
        star_name = match.group(1).lower()
    else:
        star_name = "unknown_star"

    txt_path = os.path.join(folder, f"{star_name}_farbindex_temp.txt")
    png_path = os.path.join(folder, f"{star_name}_farbindex_temp.png")
    save_results_txt(results, txt_path)
    plot_spectrum(df, results, png_path)

    print(f"\nErgebnisse gespeichert als:\nTXT: {txt_path}\nPNG: {png_path}")
    return results


# Skript ausführen
if __name__ == "__main__":
    folder_path = read_folder_path(PATH_FILE)
    csv_files = find_csv_files(folder_path)
    for csv_file in csv_files:
        process_file(csv_file)
