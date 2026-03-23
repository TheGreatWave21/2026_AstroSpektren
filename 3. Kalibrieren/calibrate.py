#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA100 – 1D-Spektrum kalibrieren (linear oder quadratisch) mit automatischer
Erkennung von Absorptions- oder Emissionslinien (für 0. Ordnung).
"""

from __future__ import annotations
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BALMER_LINES = {
    "H_alpha": 6562.10,
    "H_beta": 4860.74,
    "H_gamma": 4340.10,
    "H_delta": 4101.74,
    "H_epsilon": 3970.07,
    "H_zeta": 3889.05,
    "H_eta": 3835.38,
}

CHOICES = [
    ("0. Ordnung (0 A)", 0.0),
    ("Hγ (4340.10 A)", 4340.10),
    ("Hβ (4860.74 A)", 4860.74),
    ("Hα (6562.10 A)", 6562.10),
]

# Ordner aus TXT laden

# Nur Dateien, die exakt auf 1draw.csv oder 1dbgsub.csv enden
VALID_PATTERN = re.compile(r".*(1draw|1dbgsub)\.csv$", re.IGNORECASE)
TXT_PATH = r"C:\Users\benne\Desktop\Visual Studio Code\Astro\1. stack + align\ordner.txt"

def get_folder_from_txt() -> str:
    if not os.path.isfile(TXT_PATH):
        raise FileNotFoundError(f"TXT nicht gefunden: {TXT_PATH}")
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        folder = f.read().strip().strip('"')
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Ordner aus TXT existiert nicht: {folder}")
    print(f"→ Ordner aus TXT geladen: {folder}")
    return folder

def find_csv_files(folder: str):
    files = os.listdir(folder)
    candidates = [f for f in files if VALID_PATTERN.match(f)]
    if not candidates:
        raise FileNotFoundError("Keine gültigen CSV-Dateien gefunden (1draw/1dbgsub).")
    if len(candidates) == 1:
        print(f"→ Gefundene Datei: {candidates[0]}")
        return os.path.join(folder, candidates[0])
    # mehrere → Auswahl
    print("\nMehrere gültige CSV-Dateien gefunden:")
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


# 1D-Datei laden


def load_1d(path: str):
    """Lädt CSV-Datei, überspringt Header und unterstützt 1- oder 2-Spalten-Daten."""
    arr = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
    if arr.ndim == 0:
        # Ein Zeilenfall
        y = np.array([arr[arr.dtype.names[1]]])
        x = np.array([arr[arr.dtype.names[0]]])
    elif len(arr.dtype.names) == 1:
        # Nur 1 Spalte (Flux)
        y = np.asarray(arr[arr.dtype.names[0]], dtype=float)
        x = np.arange(y.size, dtype=float)
    else:
        # 2 Spalten (x, y)
        x = np.asarray(arr[arr.dtype.names[0]], dtype=float)
        y = np.asarray(arr[arr.dtype.names[1]], dtype=float)
    base = os.path.splitext(os.path.basename(path))[0]
    return x, y, base

# Subpixel-Refinement


def gauss_lin(x, a0, a1, A, mu, sig):
    return a0 + a1*(x-mu) + A*np.exp(-0.5*((x-mu)/np.maximum(sig, 1e-6))**2)

def refine_position(x, y, x_click: float, half_width: int = 8):
    n = len(y)
    i0 = int(round(x_click))
    lo = max(0, i0 - half_width)
    hi = min(n, i0 + half_width + 1)
    xw = x[lo:hi]
    yw = y[lo:hi]
    if len(xw) < 5:
        return float(x_click), np.nan, np.nan, False
    med = np.median(yw)
    ymin, ymax = np.min(yw), np.max(yw)
    cmin = med - ymin
    cmax = ymax - med
    if cmin > 0.5 * cmax:
        i_best = np.argmin(yw)
        A0 = -cmin
    else:
        i_best = np.argmax(yw)
        A0 = cmax
    mu_0 = xw[i_best]
    a0_0 = float(med)
    a1_0 = 0.0
    sig_0 = 2.0
    p0 = [a0_0, a1_0, A0, mu_0, sig_0]
    bounds = ([-np.inf, -np.inf, -10*abs(A0) or -1.0, xw[0], 0.3],
              [ np.inf,  np.inf,  10*abs(A0) or  1.0, xw[-1], 20.0])
    try:
        popt, _ = curve_fit(gauss_lin, xw, yw, p0=p0, bounds=bounds, maxfev=5000)
        a0, a1, A, mu, sig = popt
        return float(mu), float(abs(sig)), float(A), True
    except Exception:
        return mu_0, np.nan, np.nan, False


# Interaktive Punktewahl


def get_toolbar_mode(fig):
    mode = ''
    tb = getattr(fig.canvas.manager, 'toolbar', None)
    if tb is not None and hasattr(tb, 'mode'):
        mode = tb.mode or ''
    return mode

def pick_points_on_1d(x, y, degree: int):
    need_min = 2 if degree == 1 else 3
    xs_raw, xs_ref, xs_sig, lambdas = [], [], [], []
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, lw=1)
    ax.grid(True)
    ax.set_title("SPACE: Punkt setzen | ENTER: fertig | U: Undo | Pan/Zoom über Toolbar")
    plt.tight_layout()
    armed = {'val': False}
    vlines = []

    def update_status():
        ax.set_title(f"{len(xs_ref)} Punkte – SPACE→setzen | ENTER→fertig | U→Undo")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == ' ':
            armed['val'] = True
        elif event.key in ('u', 'U', 'backspace'):
            if xs_ref:
                xs_ref.pop(); xs_raw.pop(); xs_sig.pop(); lambdas.pop()
                if vlines:
                    ln = vlines.pop(); ln.remove()
                update_status()
        elif event.key in ('enter', 'return'):
            if len(xs_ref) >= need_min:
                plt.close(fig)
            else:
                print(f"Mind. {need_min} Punkte nötig.")

    def on_click(event):
        if event.inaxes is not ax or event.button != 1 or get_toolbar_mode(fig):
            return
        if not armed['val']:
            return
        armed['val'] = False
        xk = float(event.xdata)
        xs_raw.append(xk)
        print("Welche Linie?")
        for j, (lab, wav) in enumerate(CHOICES, 1):
            print(f"  {j}. {lab}")
        try:
            pick = int(input("Nummer: "))
        except ValueError:
            pick = 1
        pick = max(1, min(pick, len(CHOICES)))
        lambdas.append(float(CHOICES[pick-1][1]))
        mu, sig, amp, ok = refine_position(x, y, xk)
        xs_ref.append(mu)
        xs_sig.append(sig)
        color = 'C3' if ok else 'C1'
        vlines.append(ax.axvline(mu, color=color, ls='--', alpha=0.8))
        update_status()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return np.array(xs_ref), np.array(lambdas), np.array(xs_raw), np.array(xs_sig)


# Kalibrierung & Anwendung


def fit_wavelength(xs, lambdas, degree):
    xs = np.asarray(xs, float)
    lambdas = np.asarray(lambdas, float)
    x_ref = np.mean(xs)
    X = xs - x_ref
    coeffs = np.polyfit(X, lambdas, degree)
    lam_fit = np.polyval(coeffs, X)
    rms = np.sqrt(np.mean((lam_fit - lambdas)**2))
    return coeffs, x_ref, rms

def apply_calibration(n, coeffs, x_ref):
    xfull = np.arange(n, dtype=float)
    lam = np.polyval(coeffs, xfull - x_ref)
    return lam


# Speichern (nur CSV im Input-Ordner)


def save_csv_only(base, lam, y, folder):
    mask = (lam >= 4000) & (lam <= 7000)
    lam_save = lam[mask]
    y_save = y[mask]
    base_clean = os.path.splitext(base)[0]
    csv_path = os.path.join(folder, f"{base_clean}c.csv")
    np.savetxt(csv_path, np.column_stack([lam_save, y_save]),
               delimiter=',', header='lambda_A,flux', comments='', fmt='%.6f')
    print(f"\n CSV gespeichert: {csv_path}")


# Hauptprogramm

def main():
    folder = get_folder_from_txt()
    path = find_csv_files(folder)
    x, y, base = load_1d(path)
    plt.figure(); plt.plot(x, y); plt.title("Unkalibriertes 1D-Spektrum"); plt.show()

    degree = 1 if input("linear oder quadratisch? [l/q]: ").lower() != 'q' else 2
    xs_ref, lambdas, xs_raw, xs_sig = pick_points_on_1d(x, y, degree)
    coeffs, x_ref, rms = fit_wavelength(xs_ref, lambdas, degree)
    lam = apply_calibration(len(y), coeffs, x_ref)

    plt.figure(figsize=(10,5))
    plt.plot(lam, y)
    plt.xlabel("Wellenlänge [A]")
    plt.ylabel("Flux")
    plt.title(f"Kalibriertes Spektrum (RMS={rms:.2f} A)")
    plt.grid(True)
    plt.show()

    save_csv_only(base, lam, y, folder)

if __name__ == "__main__":
    main()
