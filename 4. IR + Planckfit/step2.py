#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCHRITT 2: Berechnung der Instrument Spectral Response (IR)
mit automatischer Belichtungszeit-Auslesung aus FITS-Header.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
import os
import csv

# KONFIGURATION

# Eingabe-Dateiname (atmosphärisch korrigierte Vega-Daten)
INPUT_FILE = r'C:\Users\benne\Desktop\jufo 2025-26\data\Auswertung\Spektralklassen\sa100gersbach\Messungencropped\VegaNeu\251212_17.25_veganeu_atmosphere_corrected.csv'

# Referenzspektrum
REF_FILE = r'C:\Users\benne\Documents\RSpec\ReferenceLibrary\a0v.dat'

# FITS-Header der Originalaufnahme (für EXPTIME, OBJECT usw.)
HEADER_FILE = r'C:\Users\benne\Desktop\jufo 2025-26\data\Auswertung\Spektralklassen\sa100gersbach\Messungencropped\VegaNeu\VegaNeu-1.fits'

# Wellenlängenbereich für IR-Berechnung
LAMBDA_MIN = 3800.0
LAMBDA_MAX = 7000.0

# Polynom-Grade, die getestet werden
POLYNOM_GRADES = [3, 5, 7, 9]

# Extrahiere Ordnerpfad aus INPUT_FILE für Ausgabedateien
folder_path = os.path.dirname(os.path.abspath(INPUT_FILE))

# FITS-HEADER AUSLESEN (Belichtungszeit und Objekt)

def get_exposure_from_header(fits_path):
    """Liest Belichtungszeit (EXPTIME oder EXPOSURE) und Objekt aus FITS-Header."""
    with fits.open(fits_path) as hdul:
        header = hdul[0].header

    exposure = header.get("EXPTIME", header.get("EXPOSURE", None))
    obj = header.get("OBJECT", "Unbekannt")

    if exposure is None:
        print("WARNUNG: Keine Belichtungszeit im Header gefunden!")
        try:
            exposure = float(input("Bitte Belichtungszeit manuell eingeben (Sekunden): "))
        except ValueError:
            print("FEHLER: Ungültige Eingabe.")
            exit(1)

    return exposure, obj


print("="*70)
print("SCHRITT 2: BERECHNUNG DER INSTRUMENTELLEN RESPONSE (IR)")
print("="*70)

try:
    t, object_name = get_exposure_from_header(HEADER_FILE)
    print(f"Header geladen aus: {HEADER_FILE}")
    print(f"  Objekt: {object_name}")
    print(f"  Belichtungszeit: {t:.2f} s")
except Exception as e:
    print(f"FEHLER beim Lesen des Headers: {e}")
    exit(1)

# DATen laden und Ir korrigieren

# Lade atmosphärisch korrigierte Vega-Daten (aus Schritt 1)
try:
    # CSV-Datei mit Headern lesen
    lambda_obs = []
    s_obs_atm_corrected = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header_found = False
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            # Erste Datenzeile nach Headern - prüfe ob es die Spaltenüberschrift ist
            if not header_found and len(row) >= 3:
                if 'lambda' in row[0].lower() or 'wellenlänge' in row[0].lower():
                    header_found = True
                    continue
            # Datenzeilen
            if len(row) >= 3:
                try:
                    lambda_val = float(row[0].replace(',', '.'))
                    s_obs_val = float(row[2].replace(',', '.'))  # Spalte 2 = S_obs_atm_corrected
                    lambda_obs.append(lambda_val)
                    s_obs_atm_corrected.append(s_obs_val)
                except (ValueError, IndexError):
                    continue
    
    lambda_obs = np.array(lambda_obs)
    s_obs_atm_corrected = np.array(s_obs_atm_corrected)
    
    if len(lambda_obs) == 0:
        raise ValueError("Keine gültigen Daten in CSV-Datei gefunden!")
    
    print(f"\nAtmosphärisch korrigierte Daten geladen: {len(lambda_obs)} Punkte")
    print(f"Wellenlängenbereich: {lambda_obs.min():.2f} - {lambda_obs.max():.2f} Å")
    
except FileNotFoundError:
    print(f"\nFEHLER: Atmosphärisch korrigierte Datei nicht gefunden: {INPUT_FILE}")
    exit(1)
except Exception as e:
    print(f"\nFEHLER beim Lesen der CSV-Datei: {e}")
    exit(1)

# Lade Referenzspektrum
try:
    ref_data = []
    with open(REF_FILE, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    lambda_ref = float(parts[0])
                    s_true = float(parts[1])
                    ref_data.append([lambda_ref, s_true])
                except ValueError:
                    continue
    
    ref_data = np.array(ref_data)
    lambda_ref = ref_data[:, 0]
    s_true_ref = ref_data[:, 1]
    
    print(f"Referenzspektrum geladen: {len(lambda_ref)} Punkte")
    print(f"Wellenlängenbereich: {lambda_ref.min():.2f} - {lambda_ref.max():.2f} Å")
    
except FileNotFoundError:
    print("\nFEHLER: REF_FILE nicht gefunden!")
    exit(1)

# Interpoliere S_true auf die Wellenlängen von S_obs
interp_func = interp1d(lambda_ref, s_true_ref, kind='linear', 
                       bounds_error=False, fill_value=0)
s_true_interp = interp_func(lambda_obs)

# Filtere gültige Datenpunkte
mask = ((lambda_obs >= LAMBDA_MIN) & (lambda_obs <= LAMBDA_MAX) & 
        (s_true_interp > 0) & (s_obs_atm_corrected > 0) & np.isfinite(s_true_interp))
lambda_valid = lambda_obs[mask]
s_obs_valid = s_obs_atm_corrected[mask]
s_true_valid = s_true_interp[mask]

print(f"\nWellenlängenfilter: {LAMBDA_MIN} - {LAMBDA_MAX} Å")
print(f"Gültige Datenpunkte für IR-Berechnung: {len(lambda_valid)}")

if len(lambda_valid) == 0:
    print("\nFEHLER: Keine gültigen Datenpunkte im Wellenlängenbereich!")
    exit(1)

# Berechne IR: IR = S_obs_atm_corrected / (K * t * S_true)
ir_raw = s_obs_valid / (t * s_true_valid)

# Normiere IR so dass Maximum = 1 (bestimmt K implizit)
K = np.max(s_obs_valid / (t * s_true_valid))
ir_normalized = ir_raw / K

print(f"Normierungskonstante K: {K:.6e}")


# POLYNOM-FIT

print(f"\n{'='*70}")
print("POLYNOM-FIT DER INSTRUMENTELLEN RESPONSE")
print(f"{'='*70}")

best_degree = None
best_residual = float('inf')
poly_fits = {}

for degree in POLYNOM_GRADES:
    coeffs = np.polyfit(lambda_valid, ir_normalized, degree)
    poly = np.poly1d(coeffs)
    ir_fit = poly(lambda_valid)
    residual = np.sum((ir_normalized - ir_fit)**2)
    poly_fits[degree] = (coeffs, poly, residual)
    
    print(f"\nPolynom Grad {degree}:")
    print(f"  Residuum: {residual:.6e}")
    
    if residual < best_residual:
        best_residual = residual
        best_degree = degree

# Wähle besten Fit
fit_degree = best_degree
coeffs, poly, residual = poly_fits[fit_degree]

print(f"\n{'='*60}")
print(f"Gewählter Polynom-Grad: {fit_degree}")
print(f"Polynom-Koeffizienten (absteigend von x^{fit_degree} bis x^0):")
for i, c in enumerate(coeffs):
    power = fit_degree - i
    print(f"  a_{power}: {c:.10e}")
print(f"{'='*60}")

# Speichere IR-Parameter im selben Ordner wie die Eingabedatei
ir_params_path = os.path.join(folder_path, 'ir_polynomial_parameters.txt')
with open(ir_params_path, 'w') as f:
    f.write(f"Instrument Spectral Response - Polynom-Fit\n")
    f.write(f"{'='*60}\n")
    f.write(f"WICHTIG: Basiert auf atmosphärisch korrigierten Daten!\n")
    f.write(f"{'='*60}\n")
    f.write(f"Objekt: {object_name}\n")
    f.write(f"Belichtungszeit t: {t} s\n")
    f.write(f"Normierungskonstante K: {K:.10e}\n")
    f.write(f"Wellenlängenbereich: {LAMBDA_MIN} - {LAMBDA_MAX} Å\n")
    f.write(f"Anzahl Datenpunkte: {len(lambda_valid)}\n")
    f.write(f"Polynom-Grad: {fit_degree}\n")
    f.write(f"Residuum: {residual:.10e}\n\n")
    f.write(f"Polynom-Koeffizienten (absteigend):\n")
    for i, c in enumerate(coeffs):
        power = fit_degree - i
        f.write(f"a_{power} = {c:.15e}\n")

print(f"\nParameter gespeichert in: {ir_params_path}")

# KALIBRIERUNG UND AUSGABE

# Kalibriere das atmosphärisch korrigierte Spektrum
ir_fit_values = poly(lambda_valid)
s_obs_calibrated = s_obs_valid / (K * t * ir_fit_values)

# Berechne Abweichungen
absolute_deviation = s_obs_calibrated - s_true_valid
relative_deviation = (s_obs_calibrated - s_true_valid) / s_true_valid * 100

# Speichere kalibrierte Daten im selben Ordner
calibrated_spectrum_path = os.path.join(folder_path, 'vega_calibrated_spectrum.dat')
with open(calibrated_spectrum_path, 'w') as f:
    f.write("# Kalibriertes Vega-Spektrum (atmosphärisch korrigiert + IR-kalibriert)\n")
    f.write(f"# Objekt: {object_name}\n")
    f.write(f"# Belichtungszeit: {t} s\n")
    f.write(f"# Normierungskonstante K: {K:.10e}\n")
    f.write("# Wellenlänge[Å]  S_obs_atm_corr  S_true(Referenz)  S_calibrated  Abs_Abw  Rel_Abw[%]\n")
    for i in range(len(lambda_valid)):
        f.write(f"{lambda_valid[i]:10.3f}  {s_obs_valid[i]:12.6e}  {s_true_valid[i]:12.6e}  "
               f"{s_obs_calibrated[i]:12.6e}  {absolute_deviation[i]:12.6e}  "
               f"{relative_deviation[i]:10.4f}\n")

print(f"Kalibriertes Spektrum gespeichert in: {calibrated_spectrum_path}")


# VISUALISIERUNG

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: IR(λ) und Polynom-Fit
ax1 = axes[0, 0]
ax1.plot(lambda_valid, ir_normalized, 'ko', markersize=2, alpha=0.5, label='IR(λ) berechnet')
lambda_smooth = np.linspace(lambda_valid.min(), lambda_valid.max(), 500)
ir_smooth = poly(lambda_smooth)
ax1.plot(lambda_smooth, ir_smooth, 'r-', linewidth=2, label=f'Polynom-Fit (Grad {fit_degree})')
ax1.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax1.set_ylabel('IR(λ) [normiert]', fontsize=11)
ax1.set_title('Instrumentelle Response', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Residuen des Fits
ax2 = axes[0, 1]
residuals_plot = ir_normalized - poly(lambda_valid)
ax2.plot(lambda_valid, residuals_plot, 'go', markersize=2, alpha=0.6)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax2.set_ylabel('Residuum (IR - Fit)', fontsize=11)
ax2.set_title('Residuen des Polynom-Fits', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Kalibriertes Spektrum vs Referenz
ax3 = axes[1, 0]
ax3.plot(lambda_valid, s_obs_calibrated, 'b-', linewidth=1.5, alpha=0.7, 
        label='S_calibrated')
ax3.plot(lambda_valid, s_true_valid, 'r-', linewidth=1.5, alpha=0.7, 
        label='S_true (Referenz)')
ax3.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax3.set_ylabel('Flux', fontsize=11)
ax3.set_title('Kalibriertes Spektrum vs. Referenz', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Abweichungen
ax4 = axes[1, 1]
ax4.plot(lambda_valid, relative_deviation, 'm-', linewidth=1, alpha=0.7)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax4.fill_between(lambda_valid, 0, relative_deviation, alpha=0.3, color='m')
ax4.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax4.set_ylabel('Relative Abweichung [%]', fontsize=11)
ax4.set_title('Kalibrationsgüte', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

mean_rel_dev = np.mean(np.abs(relative_deviation))
std_rel_dev = np.std(relative_deviation)
ax4.text(0.02, 0.98, f'Mittl. Abw.: {mean_rel_dev:.2f}%\nStd. Abw.: {std_rel_dev:.2f}%',
        transform=ax4.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=9)

plt.tight_layout()
plot_path = os.path.join(folder_path, 'step2_ir_calculation.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Grafik gespeichert als: {plot_path}")

# STATISTIK

print(f"\n{'='*70}")
print("STATISTISCHE ZUSAMMENFASSUNG")
print(f"{'='*70}")
print(f"Objekt: {object_name}")
print(f"Belichtungszeit: {t:.2f} s")
print(f"Anzahl Datenpunkte: {len(lambda_valid)}")
print(f"Wellenlängenbereich: {lambda_valid.min():.2f} - {lambda_valid.max():.2f} Å")
print(f"\nAbsolute Abweichungen:")
print(f"  Mittelwert: {np.mean(np.abs(absolute_deviation)):.6e}")
print(f"  Standardabweichung: {np.std(absolute_deviation):.6e}")
print(f"\nRelative Abweichungen:")
print(f"  Mittelwert: {mean_rel_dev:.4f}%")
print(f"  Standardabweichung: {std_rel_dev:.4f}%")
print(f"  Maximum: {np.max(np.abs(relative_deviation)):.4f}%")
print(f"{'='*70}")

print("\n" + "="*70)
print("[OK] SCHRITT 2 ABGESCHLOSSEN!")
print("="*70)
print("\nErstellt wurden:")
print(f"  1. {ir_params_path} - IR-Parameter")
print(f"  2. {calibrated_spectrum_path} - Kalibriertes Spektrum")
print(f"  3. {plot_path} - Visualisierung")
print("\nDie IR kann nun auf andere Objekte angewendet werden!")
print("="*70)
