import numpy as np
import matplotlib.pyplot as plt
import os
import csv

"""
SCHRITT 3: Anwendung der Instrumentellen Response auf andere Objekte
S_obs_atm_corrected(λ) = K * t * IR(λ) * S_true(λ)
=> S_true(λ) = S_obs_atm_corrected(λ) / (K * t * IR(λ))
"""

from astropy.io import fits

# KONFIGURATION - Automatische Dateisuche

# Ordnerpfad aus ordner.txt auslesen
with open(r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt", "r") as f:
    folder_path = f.read().strip()


# Alle Dateien im Ordner auflisten
all_files = os.listdir(folder_path)
print("Alle Dateien im Ordner:", all_files)  # zur Kontrolle


# INPUT_SPECTRUM_FILE auswählen (endet mit c.csv)

spectrum_files = [f for f in all_files if f.endswith("c.csv")]

if not spectrum_files:
    raise FileNotFoundError("Keine passende CSV-Datei gefunden! (Suche nach *c.csv)")

INPUT_SPECTRUM_FILE = os.path.join(folder_path, spectrum_files[0])
print("Gefundene CSV-Datei:", INPUT_SPECTRUM_FILE)


# INPUT_FITS_FILE auswählen (endet mit -1.fits)

fits_files = [f for f in all_files if f.endswith("-1.fits")]

if not fits_files:
    raise FileNotFoundError("Keine passende FITS-Datei gefunden! (Suche nach *-1.fits)")

INPUT_FITS_FILE = os.path.join(folder_path, fits_files[0])
print("Gefundene FITS-Datei:", INPUT_FITS_FILE)

# Normierung bei 550nm auf 1.0
NORMALIZE_AT_550NM = True


# Belichtungszeit und Objektname aus FITS auslesen   
try:
    with fits.open(INPUT_FITS_FILE) as hdul:
        header = hdul[0].header
        EXPOSURE_TIME = header.get('EXPTIME')
        OBJECT_NAME = header.get('OBJECT', 'Unknown')
        if EXPOSURE_TIME is None:
            raise KeyError("Belichtungszeit 'EXPTIME' nicht im FITS-Header gefunden!")
    print(f"Belichtungszeit aus FITS ausgelesen: {EXPOSURE_TIME} s")
    print(f"Objektname: {OBJECT_NAME}")
except Exception as e:
    print(f"FEHLER beim Lesen der Belichtungszeit aus FITS: {e}")
    exit(1)


# Ausgabe-Dateiname generieren: (star_name)_IRC

star_name_clean = OBJECT_NAME.lower().replace(" ", "_").replace("/", "_")
OUTPUT_FILE = os.path.join(folder_path, f"{star_name_clean}_IRC.csv")
print(f"Ausgabe-Datei: {OUTPUT_FILE}")
def load_ir_parameters(filename='ir_polynomial_parameters.txt'):
    """
    Lädt die IR-Polynom-Parameter aus Datei
    
    Returns:
        K, t_vega, coeffs, degree
    """
    K = None
    t_vega = None
    degree = None
    coeffs = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Normierungskonstante K:' in line:
                K = float(line.split(':')[1].strip())
            elif 'Belichtungszeit t:' in line:
                t_vega = float(line.split(':')[1].split()[0])
            elif 'Polynom-Grad:' in line:
                degree = int(line.split(':')[1].strip())
            elif line.startswith('a_'):
                # Format: a_5 = 1.234e-10
                value = float(line.split('=')[1].strip())
                coeffs.append(value)
    
    if K is None or degree is None or len(coeffs) == 0:
        raise ValueError("Konnte nicht alle Parameter aus Datei lesen!")
    
    # Koeffizienten sind in absteigender Reihenfolge (a_n, a_n-1, ..., a_0)
    coeffs = np.array(coeffs)
    
    return K, t_vega, coeffs, degree


def apply_ir_calibration(wavelengths, flux_atm_corrected, exposure_time, K, ir_poly):
    """
    Wendet IR-Kalibration an
    
    Parameters:
        wavelengths: Wellenlängen-Array
        flux_atm_corrected: Atmosphärisch korrigierter Flux
        exposure_time: Belichtungszeit der Messung
        K: Normierungskonstante aus IR-Berechnung
        ir_poly: np.poly1d Objekt der IR
    
    Returns:
        flux_calibrated: Kalibrierter Flux
    """
    
    # Berechne IR-Werte für diese Wellenlängen
    ir_values = ir_poly(wavelengths)
    
    # S_true = S_obs_atm_corrected / (K * t * IR)
    flux_calibrated = flux_atm_corrected / (K * exposure_time * ir_values)
    
    return flux_calibrated


# hauptprogramm

print("="*70)
print("SCHRITT 3: ANWENDUNG DER INSTRUMENTELLEN RESPONSE")
print("="*70)

# Lade IR-Parameter aus festem Pfad
ir_params_file = r'C:\Users\benne\Desktop\jufo 2025-26\data\Auswertung\Spektralklassen\sa100gersbach\Messungencropped\VegaNeu\ir_polynomial_parameters.txt'
try:
    K, t_vega, coeffs, degree = load_ir_parameters(ir_params_file)
    ir_poly = np.poly1d(coeffs)
    
    print(f"\nIR-Parameter geladen aus: {ir_params_file}")
    print(f"  Normierungskonstante K: {K:.6e}")
    print(f"  Vega Belichtungszeit: {t_vega} s")
    print(f"  Polynom-Grad: {degree}")
    
except FileNotFoundError:
    print(f"\nFEHLER: 'ir_polynomial_parameters.txt' nicht gefunden: {ir_params_file}")
    print("Bitte überprüfen Sie den Pfad zur IR-Parameter-Datei.")
    exit(1)
except Exception as e:
    print(f"\nFEHLER beim Laden der IR-Parameter: {e}")
    exit(1)

# Lade atmosphärisch korrigiertes Input-Spektrum
print(f"\n{'='*70}")
print(f"LADE INPUT-SPEKTRUM")
print(f"{'='*70}")
print(f"Datei: {INPUT_SPECTRUM_FILE}")
print(f"Belichtungszeit: {EXPOSURE_TIME} s")

try:
    # CSV-Datei mit Headern lesen (aus step1.py Format)
    wavelengths = []
    flux_atm_corrected = []
    is_step1_format = False
    
    with open(INPUT_SPECTRUM_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header_found = False
        for row in reader:
            if not row or row[0].startswith('#'):
                # Prüfe ob es step1 Format ist
                if row and 'S_obs_atm_corrected' in row[0]:
                    is_step1_format = True
                continue
            # Erste Datenzeile nach Headern - prüfe ob es die Spaltenüberschrift ist
            if not header_found and len(row) >= 3:
                if 'lambda' in row[0].lower() or 'wellenlänge' in row[0].lower():
                    header_found = True
                    # Prüfe ob S_obs_atm_corrected vorhanden ist
                    if len(row) >= 3 and 's_obs_atm_corrected' in row[2].lower():
                        is_step1_format = True
                    continue
            # Datenzeilen
            if len(row) >= 2:
                try:
                    lambda_val = float(row[0].replace(',', '.'))
                    if is_step1_format and len(row) >= 3:
                        # step1 Format: Spalte 2 = S_obs_atm_corrected
                        flux_val = float(row[2].replace(',', '.'))
                    else:
                        # Normales Format: Spalte 1 = Flux
                        flux_val = float(row[1].replace(',', '.'))
                    wavelengths.append(lambda_val)
                    flux_atm_corrected.append(flux_val)
                except (ValueError, IndexError):
                    continue
    
    wavelengths = np.array(wavelengths)
    flux_atm_corrected = np.array(flux_atm_corrected)
    
    if len(wavelengths) == 0:
        raise ValueError("Keine gültigen Daten in CSV-Datei gefunden!")
    
    print(f"\nSpektrum geladen: {len(wavelengths)} Datenpunkte")
    print(f"Wellenlängenbereich: {wavelengths.min():.2f} - {wavelengths.max():.2f} Å")
    print(f"Flux-Bereich: {flux_atm_corrected.min():.6e} - {flux_atm_corrected.max():.6e}")
    
    # Filtere ungültige Werte
    mask = (flux_atm_corrected > 0) & np.isfinite(flux_atm_corrected)
    wavelengths = wavelengths[mask]
    flux_atm_corrected = flux_atm_corrected[mask]
    
    print(f"Nach Filterung: {len(wavelengths)} Datenpunkte")
    
except FileNotFoundError:
    print(f"\nFEHLER: Datei '{INPUT_SPECTRUM_FILE}' nicht gefunden!")
    print("\nVerfügbare Dateien sollten sein:")
    print("  - vega_atmosphere_corrected.dat (aus step1)")
    print("  - oder ein anderes atmosphärisch korrigiertes Spektrum")
    exit(1)
except Exception as e:
    print(f"\nFEHLER beim Laden: {e}")
    exit(1)

# Wende IR-Kalibration an
print(f"\n{'='*70}")
print("ANWENDUNG DER INSTRUMENTELLEN RESPONSE")
print(f"{'='*70}")

flux_calibrated = apply_ir_calibration(wavelengths, flux_atm_corrected, 
                                       EXPOSURE_TIME, K, ir_poly)

print(f"Kalibration abgeschlossen")
print(f"Kalibrierter Flux-Bereich: {flux_calibrated.min():.6e} - {flux_calibrated.max():.6e}")

# Normierung bei 550nm (optional)
if NORMALIZE_AT_550NM:
    idx_550nm = np.argmin(np.abs(wavelengths - 5500.0))
    normalization_value = flux_calibrated[idx_550nm]
    flux_normalized = flux_calibrated / normalization_value
    
    print(f"\nNormierung bei 550nm (5500 Å):")
    print(f"  Wellenlänge: {wavelengths[idx_550nm]:.2f} Å")
    print(f"  Normierungswert: {normalization_value:.6e}")
    print(f"  Flux nach Normierung: {flux_normalized[idx_550nm]:.6f}")
else:
    flux_normalized = flux_calibrated
    normalization_value = 1.0

# Speichere kalibriertes Spektrum als CSV
print(f"\n{'='*70}")
print("SPEICHERE KALIBRIERTES SPEKTRUM")
print(f"{'='*70}")

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["# Kalibriertes Spektrum (atmosphärisch korrigiert + IR-kalibriert)"])
    writer.writerow(["# Input", INPUT_SPECTRUM_FILE])
    writer.writerow(["# Belichtungszeit[s]", f"{EXPOSURE_TIME}"])
    writer.writerow(["# Normierungskonstante K", f"{K:.10e}"])
    if NORMALIZE_AT_550NM:
        writer.writerow(["# Normiert bei 550nm auf 1.0"])
        writer.writerow(["# Normierungswert", f"{normalization_value:.10e}"])
    writer.writerow([])
    if NORMALIZE_AT_550NM:
        writer.writerow(["lambda_A", "Flux_atm_corr", "Flux_calibrated", "Flux_normalized"])
        for i in range(len(wavelengths)):
            writer.writerow([
                f"{wavelengths[i]:.3f}",
                f"{flux_atm_corrected[i]:.6e}",
                f"{flux_calibrated[i]:.6e}",
                f"{flux_normalized[i]:.6e}"
            ])
    else:
        writer.writerow(["lambda_A", "Flux_atm_corr", "Flux_calibrated"])
        for i in range(len(wavelengths)):
            writer.writerow([
                f"{wavelengths[i]:.3f}",
                f"{flux_atm_corrected[i]:.6e}",
                f"{flux_calibrated[i]:.6e}"
            ])

print(f"Kalibriertes Spektrum gespeichert in: {OUTPUT_FILE}")

# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Atmosphärisch korrigiertes Spektrum
ax1 = axes[0, 0]
ax1.plot(wavelengths, flux_atm_corrected, 'b-', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax1.set_ylabel('Flux', fontsize=11)
ax1.set_title('Input: Atmosphärisch korrigiertes Spektrum', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: IR-Kurve
ax2 = axes[0, 1]
ir_values = ir_poly(wavelengths)
ax2.plot(wavelengths, ir_values, 'g-', linewidth=2)
ax2.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax2.set_ylabel('IR(λ) [normiert]', fontsize=11)
ax2.set_title('Instrumentelle Response', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Kalibriertes Spektrum
ax3 = axes[1, 0]
if NORMALIZE_AT_550NM:
    ax3.plot(wavelengths, flux_normalized, 'r-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=5500, color='g', linestyle=':', linewidth=1, alpha=0.5, label='550nm')
    ax3.set_ylabel('Normierter Flux (bei 550nm = 1)', fontsize=11)
    ax3.set_title('Output: Kalibriertes & Normiertes Spektrum', fontsize=12, fontweight='bold')
else:
    ax3.plot(wavelengths, flux_calibrated, 'r-', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Flux', fontsize=11)
    ax3.set_title('Output: Kalibriertes Spektrum', fontsize=12, fontweight='bold')

ax3.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Vergleich vorher/nachher (normiert)
ax4 = axes[1, 1]
flux_atm_norm = flux_atm_corrected / np.max(flux_atm_corrected)
flux_cal_norm = flux_calibrated / np.max(flux_calibrated)
ax4.plot(wavelengths, flux_atm_norm, 'b-', linewidth=1.5, alpha=0.6, 
        label='Vor Kalibration (normiert)')
ax4.plot(wavelengths, flux_cal_norm, 'r-', linewidth=1.5, alpha=0.6,
        label='Nach Kalibration (normiert)')
ax4.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
ax4.set_ylabel('Normierter Flux', fontsize=11)
ax4.set_title('Vergleich: Vorher vs. Nachher', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(folder_path, f"{star_name_clean}_IRC.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Grafik gespeichert als: {plot_path}")

# Zusammenfassung
print(f"\n{'='*70}")
print("[OK] SCHRITT 3 ABGESCHLOSSEN!")
print(f"{'='*70}")
print("\nErstellt wurden:")
print(f"  1. {OUTPUT_FILE} - Kalibriertes Spektrum")
print(f"  2. {plot_path} - Visualisierung")
print("\nNächster Schritt:")
print("  -> Führen Sie 'step4_blackbody_fit.py' aus für Temperaturbestimmung")
print(f"{'='*70}")