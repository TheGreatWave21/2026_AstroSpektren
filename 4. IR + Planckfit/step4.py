import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import h, c, k
from scipy.signal import find_peaks
import os
import csv

"""
SCHRITT 4: Schwarzkörper-Fit zur Temperaturbestimmung

Fittet das kalibrierte und normierte Spektrum an die Plancksche
Strahlungsformel zur Bestimmung der effektiven Temperatur (was wir als die T_eff interpretieren)!!!!! bitte beachten btw
"""
# KONFIGURATION


# Ordnerpfad aus ordner.txt auslesen
with open(r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt", "r") as f:
    folder_path = f.read().strip()


# Alle Dateien im Ordner auflisten
all_files = os.listdir(folder_path)
print("Alle Dateien im Ordner:", all_files)  

# INPUT_FILE auswählen (endet mit _IRC.csv)

input_files = [f for f in all_files if f.endswith("_IRC.csv")]

if not input_files:
    raise FileNotFoundError("Keine passende CSV-Datei gefunden! (Suche nach *_IRC.csv)")

INPUT_FILE = os.path.join(folder_path, input_files[0])
print("Gefundene CSV-Datei:", INPUT_FILE)

LAMBDA_MIN = 4000.0  # Å
LAMBDA_MAX = 7000.0  # Å

# Startwert für Temperatur
T_INITIAL = 9000.0  # Kelvin nicht C oder F

# Parameter für Absorptionslinien-Erkennung
ABSORPTION_THRESHOLD = 0.3   
MASK_WIDTH = 0 # ±Datenpunkte

# Manuelle Absorptionslinien (optional)
MANUAL_ABSORPTION_LINES = [(6560.0, 10.0)]  # (Wellenlänge Å, Breite Å)




def planck_law(wavelength_angstrom, temperature, scaling_factor):
    """Plancksches Strahlungsgesetz"""
    wavelength_m = wavelength_angstrom * 1e-10
    numerator = 2.0 * h * c**2 / wavelength_m**5
    exponent = h * c / (wavelength_m * k * temperature)
    exponent = np.clip(exponent, 0, 700)
    denominator = np.exp(exponent) - 1.0
    denominator = np.where(denominator > 0, denominator, 1e-100)
    intensity = numerator / denominator
    return scaling_factor * intensity


def wien_law_peak(temperature):
    """Wiensches Verschiebungsgesetz"""
    b = 2.897771955e-3
    lambda_max_m = b / temperature
    lambda_max_angstrom = lambda_max_m * 1e10
    return lambda_max_angstrom


def detect_absorption_lines(wavelengths, flux, threshold=0.15, width=10):
    """Erkennt Absorptionslinien im Spektrum"""
    from scipy.ndimage import gaussian_filter1d, median_filter
    
    flux_smooth = gaussian_filter1d(flux, sigma=3)
    continuum = median_filter(flux_smooth, size=51)
    residuals = (flux - continuum) / continuum
    
    peaks, properties = find_peaks(-residuals, 
                                   height=threshold,
                                   distance=5,
                                   prominence=0.05)
    
    mask = np.ones(len(wavelengths), dtype=bool)
    
    for peak_idx in peaks:
        start_idx = max(0, peak_idx - width)
        end_idx = min(len(wavelengths), peak_idx + width + 1)
        mask[start_idx:end_idx] = False
    
    return mask, peaks, continuum, residuals


def apply_manual_masks(wavelengths, mask, manual_lines):
    """Wendet manuelle Masken an"""
    for lambda_center, width in manual_lines:
        line_mask = np.abs(wavelengths - lambda_center) <= width / 2.0
        mask[line_mask] = False
    return mask


def get_spectral_lines_database():
    """
    Datenbank bekannter Absorptions- und Emissionslinien
    
    Returns:
        Dictionary mit Linien-Informationen
    """
    
    lines = {
        # Balmer-Serie (Wasserstoff)
        6562.8: {"name": "H-alpha", "element": "H I", "series": "Balmer", "type": "absorption"},
        4861.3: {"name": "H-beta", "element": "H I", "series": "Balmer", "type": "absorption"},
        4340.5: {"name": "H-gamma", "element": "H I", "series": "Balmer", "type": "absorption"},
        4101.7: {"name": "H-delta", "element": "H I", "series": "Balmer", "type": "absorption"},
        3970.1: {"name": "H-epsilon", "element": "H I", "series": "Balmer", "type": "absorption"},
        3889.0: {"name": "H-zeta", "element": "H I", "series": "Balmer", "type": "absorption"},
        3835.4: {"name": "H-eta", "element": "H I", "series": "Balmer", "type": "absorption"},
        
        # Paschen-Serie (Wasserstoff, IR)
        8750.5: {"name": "Pa-gamma", "element": "H I", "series": "Paschen", "type": "absorption"},
        8598.4: {"name": "Pa-delta", "element": "H I", "series": "Paschen", "type": "absorption"},
        
        # Helium
        5875.6: {"name": "He I", "element": "He I", "series": "-", "type": "absorption"},
        4471.5: {"name": "He I", "element": "He I", "series": "-", "type": "absorption"},
        4026.2: {"name": "He I", "element": "He I", "series": "-", "type": "absorption"},
        
        # Metalle (wichtig für Spektralklassifikation)
        3933.7: {"name": "Ca II K", "element": "Ca II", "series": "K-line", "type": "absorption"},
        3968.5: {"name": "Ca II H", "element": "Ca II", "series": "H-line", "type": "absorption"},
        
        # Magnesium
        5183.6: {"name": "Mg I b", "element": "Mg I", "series": "b-triplet", "type": "absorption"},
        5172.7: {"name": "Mg I b", "element": "Mg I", "series": "b-triplet", "type": "absorption"},
        5167.3: {"name": "Mg I b", "element": "Mg I", "series": "b-triplet", "type": "absorption"},
        
        # Natrium D-Linien
        5895.9: {"name": "Na I D2", "element": "Na I", "series": "D-doublet", "type": "absorption"},
        5889.9: {"name": "Na I D1", "element": "Na I", "series": "D-doublet", "type": "absorption"},
        
        # Eisen (wichtig für Sternklassifikation)
        4383.5: {"name": "Fe I", "element": "Fe I", "series": "-", "type": "absorption"},
        4325.8: {"name": "Fe I", "element": "Fe I", "series": "-", "type": "absorption"},
        4271.8: {"name": "Fe I", "element": "Fe I", "series": "-", "type": "absorption"},
        
        # Sauerstoff (atmosphärisch, tellurich)
        6300.3: {"name": "O I (telluric)", "element": "O I", "series": "-", "type": "telluric"},
        6363.8: {"name": "O I (telluric)", "element": "O I", "series": "-", "type": "telluric"},
        7594.0: {"name": "O2 A-band (telluric)", "element": "O2", "series": "A-band", "type": "telluric"},
        6867.0: {"name": "O2 B-band (telluric)", "element": "O2", "series": "B-band", "type": "telluric"},
        6883.0: {"name": "O2 B-band (telluric)", "element": "O2", "series": "B-band", "type": "telluric"},
        6899.0: {"name": "O2 B-band (telluric)", "element": "O2", "series": "B-band", "type": "telluric"},
        6278.0: {"name": "O2 (telluric)", "element": "O2", "series": "-", "type": "telluric"},
        
        # Wasser (atmosphärisch, tellurich)
        5900.0: {"name": "H2O (telluric)", "element": "H2O", "series": "-", "type": "telluric"},
        7200.0: {"name": "H2O (telluric)", "element": "H2O", "series": "-", "type": "telluric"},
    }
    
    return lines


def identify_spectral_line(wavelength, tolerance=15.0):
    """
    Identifiziert eine Spektrallinie basierend auf der Wellenlänge
    
    Parameters:
        wavelength: Gemessene Wellenlänge [Å]
        tolerance: Toleranz für Zuordnung [Å] (Standard: 15 Å)
    
    Returns:
        Dictionary mit Linien-Info oder None
    """
    
    lines_db = get_spectral_lines_database()
    
    # Suche nächste Linie in Datenbank
    best_match = None
    best_diff = float('inf')
    
    for line_wavelength, line_info in lines_db.items():
        diff = abs(wavelength - line_wavelength)
        if diff < tolerance and diff < best_diff:
            best_diff = diff
            best_match = {
                'wavelength_ref': line_wavelength,
                'wavelength_obs': wavelength,
                'delta': diff,
                **line_info
            }
    
    return best_match



# main


print("="*70)
print("SCHRITT 4: SCHWARZKÖRPER-FIT (MIT ABSORPTIONSLINIEN-MASKIERUNG)")
print("="*70)

# Lade kalibriertes Spektrum (CSV-Format)
try:
    wavelengths_all = []
    flux_normalized_all = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header_found = False
        has_normalized = False
        
        for row in reader:
            if not row or row[0].startswith('#'):
                # Prüfe ob Flux_normalized vorhanden ist
                if row and 'flux_normalized' in row[0].lower():
                    has_normalized = True
                continue
            # Erste Datenzeile nach Headern - prüfe ob es die Spaltenüberschrift ist
            if not header_found and len(row) >= 2:
                if 'lambda' in row[0].lower() or 'wellenlänge' in row[0].lower():
                    header_found = True
                    # Prüfe ob Flux_normalized vorhanden ist
                    if len(row) >= 4 and 'flux_normalized' in row[3].lower():
                        has_normalized = True
                    continue
            # Datenzeilen
            if len(row) >= 2:
                try:
                    lambda_val = float(row[0].replace(',', '.'))
                    if has_normalized and len(row) >= 4:
                        # CSV hat Flux_normalized in Spalte 3 (Index 3)
                        flux_val = float(row[3].replace(',', '.'))
                    elif len(row) >= 3:
                        # CSV hat Flux_calibrated in Spalte 2 (Index 2)
                        flux_val = float(row[2].replace(',', '.'))
                    else:
                        # Fallback: Spalte 1
                        flux_val = float(row[1].replace(',', '.'))
                    wavelengths_all.append(lambda_val)
                    flux_normalized_all.append(flux_val)
                except (ValueError, IndexError):
                    continue
    
    wavelengths_all = np.array(wavelengths_all)
    flux_normalized_all = np.array(flux_normalized_all)
    
    if len(wavelengths_all) == 0:
        raise ValueError("Keine gültigen Daten in CSV-Datei gefunden!")
    
    # Falls keine normalisierten Daten vorhanden, normiere bei 550nm
    if not has_normalized:
        idx_550nm = np.argmin(np.abs(wavelengths_all - 5500.0))
        normalization_value = flux_normalized_all[idx_550nm]
        flux_normalized_all = flux_normalized_all / normalization_value
        print(f"\nKalibriertes Spektrum geladen und bei 550nm normiert")
    else:
        print(f"\nKalibriertes & normiertes Spektrum geladen")
    
    print(f"Datenpunkte: {len(wavelengths_all)}")
    print(f"Wellenlängenbereich: {wavelengths_all.min():.2f} - {wavelengths_all.max():.2f} Å")
    
except FileNotFoundError:
    print(f"\nFEHLER: Datei '{INPUT_FILE}' nicht gefunden!")
    print("Bitte führen Sie zuerst 'step3.py' aus.")
    exit(1)
except Exception as e:
    print(f"\nFEHLER beim Lesen der CSV-Datei: {e}")
    exit(1)

# Filtere Fit-Bereich
range_mask = (wavelengths_all >= LAMBDA_MIN) & (wavelengths_all <= LAMBDA_MAX)
wavelengths = wavelengths_all[range_mask]
flux_normalized = flux_normalized_all[range_mask]

print(f"\nFit-Bereich: {LAMBDA_MIN:.0f} - {LAMBDA_MAX:.0f} Å")
print(f"Datenpunkte im Bereich: {len(wavelengths)}")

# Erkenne Absorptionslinien
print(f"\n{'='*70}")
print("ABSORPTIONSLINIEN-ERKENNUNG")
print(f"{'='*70}")
print(f"Schwellwert: {ABSORPTION_THRESHOLD*100:.1f}%")
print(f"Maskenbreite: ±{MASK_WIDTH} Datenpunkte")

absorption_mask, absorption_peaks, continuum, residuals = detect_absorption_lines(
    wavelengths, flux_normalized, 
    threshold=ABSORPTION_THRESHOLD, 
    width=MASK_WIDTH
)

print(f"\nAutomatisch erkannte Absorptionslinien: {len(absorption_peaks)}")
if len(absorption_peaks) > 0:
    print("Wellenlängen der erkannten Linien:")
    print("(Hinweis: Verschiebungen können durch Doppler-Effekt, Instrumentenkalibrierung,")
    print(" atmosphärische Effekte oder Radialgeschwindigkeit verursacht werden)")
    print()
    identified_lines = []
    for i, peak_idx in enumerate(absorption_peaks):
        lambda_peak = wavelengths[peak_idx]
        depth = -residuals[peak_idx]*100
        
        # Versuche Linie zu identifizieren
        line_id = identify_spectral_line(lambda_peak, tolerance=15.0)
        
        if line_id:
            print(f"  {i+1}. λ = {lambda_peak:.2f} Å → {line_id['name']} ({line_id['element']}) "
                  f"[λ_ref = {line_id['wavelength_ref']:.1f} Å, Δλ = {line_id['delta']:.1f} Å] "
                  f"(Tiefe: {depth:.1f}%)")
            identified_lines.append(line_id)
        else:
            print(f"  {i+1}. λ = {lambda_peak:.2f} Å → Unbekannte Linie (Tiefe: {depth:.1f}%)")
    
    # Zusammenfassung nach Elementen
    if identified_lines:
        print("\nZusammenfassung gefundener Elemente:")
        elements = {}
        for line in identified_lines:
            element = line['element']
            if element not in elements:
                elements[element] = []
            elements[element].append(line['name'])
        
        for element, lines_list in sorted(elements.items()):
            print(f"  {element}: {', '.join(lines_list)}")


# Manuelle Masken
if len(MANUAL_ABSORPTION_LINES) > 0:
    print(f"\nManuelle Absorptionslinien: {len(MANUAL_ABSORPTION_LINES)}")
    absorption_mask = apply_manual_masks(wavelengths, absorption_mask, MANUAL_ABSORPTION_LINES)

# Statistik
n_total = len(wavelengths)
n_masked = np.sum(~absorption_mask)
n_used = np.sum(absorption_mask)

print(f"\n{'='*70}")
print(f"Datenpunkte gesamt: {n_total}")
print(f"Datenpunkte maskiert: {n_masked} ({n_masked/n_total*100:.1f}%)")
print(f"Datenpunkte für Fit: {n_used} ({n_used/n_total*100:.1f}%)")
print(f"{'='*70}")

if n_used < 20:
    print("\nWARNUNG: Sehr wenige Datenpunkte für den Fit!")

# Extrahiere Fit-Daten
wavelengths_fit = wavelengths[absorption_mask]
flux_fit = flux_normalized[absorption_mask]

# Initialisierung
scaling_initial = 1.0 / planck_law(5500.0, T_INITIAL, 1.0)

print(f"\nStartwerte für Fit:")
print(f"  Temperatur: {T_INITIAL:.0f} K")

# Fit
try:
    popt, pcov = curve_fit(
        planck_law, 
        wavelengths_fit, 
        flux_fit,
        p0=[T_INITIAL, scaling_initial],
        bounds=([3000.0, 0], [50000.0, np.inf]),
        maxfev=10000
    )
    
    T_fit = popt[0]
    scaling_fit = popt[1]
    perr = np.sqrt(np.diag(pcov))
    T_error = perr[0]
    
    print("\n" + "="*70)
    print("FIT-ERGEBNISSE")
    print("="*70)
    print(f"\nEffektive Temperatur: T_eff = {T_fit:.0f} ± {T_error:.0f} K")
    print(f"Skalierungsfaktor: {scaling_fit:.6e} ± {perr[1]:.6e}")
    
    # Wien
    lambda_max = wien_law_peak(T_fit)
    print(f"\nWiensches Verschiebungsgesetz:")
    print(f"  Wellenlänge des Maximums: λ_max = {lambda_max:.1f} Å")
    
    # Fit-Qualität
    flux_fit_eval = planck_law(wavelengths_fit, T_fit, scaling_fit)
    residuals_fit = flux_fit - flux_fit_eval
    chi_squared = np.sum(residuals_fit**2)
    dof = len(wavelengths_fit) - 2
    chi_squared_reduced = chi_squared / dof
    
    print(f"\nFit-Qualität:")
    print(f"  Chi² (reduziert): {chi_squared_reduced:.6f}")
    print(f"  RMS der Residuen: {np.sqrt(np.mean(residuals_fit**2)):.6f}")
    
    # Klassifikation
    print("\n" + "="*70)
    print("KLASSIFIKATION")
    print("="*70)
    stellar_types = [
        ("O5V", 42000, "blau"),
        ("B0V", 30000, "bläulich-weiß"),
        ("A0V", 9600, "weiß (wie Vega)"),
        ("F0V", 7200, "gelb-weiß"),
        ("G0V", 6000, "gelb"),
        ("G2V", 5778, "gelb (wie Sonne)"),
        ("K0V", 5250, "orange"),
        ("M0V", 3800, "rot"),
    ]
    
    closest_type = min(stellar_types, key=lambda x: abs(x[1] - T_fit))
    print(f"\nDas Objekt entspricht am ehesten einem {closest_type[0]} Stern")
    print(f"(T_eff ≈ {closest_type[1]} K, Farbe: {closest_type[2]})")
    
    relative_luminosity = (T_fit / 5778)**4
    print(f"\nRelative Strahlungsleistung (bezogen auf Sonne):")
    print(f"  L/L_☉ ∝ {relative_luminosity:.2f} (bei gleicher Oberfläche)")
    
    # Speichere Ergebnisse im selben Ordner
    results_file = os.path.join(folder_path, 'step4_blackbody_fit_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Schwarzkörper-Fit Ergebnisse\n")
        f.write("="*70 + "\n")
        f.write(f"Input: {INPUT_FILE}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Effektive Temperatur: T_eff = {T_fit:.0f} ± {T_error:.0f} K\n")
        f.write(f"Skalierungsfaktor: {scaling_fit:.6e} ± {perr[1]:.6e}\n\n")
        f.write(f"Wellenlänge des Maximums (Wien): λ_max = {lambda_max:.1f} Å\n\n")
        f.write(f"Fit-Bereich: {LAMBDA_MIN:.0f} - {LAMBDA_MAX:.0f} Å\n")
        f.write(f"Datenpunkte gesamt: {n_total}\n")
        f.write(f"Datenpunkte maskiert: {n_masked} ({n_masked/n_total*100:.1f}%)\n")
        f.write(f"Datenpunkte für Fit: {n_used} ({n_used/n_total*100:.1f}%)\n\n")
        f.write(f"Absorptionslinien erkannt: {len(absorption_peaks)}\n")
        if len(absorption_peaks) > 0:
            f.write("Erkannte und identifizierte Absorptionslinien:\n")
            for i, peak_idx in enumerate(absorption_peaks):
                lambda_peak = wavelengths[peak_idx]
                depth = -residuals[peak_idx]*100
                line_id = identify_spectral_line(lambda_peak, tolerance=15.0)
                
                if line_id:
                    f.write(f"  {i+1}. λ_obs = {lambda_peak:.2f} Å → {line_id['name']} ({line_id['element']})\n")
                    f.write(f"      λ_ref = {line_id['wavelength_ref']:.1f} Å, Δλ = {line_id['delta']:.1f} Å, Tiefe = {depth:.1f}%\n")
                else:
                    f.write(f"  {i+1}. λ_obs = {lambda_peak:.2f} Å → Unbekannte Linie (Tiefe = {depth:.1f}%)\n")
            
            # Zusammenfassung Elemente
            f.write("\nGefundene Elemente:\n")
            elements = {}
            for peak_idx in absorption_peaks:
                lambda_peak = wavelengths[peak_idx]
                line_id = identify_spectral_line(lambda_peak, tolerance=15.0)
                if line_id:
                    element = line_id['element']
                    if element not in elements:
                        elements[element] = []
                    elements[element].append(line_id['name'])
            
            for element, lines_list in sorted(elements.items()):
                f.write(f"  {element}: {', '.join(set(lines_list))}\n")
        
        f.write(f"\nFit-Qualität:\n")
        f.write(f"  Chi² (reduziert): {chi_squared_reduced:.6f}\n")
        f.write(f"  RMS: {np.sqrt(np.mean(residuals_fit**2)):.6f}\n\n")
        f.write(f"Klassifikation: {closest_type[0]} Stern\n")
        f.write(f"Referenztemperatur: {closest_type[1]} K\n")
        f.write(f"Farbe: {closest_type[2]}\n")
    
    print(f"\nErgebnisse gespeichert in: {results_file}")
    
    # Visualisierung
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Absorptionslinien
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(wavelengths, flux_normalized, 'b-', linewidth=1, alpha=0.7, label='Spektrum')
    ax1.plot(wavelengths, continuum, 'g--', linewidth=1.5, alpha=0.7, label='Kontinuum')
    ax1.plot(wavelengths[absorption_peaks], flux_normalized[absorption_peaks], 
            'ro', markersize=8, label=f'Absorptionslinien ({len(absorption_peaks)})')
    
    # Beschrifte wichtigste Linien
    for peak_idx in absorption_peaks[:5]:  # Nur erste 5 beschriften
        lambda_peak = wavelengths[peak_idx]
        line_id = identify_spectral_line(lambda_peak, tolerance=15.0)
        if line_id:
            ax1.annotate(line_id['name'], 
                        xy=(lambda_peak, flux_normalized[peak_idx]),
                        xytext=(0, -15), textcoords='offset points',
                        ha='center', fontsize=7, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5))
    
    ax1.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax1.set_ylabel('Normierter Flux', fontsize=11)
    ax1.set_title('Absorptionslinien-Erkennung & Identifikation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maske
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(wavelengths, flux_normalized, 'b-', linewidth=1, alpha=0.3, label='Alle Daten')
    ax2.plot(wavelengths[absorption_mask], flux_normalized[absorption_mask], 
            'g-', linewidth=1.5, alpha=0.8, label=f'Fit-Daten ({n_used})')
    ax2.plot(wavelengths[~absorption_mask], flux_normalized[~absorption_mask], 
            'rx', markersize=4, alpha=0.6, label=f'Maskiert ({n_masked})')
    ax2.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax2.set_ylabel('Normierter Flux', fontsize=11)
    ax2.set_title('Datenpunkte-Maske', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hauptplot
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(wavelengths_all, flux_normalized_all, 'b-', linewidth=0.5, alpha=0.3, 
            label='Vollständiges Spektrum')
    ax3.plot(wavelengths, flux_normalized, 'b-', linewidth=1, alpha=0.7, 
            label='Fit-Bereich')
    ax3.plot(wavelengths[~absorption_mask], flux_normalized[~absorption_mask], 
            'rx', markersize=3, alpha=0.4, label='Maskiert')
    
    lambda_smooth = np.linspace(wavelengths_all.min(), wavelengths_all.max(), 1000)
    flux_planck_smooth = planck_law(lambda_smooth, T_fit, scaling_fit)
    ax3.plot(lambda_smooth, flux_planck_smooth, 'r--', linewidth=2,
            label=f'Planck-Fit (T = {T_fit:.0f} K)')
    
    ax3.axvline(lambda_max, color='g', linestyle=':', linewidth=2, alpha=0.7,
               label=f'λ_max = {lambda_max:.0f} Å')
    
    ax3.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax3.set_ylabel('Normierter Flux (bei 550nm = 1)', fontsize=11)
    ax3.set_title(f'Schwarzkörper-Fit: T_eff = {T_fit:.0f} ± {T_error:.0f} K', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuen
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(wavelengths_fit, residuals_fit, 'go-', markersize=2, linewidth=0.5, alpha=0.6)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax4.fill_between(wavelengths_fit, 0, residuals_fit, alpha=0.3, color='g')
    ax4.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax4.set_ylabel('Residuen', fontsize=11)
    ax4.set_title('Fit-Residuen', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    textstr = f'RMS: {np.sqrt(np.mean(residuals_fit**2)):.4f}\nChi²_red: {chi_squared_reduced:.4f}'
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Log-Skala
    ax5 = plt.subplot(3, 2, 5)
    ax5.semilogy(wavelengths_all, flux_normalized_all, 'b-', linewidth=1.5, alpha=0.7,
                label='Spektrum')
    ax5.semilogy(lambda_smooth, flux_planck_smooth, 'r--', linewidth=2,
                label=f'Planck (T={T_fit:.0f}K)')
    ax5.axvline(lambda_max, color='g', linestyle=':', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax5.set_ylabel('Normierter Flux (log)', fontsize=11)
    ax5.set_title('Logarithmische Darstellung', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: HR-Diagramm
    ax6 = plt.subplot(3, 2, 6)
    hr_temps = np.array([42000, 30000, 9600, 7200, 6000, 5778, 5250, 3800])
    hr_names = ['O5V', 'B0V', 'A0V', 'F0V', 'G0V', 'G2V\n(Sonne)', 'K0V', 'M0V']
    hr_colors_plot = ['#9bb0ff', '#aabfff', '#ffffff', '#f8f7ff', '#fff4ea', 
                     '#ffcc6f', '#ff9966', '#ff6666']
    hr_luminosity = [(T/5778)**4 for T in hr_temps]
    
    ax6.scatter(hr_temps, hr_luminosity, c=hr_colors_plot, s=200, 
              edgecolors='black', linewidths=2, alpha=0.8, zorder=3)
    
    for i, (T, L, name) in enumerate(zip(hr_temps, hr_luminosity, hr_names)):
        ax6.annotate(name, (T, L), xytext=(0, 15), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')
    
    fitted_luminosity = (T_fit/5778)**4
    ax6.scatter([T_fit], [fitted_luminosity], c='red', s=400, marker='*',
              edgecolors='black', linewidths=2, zorder=4,
              label=f'Gemessenes Objekt\n(T={T_fit:.0f}K)')
    
    ax6.set_xlabel('Effektive Temperatur [K]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Relative Leuchtkraft (L/L☉)', fontsize=11, fontweight='bold')
    ax6.set_title('Position im HR-Diagramm', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.invert_xaxis()
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=10, loc='lower left')
    
    plt.tight_layout()
    plot_file = os.path.join(folder_path, 'step4_blackbody_fit.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Grafik gespeichert als: {plot_file}")
    
    print("\n" + "="*70)
    print("[OK] SCHRITT 4 ABGESCHLOSSEN!")
    print("="*70)
    print("\nEffektive Temperatur bestimmt:")
    print(f"  T_eff = {T_fit:.0f} ± {T_error:.0f} K")
    print(f"  Spektraltyp: {closest_type[0]}")
    print("="*70)
    
except RuntimeError as e:
    print(f"\nFEHLER beim Fit: {e}")
    print("\nMögliche Lösungen:")
    print("  - Passen Sie T_INITIAL an")
    print("  - Ändern Sie ABSORPTION_THRESHOLD oder MASK_WIDTH")

