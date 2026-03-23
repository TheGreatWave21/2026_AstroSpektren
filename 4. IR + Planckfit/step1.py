#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA100 – Atmosphärische Extinktionskorrektur mit FITS-Header-Autokonfiguration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import pytz
import csv

try:
    from astropy.io import fits
except Exception:
    fits = None

# DATEIEN – nur hier anpassen

import os


# Ordnerpfad aus ordner.txt auslesen

with open(r"C:\Users\joche\Desktop\python\Astro\1. stack + align\ordner.txt", "r") as f:
    folder_path = f.read().strip()
# Alle Dateien im Ordner auflisten

all_files = os.listdir(folder_path)
print("Alle Dateien im Ordner:", all_files)  # zur Kontrolle

# INPUT_FILE auswählen (jede Datei, die auf c.csv endet)

input_files = [f for f in all_files if f.endswith("c.csv")]

if not input_files:
    raise FileNotFoundError("Keine passende CSV-Datei gefunden!")

INPUT_FILE = os.path.join(folder_path, input_files[0])
print("Gefundene CSV-Datei:", INPUT_FILE)


# HEADER_FILE auswählen (jede Datei, die auf -1.fits endet)

header_files = [f for f in all_files if f.endswith("-1.fits")]

if not header_files:
    raise FileNotFoundError("Keine passende FITS-Datei gefunden!")

HEADER_FILE = os.path.join(folder_path, header_files[0])
print("Gefundene FITS-Datei:", HEADER_FILE)

EXTINCTION_FILE = None

# Standard-Observatorium (wird überschrieben, falls im Header vorhanden)
OBSERVATORY = "Standard"

LAMBDA_MIN = 3800.0
LAMBDA_MAX = 7000.0

def read_header_info(fits_file):
    """Extrahiert Beobachtungsinformationen aus FITS-Header."""
    if fits is None:
        raise RuntimeError("Astropy erforderlich, um FITS-Header zu lesen.")

    hdul = fits.open(fits_file)
    header = hdul[0].header
    hdul.close()

    info = {}
    
    date_obs = header.get("DATE-OBS", None)
    if date_obs:
        try:
            # ISO-Format z. B. '2025-08-10T20:09:56.856'
            dt_utc = datetime.fromisoformat(date_obs.replace("Z", ""))
            info["datetime_utc"] = dt_utc
        except Exception:
            info["datetime_utc"] = None

    info["object_name"] = header.get("OBJECT", "Unbekannt")


    info["location_name"] = header.get("ORIGIN", "Unbekannt")
    # Hardcoded observatory coordinates: 47.70003° N, 7.94875° E -> if needed please change them since they are included here: see one line below and two lines below:
    info["latitude"] = 47.70003
    info["longitude"] = 7.94875

    # Beobachtungsinstrument -> nicht allzu wichtig aber hat immer noch einfluss
    info["instrument"] = header.get("INSTRUME", "Unbekannt")

    return info


print("="*70)
print("AUTOMATISCHE KONFIGURATION AUS FITS-HEADER")
print("="*70)

try:
    header_info = read_header_info(HEADER_FILE)
except Exception as e:
    print(f"Konnte Header nicht lesen: {e}")
    print("Bitte überprüfe HEADER_FILE oder installiere astropy.")
    exit(1)

# Extrahiere Werte oder setze Defaults
OBJECT_NAME = header_info.get("object_name", "Unbekannt")
LOCATION_NAME = header_info.get("location_name", "Unbekannt")
LATITUDE = header_info.get("latitude", 0.0)
LONGITUDE = header_info.get("longitude", 0.0)

obs_datetime_utc = header_info.get("datetime_utc", None)
if obs_datetime_utc is None:
    print("Keine gültige Beobachtungszeit im Header gefunden!")
    OBSERVATION_DATE = input("Bitte Datum eingeben (YYYY-MM-DD): ")
    OBSERVATION_TIME = input("Bitte Zeit eingeben (HH:MM, Ortszeit): ")
    TIMEZONE = "Europe/Berlin"
    local_tz = pytz.timezone(TIMEZONE)
    obs_datetime_local = local_tz.localize(datetime.strptime(
        f"{OBSERVATION_DATE} {OBSERVATION_TIME}", "%Y-%m-%d %H:%M"))
    obs_datetime_utc = obs_datetime_local.astimezone(pytz.UTC)
else:
    obs_datetime_local = obs_datetime_utc.astimezone(pytz.timezone("Europe/Berlin"))
    OBSERVATION_DATE = obs_datetime_local.strftime("%Y-%m-%d")
    OBSERVATION_TIME = obs_datetime_local.strftime("%H:%M")
    TIMEZONE = "Europe/Berlin"

print(f"\nHeaderdaten erfolgreich geladen aus: {HEADER_FILE}")
print(f"  Objekt: {OBJECT_NAME}")
print(f"  Beobachtungsdatum (UTC): {obs_datetime_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Standort: {LOCATION_NAME}")
print(f"  Koordinaten: {LATITUDE:.4f}°N, {LONGITUDE:.4f}°E")
print(f"  Instrument: {header_info.get('instrument', 'Unbekannt')}")
print("="*70 + "\n")

# Danach läuft dein Originalcode unverändert weiter ↓
# (ab: „def get_standard_extinction(...):“)


def get_standard_extinction(observatory='Standard'):
    """
    Gibt Standard-Extinktionskurven für verschiedene Observatorien zurück
    """
    
    if observatory == 'CTIO':
        wavelengths = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000])
        k_ext = np.array([0.60, 0.45, 0.30, 0.22, 0.17, 0.13, 0.10, 0.08, 0.06, 0.04, 0.03])
        
    elif observatory == 'LaSilla':
        wavelengths = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000])
        k_ext = np.array([0.55, 0.40, 0.28, 0.20, 0.15, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03])
        
    elif observatory == 'Paranal':
        wavelengths = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000])
        k_ext = np.array([0.50, 0.38, 0.25, 0.18, 0.13, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02])
        
    elif observatory == 'MaunaKea':
        wavelengths = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000])
        k_ext = np.array([0.45, 0.35, 0.23, 0.17, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02])
        
    else:  # Standard
        wavelengths = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000])
        k_ext = np.array([0.55, 0.42, 0.28, 0.20, 0.15, 0.12, 0.09, 0.07, 0.06, 0.04, 0.03])
    
    return wavelengths, k_ext


def load_extinction_curve(filename):
    """Lädt Extinktionskurve aus Datei"""
    data = np.loadtxt(filename)
    wavelengths = data[:, 0]
    k_ext = data[:, 1]
    return wavelengths, k_ext


def apply_atmospheric_correction(wavelengths, flux, airmass, k_ext_wavelengths, k_ext_values):
    """
    Wendet atmosphärische Extinktionskorrektur an
    """
    
    # Interpoliere Extinktionskoeffizienten
    interp_func = interp1d(k_ext_wavelengths, k_ext_values, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
    k_ext_interp = interp_func(wavelengths)
    
    # Berechne Korrekturfaktor: S_true = S_obs * 10^(+0.4 * k * X)
    extinction_factor = 10**(0.4 * k_ext_interp * airmass)
    
    # Korrigiere Flux
    flux_corrected = flux * extinction_factor
    
    return flux_corrected, extinction_factor


def calculate_object_position(latitude, longitude, obs_datetime_utc, 
                              ra_hours, dec_degrees):
    """
    Berechnet Altitude und Airmass für ein astronomisches Objekt
    
    Parameters:
        latitude: Beobachterbreite [Grad]
        longitude: Beobachterlänge [Grad]
        obs_datetime_utc: Beobachtungszeit (UTC, datetime-Objekt)
        ra_hours: Rektaszension [Stunden]
        dec_degrees: Deklination [Grad]
    
    Returns:
        altitude, azimuth, airmass
    """
    from datetime import datetime
    
    # Berechne Julianisches Datum
    year = obs_datetime_utc.year
    month = obs_datetime_utc.month
    day = obs_datetime_utc.day
    hour = obs_datetime_utc.hour
    minute = obs_datetime_utc.minute
    second = obs_datetime_utc.second
    
    if month <= 2:
        year -= 1
        month += 12
    
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    JD += (hour + minute/60.0 + second/3600.0) / 24.0
    
    # Greenwich Sternzeit
    T = (JD - 2451545.0) / 36525.0
    GST = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T**2 - T**3 / 38710000.0
    GST = GST % 360.0
    
    # Lokale Sternzeit
    LST = (GST + longitude) % 360.0
    
    # Stundenwinkel
    RA_deg = ra_hours * 15.0
    hour_angle = LST - RA_deg
    
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(dec_degrees)
    ha_rad = np.radians(hour_angle)
    
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    altitude = np.degrees(np.arcsin(sin_alt))
    
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(np.radians(altitude)))
    azimuth = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    if np.sin(ha_rad) > 0:
        azimuth = 360 - azimuth
    
    zenith_angle = 90.0 - altitude
    

    z_rad = np.radians(zenith_angle)
    airmass = 1.0 / (np.cos(z_rad) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    return altitude, azimuth, airmass, zenith_angle


def get_object_coordinates(object_name):
    """
    Gibt die Koordinaten für bekannte Objekte zurück
    
    Returns:
        ra_hours, dec_degrees
    """
    

    objects = {
        "Vega": (18.6156, 38.7837),
        "VegaNeu": (18.6156, 38.7837),
        "Vega2": (18.6156, 38.7837),
        "Vega_Exposure15_Count40": (18.6156, 38.7837),
        "Sirius": (6.7525, -16.7161),
        "Arcturus": (14.2610, 19.1824),
        "Capella": (5.2781, 45.9980),
        "Rigel": (5.2423, -8.2017),
        "Procyon": (7.6550, 5.2250),
        "Betelgeuse": (5.9195, 7.4070),
        "Altair": (19.8464, 8.8683),
        "Aldebaran": (4.5987, 16.5093),
        "Spica": (13.4199, -11.1614),
        "Antares": (16.4901, -26.4320),
        "Pollux": (7.7553, 28.0262),
        "Fomalhaut": (22.9608, -29.6222),
        "Deneb": (20.6906, 45.2803),
        "Regulus": (10.1395, 11.9672),
        "10lac": (22.5080, 39.0500),
        "Alderamin": (21.3094, 62.5856),
        "alphecca": (15.5781, 26.7147),
        "Alshain": (19.9219, 6.4068),
        "Altair": (19.8464, 8.8683),
        "Arcturus": (14.2610, 19.1824),
        "Deneb": (20.6906, 45.2803),
        "LamCyg": (20.7741, 36.3515),
        "MyCephei": (21.7251, 58.7801),
        "pcygni": (20.2964, 38.0330),
        "Sadr": (20.3705, 40.2567),
        "Sheliak": (18.8347, 33.3606),
        "TCorBor": (15.9592, 25.9204),
        "ThetaCygni": (19.6073, 50.2208),
        "wr140": (20.1542, 43.8711),
        "wr140g": (20.1542, 43.8711),
        "10lac": (22.5081, 39.0531),
        "T Coronae Borealis": (15.9917, 25.9203),
        "TCrB": (15.9917, 25.9203),
        "Bellatrix": (5.4189, 6.3497),
        "Alnitak": (5.6793, -1.9426),
        "Mintaka": (5.5333, -0.2990),
        "Pollux": (7.7553, 28.0260),
        "Procyon": (7.6550, 5.2250),
        "Rigel": (5.2422, -8.2017),
    }
    
    object_name_lower = object_name.lower()
    for name, coords in objects.items():
        if name.lower() == object_name_lower:
            return coords
   
    return None, None


# main

print("="*70)
print("SCHRITT 1: ATMOSPHÄRISCHE KORREKTUR DER ROHDATEN")
print("="*70)

# Parse Datum und Zeit
from datetime import datetime
import pytz

try:
    # Parse Beobachtungszeit
    obs_datetime_str = f"{OBSERVATION_DATE} {OBSERVATION_TIME}"
    obs_datetime_local = datetime.strptime(obs_datetime_str, "%Y-%m-%d %H:%M")
    
    # Konvertiere zu UTC
    local_tz = pytz.timezone(TIMEZONE)
    obs_datetime_local = local_tz.localize(obs_datetime_local)
    obs_datetime_utc = obs_datetime_local.astimezone(pytz.UTC)
    
    print(f"\nBeobachtungsinformationen:")
    print(f"  Datum: {OBSERVATION_DATE}")
    print(f"  Uhrzeit: {OBSERVATION_TIME} ({TIMEZONE})")
    print(f"  UTC: {obs_datetime_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ort: {LOCATION_NAME}")
    print(f"  Koordinaten: {LATITUDE:.4f}°N, {LONGITUDE:.4f}°E")
    print(f"  Objekt: {OBJECT_NAME}")
    
except Exception as e:
    print(f"\nFEHLER beim Parsen von Datum/Zeit: {e}")
    print("Bitte überprüfen Sie das Format:")
    print("  OBSERVATION_DATE: 'YYYY-MM-DD'")
    print("  OBSERVATION_TIME: 'HH:MM'")
    exit(1)

# Berechne Airmass für das Objekt
print(f"\n{'='*70}")
print("BERECHNUNG DER OBJEKT-POSITION UND AIRMASS")
print(f"{'='*70}")

ra_hours, dec_degrees = get_object_coordinates(OBJECT_NAME)

if ra_hours is None or dec_degrees is None:
    print(f"\nWARNUNG: Objekt '{OBJECT_NAME}' nicht in Datenbank gefunden!")
    print("Bitte geben Sie die Koordinaten manuell ein.")
    print("\nBekannte Objekte:")
    print("  Vega, Sirius, Arcturus, Capella, Rigel, Procyon, Betelgeuse,")
    print("  Altair, Aldebaran, Spica, Antares, Pollux, Fomalhaut, Deneb, Regulus")
    
    # Frage nach manuellen Koordinaten
    try:
        ra_hours = float(input("\nRektaszension (Stunden, 0-24): "))
        dec_degrees = float(input("Deklination (Grad, -90 bis +90): "))
    except:
        print("Ungültige Eingabe. Verwende Standard-Airmass = 1.2")
        AIRMASS = 1.2
        altitude = 56.8
        azimuth = 180.0
        zenith_angle = 33.2
        ra_hours = 0.0
        dec_degrees = 0.0
else:
    print(f"Objektkoordinaten (J2000):")
    print(f"  Rektaszension: {ra_hours:.4f}h = {ra_hours*15:.4f}°")
    print(f"  Deklination: {dec_degrees:.4f}°")
    
    # Berechne Position
    altitude, azimuth, AIRMASS, zenith_angle = calculate_object_position(
        LATITUDE, LONGITUDE, obs_datetime_utc, ra_hours, dec_degrees
    )
    
    print(f"\nBerechnete Position:")
    print(f"  Azimut: {azimuth:.2f}°")
    print(f"  Höhenwinkel: {altitude:.2f}°")
    print(f"  Zenitwinkel: {zenith_angle:.2f}°")
    print(f"  Airmass: {AIRMASS:.4f}")
    
    if altitude < 0:
        print(f"\n  WARNUNG: Objekt unter dem Horizont!")
        print(f"    Das Objekt war zum angegebenen Zeitpunkt nicht sichtbar.")
    elif altitude < 20:
        print(f"\n  WARNUNG: Objekt sehr niedrig (Höhe < 20°)")
        print(f"    Große atmosphärische Extinktion und Unsicherheit!")

print(f"\nObservatorium: {OBSERVATORY}")
print(f"Airmass: {AIRMASS:.4f}")

# Generiere Ausgabe-Dateinamen im selben Ordner wie die Eingabedateien
# Format: YYMMDD_HH.MM_objectname_atmosphere_corrected
date_str = obs_datetime_local.strftime("%y%m%d")
time_str = obs_datetime_local.strftime("%H.%M")
object_str = OBJECT_NAME.lower().replace(" ", "_")
output_basename = f"{date_str}_{time_str}_{object_str}_atmosphere_corrected"
output_csv_path = os.path.join(folder_path, f"{output_basename}.csv")
plot_filename = os.path.join(folder_path, f"{output_basename}.png")

print(f"\nAusgabe-Dateiname (CSV): {output_csv_path}")

if EXTINCTION_FILE:
    print(f"Lade Extinktionskurve aus: {EXTINCTION_FILE}")
    k_ext_wavelengths, k_ext_values = load_extinction_curve(EXTINCTION_FILE)
else:
    print(f"Verwende Standard-Extinktionskurve für: {OBSERVATORY}")
    k_ext_wavelengths, k_ext_values = get_standard_extinction(OBSERVATORY)

print(f"Extinktionskurve: {len(k_ext_wavelengths)} Stützstellen")
print(f"Wellenlängenbereich: {k_ext_wavelengths.min():.0f} - {k_ext_wavelengths.max():.0f} Å")


print(f"\n{'='*70}")
print("LADE ROHDATEN")
print(f"{'='*70}")
print(f"Datei: {INPUT_FILE}")

try:
    obs_data = None
    if INPUT_FILE.lower().endswith(('.fits', '.fit', '.fz')):
        if fits is None:
            raise RuntimeError("Astropy is required to read FITS files. Please install astropy (pip install astropy)")
        hdul = fits.open(INPUT_FILE)
        data_found = False
        # Search for the first HDU that contains usable data
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, 'data', None)
            header = getattr(hdu, 'header', None)
            if data is None:
                continue
            try:
                if hasattr(data, 'names') and data.names is not None:
                    names = [n.lower() for n in data.names]
                    wcol = None
                    for candidate in ('lambda', 'wavelength', 'wave', 'lam'):
                        if candidate in names:
                            wcol = data.names[names.index(candidate)]
                            break
                    if wcol is None:
                        wcol = data.names[0]
                    fcol = None
                    for candidate in ('flux', 's_obs', 'sobs', 'flam', 'f_lambda'):
                        if candidate in names:
                            fcol = data.names[names.index(candidate)]
                            break
                    if fcol is None:
                        if len(data.names) >= 2:
                            fcol = data.names[1]
                        else:
                            continue
                    lambda_obs = np.array(data[wcol], dtype=float)
                    s_obs = np.array(data[fcol], dtype=float)
                    data_found = True
                    break
                else:
                    arr = np.array(data)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        lambda_obs = arr[:, 0].astype(float)
                        s_obs = arr[:, 1].astype(float)
                        data_found = True
                        break
                    elif arr.ndim == 1:
                        n = arr.shape[0]
                        lambda_obs = None
                        if header is not None:
                            crval = header.get('CRVAL1')
                            cdelt = header.get('CDELT1')
                            crpix = header.get('CRPIX1', 1.0)
                            naxis1 = header.get('NAXIS1', n)
                            if (crval is not None) and (cdelt is not None):
                                pix = np.arange(n) + 1
                                lambda_obs = crval + (pix - crpix) * cdelt
                        if lambda_obs is None:
                            try:
                                from astropy.wcs import WCS
                                w = WCS(header)
                                pix = np.arange(n)
                                world = w.wcs_pix2world(pix[:, None], 0)
                                lambda_obs = np.asarray(world).squeeze()
                            except Exception:
                                lambda_obs = None
                        if lambda_obs is not None:
                            s_obs = arr.astype(float)
                            data_found = True
                            break
                        else:
                            hdr_keys = list(header.keys()) if header is not None else []
                            print(f"Skipping HDU {idx}: 1D data found but no CRVAL1/CDELT1 or usable WCS. Header keys: {hdr_keys[:20]}...")
                            continue
            except Exception as e:
                print(f"Warning: failed to parse HDU {idx}: {e}")
                continue
        hdul.close()
        if not data_found:
            raise RuntimeError(f"Could not extract wavelength/flux from FITS file: {INPUT_FILE}")
    else:

        obs_lines = None
        for encoding in ('utf-8', 'latin-1'):
            try:
                with open(INPUT_FILE, 'r', encoding=encoding) as f:
                    obs_lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        if obs_lines is None:

            with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
                obs_lines = f.readlines()

        obs_data = []

        for raw in obs_lines:
            line = raw.strip()
            if line == '' or line.startswith('#'):
                continue

            low = line.lower()
            if 'lambda' in low and ('flux' in low or 's_obs' in low or 'flux' in low):
                continue

            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                parts = line.split()
            if len(parts) >= 2:
                try:
                    lambda_obs = float(parts[0].replace(',', '.'))
                    s_obs = float(parts[1].replace(',', '.'))
                    obs_data.append([lambda_obs, s_obs])
                except ValueError:

                    continue
        obs_data = np.array(obs_data)
        if obs_data.size == 0:
            raise RuntimeError(f"No valid data found in {INPUT_FILE}")
        lambda_obs = obs_data[:, 0]
        s_obs = obs_data[:, 1]

    print(f"\nBeobachtete Rohdaten geladen: {len(lambda_obs)} Datenpunkte")
    print(f"Wellenlängenbereich: {lambda_obs.min():.2f} - {lambda_obs.max():.2f} Å")
    
    mask = (lambda_obs >= LAMBDA_MIN) & (lambda_obs <= LAMBDA_MAX) & (s_obs > 0)
    lambda_obs_filtered = lambda_obs[mask]
    s_obs_filtered = s_obs[mask]
    
    print(f"Nach Filterung ({LAMBDA_MIN}-{LAMBDA_MAX} Å): {len(lambda_obs_filtered)} Datenpunkte")
    
    if len(lambda_obs_filtered) == 0:
        print("\nFEHLER: Keine Datenpunkte im gültigen Wellenlängenbereich!")
        exit(1)

    print(f"\n{'='*70}")
    print("ATMOSPHÄRISCHE KORREKTUR")
    print(f"{'='*70}")
    
    s_obs_atm_corrected, extinction_factor = apply_atmospheric_correction(
        lambda_obs_filtered, s_obs_filtered, AIRMASS, k_ext_wavelengths, k_ext_values
    )
    
    correction_percent = (extinction_factor - 1) * 100
    mean_corr = np.mean(correction_percent)
    max_corr = np.max(correction_percent)
    
    print(f"Mittlere Korrektur: {mean_corr:.3f}%")
    print(f"Maximale Korrektur: {max_corr:.3f}%")
    print(f"Korrektur bei 4000 Å: {correction_percent[np.argmin(np.abs(lambda_obs_filtered-4000))]:.3f}%")
    print(f"Korrektur bei 5500 Å: {correction_percent[np.argmin(np.abs(lambda_obs_filtered-5500))]:.3f}%")
    print(f"Korrektur bei 7000 Å: {correction_percent[np.argmin(np.abs(lambda_obs_filtered-7000))]:.3f}%")
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# Atmosphärisch korrigierte Beobachtung"])
        writer.writerow(["# Objekt", OBJECT_NAME])
        writer.writerow(["# Datum", OBSERVATION_DATE])
        writer.writerow(["# Uhrzeit", f"{OBSERVATION_TIME} ({TIMEZONE})"])
        writer.writerow(["# Ort", LOCATION_NAME])
        writer.writerow(["# Koordinaten", f"{LATITUDE:.4f}°N, {LONGITUDE:.4f}°E"])
        writer.writerow(["# RA[h]", f"{ra_hours:.4f}", "Dec[°]", f"{dec_degrees:.4f}"])
        writer.writerow(["# Altitude[°]", f"{altitude:.2f}", "Azimuth[°]", f"{azimuth:.2f}"])
        writer.writerow(["# Airmass", f"{AIRMASS:.4f}"])
        writer.writerow(["# Observatorium", OBSERVATORY])
        writer.writerow([])
        writer.writerow(["lambda_A", "S_obs_raw", "S_obs_atm_corrected", "Extinktionsfaktor", "Korrektur_prozent"])
        for i in range(len(lambda_obs_filtered)):
            writer.writerow([
                f"{lambda_obs_filtered[i]:.3f}",
                f"{s_obs_filtered[i]:.6e}",
                f"{s_obs_atm_corrected[i]:.6e}",
                f"{extinction_factor[i]:.6f}",
                f"{correction_percent[i]:.3f}",
            ])
    
    print(f"\n{'='*70}")
    print("Atmosphärisch korrigierte Daten gespeichert in:")
    print(f"  -> {output_csv_path}")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(k_ext_wavelengths, k_ext_values, 'ro-', markersize=6, linewidth=2)
    ax1.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax1.set_ylabel('k(λ) [mag/airmass]', fontsize=11)
    ax1.set_title(f'Extinktionskurve ({OBSERVATORY})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(lambda_obs_filtered, extinction_factor, 'b-', linewidth=1.5)
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax2.set_ylabel('Extinktionsfaktor', fontsize=11)
    ax2.set_title(f'Korrekturfaktor (Airmass = {AIRMASS})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    textstr = f'Airmass: {AIRMASS:.3f}\n'
    textstr += f'Mittl. Korr.: {mean_corr:.2f}%\n'
    textstr += f'Max. Korr.: {max_corr:.2f}%'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3 = axes[1, 0]
    ax3.plot(lambda_obs_filtered, s_obs_filtered, 'b-', linewidth=1.5, alpha=0.7, 
            label='Roh (beobachtet)')
    ax3.plot(lambda_obs_filtered, s_obs_atm_corrected, 'r-', linewidth=1.5, alpha=0.7,
            label='Atmosphärisch korrigiert')
    ax3.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax3.set_ylabel('Flux', fontsize=11)
    ax3.set_title('Vega-Spektrum: Vorher vs. Nachher', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(lambda_obs_filtered, correction_percent, 'g-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax4.fill_between(lambda_obs_filtered, 0, correction_percent, alpha=0.3, color='g')
    ax4.set_xlabel('Wellenlänge λ [Å]', fontsize=11)
    ax4.set_ylabel('Korrektur [%]', fontsize=11)
    ax4.set_title('Größe der atmosphärischen Korrektur', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nGrafik gespeichert als: {plot_filename}")
    
    print("\n" + "="*70)
    print("[OK] SCHRITT 1 ABGESCHLOSSEN!")
    print("="*70)
    print(f"\nObjekt: {OBJECT_NAME}")
    print(f"Beobachtungszeit: {OBSERVATION_DATE} {OBSERVATION_TIME}")
    print(f"Airmass: {AIRMASS:.4f}")
    print(f"Ausgabedatei: {output_csv_path}")
    print("\nNächster Schritt:")
    print("  -> Führen Sie nun 'step2_calculate_ir.py' aus")
    print("     (nur wenn dies eine Vega-Beobachtung für IR-Kalibration ist)")
    print("  -> Oder führen Sie 'step3_apply_ir.py' aus")
    print("     (für andere Objekte mit bereits bestimmter IR)")
    print("="*70)
    
except FileNotFoundError:
    print(f"\nFEHLER: Datei '{INPUT_FILE}' nicht gefunden!")
    print("Bitte stellen Sie sicher, dass die Rohdaten vorhanden sind.")



# end of programm if it isnt too clear.
