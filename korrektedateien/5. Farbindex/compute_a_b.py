import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

# Referenz-Datenbank
REFERENCE_STARS = {
    "Vega": 0.00,
    "Sirius": -0.01,
    "Altair": 0.22,
    "Procyon": 0.42,
    "Arcturus": 1.23,
    "Betelgeuse": 1.85,
    "Aldebaran": 1.54,
    "Regulus": -0.16,
    "Spica": -0.23,
    "Canopus": 0.15,
    "Bellatrix": -0.21,
    "Castor": 0.04,
    "Pollux": 1.00,
    "Deneb": 0.09,
    "Sadr": 0.79,
    "Sheliak": 0.02,
    "Alderamin": 0.22,
    "Alshain": 0.57,
    "MuCephei": 2.30,
    "pCygni": 0.64,
    "LamCyg": 0.36,
    "TCrB": 1.20,
    "ThetaCygni": 0.37,
    "10Lac": -0.27,
    "wr140g": -0.25,
    "Aldebaran": 1.54,
    "Alnitak": -0.20,
    "Altair": 0.22,
    "Bellatrix": -0.21,
    "Betelgeuse": 1.85,
    "Capella": 0.80,
    "Capella2": 0.80,
    "Deneb": -0.04,
    "Mintaka": -0.22,
    "Pollux": 1.00,
    "Procyon": 0.42,
    "Rigel": -0.03,
    "Vega": 0.00,
    "Vega_Exposure15_Count40": 0.00,
    "Vega2": 0.00,
    "VegaNeu": 0.00,
}

# B- und V-Band (Johnson-Filter)
B_MIN, B_MAX = 3900, 4900
V_MIN, V_MAX = 5000, 6000

def integrate_band(wl, flux, band_min, band_max):
    mask = (wl >= band_min) & (wl <= band_max)
    if np.sum(mask) < 2:
        return np.nan
    return np.trapezoid(flux[mask], wl[mask])

def compute_CI(FB, FV):
    return -2.5 * np.log10(FB / FV)


# Ordner einlesen & Berechnungen
def compute_a_b(input_folder):
    files = glob(os.path.join(input_folder, "*.csv"))
    if not files:
        print("Keine CSV-Dateien gefunden!")
        return

    data_fit = []

    for f in files:
        basename = os.path.basename(f).replace(".csv", "")
        parts = basename.split("_")
        if len(parts) < 3:
            continue
        star_name = parts[2].capitalize()
        if star_name not in REFERENCE_STARS:
            continue

        df = pd.read_csv(f, comment="#", encoding="cp1252")
        wl = df['lambda_A'].to_numpy()
        fx = df['S_obs_atm_corrected'].to_numpy()

        FB = integrate_band(wl, fx, B_MIN, B_MAX)
        FV = integrate_band(wl, fx, V_MIN, V_MAX)
        if np.isnan(FB) or np.isnan(FV) or FB <= 0 or FV <= 0:
            continue

        CI = compute_CI(FB, FV)
        BV_catalog = REFERENCE_STARS[star_name]
        data_fit.append((CI, BV_catalog, star_name))

    if len(data_fit) < 2:
        print("Nicht genug Referenzsterne gefunden!")
        return

    # Lineare Regression B-V = a*CI + b
    CI_vals = np.array([d[0] for d in data_fit])
    BV_vals = np.array([d[1] for d in data_fit])
    X = np.vstack([CI_vals, np.ones_like(CI_vals)]).T
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ BV_vals
    a, b = beta

    residuals = BV_vals - (a * CI_vals + b)
    sigma2 = np.sum(residuals**2) / (len(CI_vals) - 2)
    cov_beta = sigma2 * XtX_inv
    se_a = np.sqrt(cov_beta[0, 0])
    se_b = np.sqrt(cov_beta[1, 1])
    rmse = np.sqrt(np.mean(residuals**2))

    # Ergebnis TXT speichern
    txt_path = os.path.join(input_folder, "calibration_results.txt")
    with open(txt_path, "w") as f:
        f.write("=== Referenzsterne ===\n")
        for CI, BV, nm in data_fit:
            f.write(f"{nm:15s} CI = {CI:.4f} BV_kat = {BV}\n")
        f.write("\n=== KALIBRATION ===\n")
        f.write(f"a = {a:.5f} ± {se_a:.5f}\n")
        f.write(f"b = {b:.5f} ± {se_b:.5f}\n")
        f.write(f"RMSE = {rmse:.4f} mag\n")
    print(f"\nErgebnisse gespeichert als TXT: {txt_path}")

    # Plot erstellen und speichern
    plt.figure(figsize=(7, 5))
    plt.scatter(CI_vals, BV_vals, label="Referenzsterne")
    xfit = np.linspace(min(CI_vals), max(CI_vals), 100)
    plt.plot(xfit, a*xfit + b, label="Fit", linewidth=2)
    plt.xlabel("CI (instrumentell)")
    plt.ylabel("B-V (katalog)")
    plt.grid()
    plt.legend()
    plt.title("Kalibration: B-V = a * CI + b")
    png_path = os.path.join(input_folder, "calibration_plot.png")
    plt.savefig(png_path)
    plt.show()
    print(f"Plot gespeichert als PNG: {png_path}")


# Start
if __name__ == "__main__":
    folder = input("Ordner mit Stern-CSV-Dateien: ").strip()
    compute_a_b(folder)
