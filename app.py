#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask app: upload Excel multisede -> meteo storico + bayesian forecast -> PDF report
"""

import os
import io
import math
import tempfile
from datetime import date, datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from meteostat import Monthly, Normals, Point

from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------
# Config
# -----------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BASE_TEMP_C_DEFAULT = 18.0

app = Flask(__name__)
app.secret_key = "cambiami_subito_per_prod"  # cambia in produzione

# -----------------------
# Helper functions
# -----------------------
def geocode_city(city_name: str):
    geolocator = Nominatim(user_agent="energy_meteo_flask_app")
    loc = geolocator.geocode(city_name, addressdetails=False, timeout=20)
    if not loc:
        raise ValueError(f"Città non trovata: {city_name}")
    return float(loc.latitude), float(loc.longitude)

def get_monthly_weather(lat: float, lon: float, start_date: date, end_date: date):
    p = Point(lat, lon)
    df = Monthly(p, start_date, end_date).fetch()
    # Ensure tavg exists
    if "tavg" not in df.columns:
        if "tmin" in df.columns and "tmax" in df.columns:
            df["tavg"] = (df["tmin"] + df["tmax"]) / 2.0
    df = df.reset_index().rename(columns={"time": "Data"})
    df["Anno"] = df["Data"].dt.year
    df["Mese"] = df["Data"].dt.month
    return df[["Anno", "Mese", "tavg"]].dropna()

def get_monthly_normals(lat: float, lon: float):
    p = Point(lat, lon)
    df = Normals(p, 1991, 2020).fetch().reset_index().rename(columns={"month":"Mese"})
    if "tavg" not in df.columns and "tmin" in df.columns and "tmax" in df.columns:
        df["tavg"] = (df["tmin"] + df["tmax"]) / 2.0
    return df[["Mese","tavg"]].dropna()

def get_seasonal_anomalies(lat: float, lon: float, year:int):
    """Uses Open-Meteo climate API for seasonal anomalies. If fails, returns zeros."""
    import requests
    url = "https://climate-api.open-meteo.com/v1/seasonal"
    params = {"latitude": lat, "longitude": lon, "monthly_temperature_2m_anomaly": "true", "models":"ecmwf_seas5"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        times = data.get("monthly", {}).get("time", [])
        anom = data.get("monthly", {}).get("temperature_2m_anomaly", [])
        records = []
        for t,a in zip(times, anom):
            d = datetime.fromisoformat(t).date()
            records.append({"Anno":d.year, "Mese": d.month, "temp_anomaly": a})
        df = pd.DataFrame(records)
        dfy = df[df["Anno"]==year][["Mese","temp_anomaly"]]
        full = pd.DataFrame({"Mese": list(range(1,13))}).merge(dfy, on="Mese", how="left")
        full["temp_anomaly"] = full["temp_anomaly"].fillna(0.0)
        return full
    except Exception:
        # fallback: zeros
        return pd.DataFrame({"Mese": list(range(1,13)), "temp_anomaly": [0.0]*12})

def add_degree_days(df, temp_col="tavg", base=18.0):
    df = df.copy()
    df["HDD"] = np.clip(base - df[temp_col], 0, None)
    df["CDD"] = np.clip(df[temp_col] - base, 0, None)
    return df

def build_model_and_forecast(hist_df, base_temp=BASE_TEMP_C_DEFAULT, next_year=None):
    """hist_df must have columns: Anno, Mese, tavg, Elettricità_kWh, Gas_m3"""
    df = hist_df.copy().dropna(subset=["tavg"])
    df = add_degree_days(df, temp_col="tavg", base=base_temp)
    # Features
    X = df[["Mese","HDD","CDD","Anno"]].astype(float)
    # One-hot month
    ct = ColumnTransformer(
        [("m", OneHotEncoder(drop="first", sparse_output=False), ["Mese"])],
        remainder="passthrough"
    )
    # Train two models
    models = {}
    resid = {}
    metrics = {}
    for col, target_name in [("Elettricità_kWh","Elettricità_kWh"), ("Gas_m3","Gas_m3")]:
        if col not in df.columns:
            models[target_name] = None
            resid[target_name] = np.array([])
            metrics[target_name] = {"R2": None, "MAE": None}
            continue
        y = df[col].astype(float)
        pipe = Pipeline([("ct", ct), ("br", BayesianRidge(compute_score=True))])
        pipe.fit(X,y)
        yhat = pipe.predict(X)
        models[target_name] = pipe
        resid[target_name] = y - yhat
        metrics[target_name] = {"R2": float(r2_score(y,yhat)), "MAE": float(mean_absolute_error(y,yhat))}
    # Forecast next year
    if next_year is None:
        next_year = int(df["Anno"].max()) + 1
    normals = None  # to be provided by caller if needed
    # Create placeholder forecast frame (tavg must be set by caller with normals+anomalies)
    forecast = pd.DataFrame({"Anno":[next_year]*12, "Mese": list(range(1,13))})
    # compute HDD/CDD after caller sets forecast['tavg']
    return models, resid, metrics, forecast

def bootstrap_pi(preds, resid, n=1000, alpha=0.2):
    rng = np.random.default_rng(123)
    lower = []
    upper = []
    if len(resid)==0:
        # fallback: wide interval
        return preds*0.9, preds*1.1
    for p in preds:
        sims = p + rng.choice(resid, size=n, replace=True)
        lower.append(np.quantile(sims, alpha/2))
        upper.append(np.quantile(sims, 1-alpha/2))
    return np.array(lower), np.array(upper)

def plot_history_and_forecast(hist_df, forecast_df, value_col_hist, value_col_fc, title, outpath):
    """
    hist_df: columns Anno,Mese,value_col_hist
    forecast_df: columns Mese, value_col_fc, low, high
    """
    plt.figure(figsize=(10,4))
    # historical years
    years = sorted(hist_df["Anno"].unique())
    for y in years:
        sub = hist_df[hist_df["Anno"]==y]
        plt.plot(sub["Mese"], sub[value_col_hist], label=str(y), marker='o', linewidth=1)
    # forecast
    plt.plot(forecast_df["Mese"], forecast_df[value_col_fc], 'k--', label=f"Forecast {forecast_df['Anno'].iat[0]}", linewidth=2, marker='o')
    # PI
    plt.fill_between(forecast_df["Mese"], forecast_df["low"], forecast_df["high"], color='gray', alpha=0.25)
    plt.title(title)
    plt.xlabel("Mese")
    plt.xticks(list(range(1,13)))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", example_columns="Sede,Città,Anno,Mese,Elettricità_kWh,Gas_m3")

@app.route("/process", methods=["POST"])
def process():
    # file upload
    file = request.files.get("file")
    if not file:
        flash("Carica un file Excel (.xlsx).")
        return redirect(url_for("index"))
    base_temp = float(request.form.get("base_temp", BASE_TEMP_C_DEFAULT))
    # save upload
    fname = os.path.join(UPLOAD_FOLDER, f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    file.save(fname)
    df = pd.read_excel(fname, sheet_name=0)
    expected = {"Sede","Città","Anno","Mese","Elettricità_kWh","Gas_m3"}
    if not expected.issubset(df.columns):
        flash(f"Il file deve contenere le colonne: {', '.join(sorted(expected))}")
        return redirect(url_for("index"))

    df["Anno"] = df["Anno"].astype(int)
    df["Mese"] = df["Mese"].astype(int)

    sedi = df["Sede"].dropna().unique().tolist()

    # container for results
    results = {}
    pdf_images = []

    for sede in sedi:
        df_s = df[df["Sede"]==sede].copy()
        city = df_s["Città"].mode().iat[0]
        years = sorted(df_s["Anno"].unique())
        start = date(min(years),1,1)
        end = date(max(years),12,31)
        try:
            lat, lon = geocode_city(city)
        except Exception as e:
            lat, lon = None, None

        # get meteo storico
        try:
            meteo = get_monthly_weather(lat, lon, start, end)
        except Exception:
            # fallback: create monthly tavg by using climatological sine + noise
            meteo = pd.DataFrame({
                "Anno": np.repeat(years, 12),
                "Mese": list(range(1,13))*len(years),
                "tavg": np.tile(10 + 10*np.sin((np.array(range(1,13))-1)/12*2*np.pi), len(years))
            })

        merged = pd.merge(df_s[["Anno","Mese","Elettricità_kWh","Gas_m3"]], meteo, on=["Anno","Mese"], how="inner")
        if merged.empty:
            flash(f"Nessun dato storico valido per la sede {sede} dopo merge meteo.")
            continue

        # train models
        models, resid, metrics, forecast_base = build_model_and_forecast(merged, base_temp)
        # normals and anomalies for next year
        next_year = max(years) + 1
        try:
            normals = get_monthly_normals(lat, lon)
        except Exception:
            normals = pd.DataFrame({"Mese": list(range(1,13)), "tavg": np.tile(10 + 10*np.sin((np.array(range(1,13))-1)/12*2*np.pi),1)})
        anomalies = get_seasonal_anomalies(lat, lon, next_year)
        # build forecast tavg
        fc = forecast_base.copy()
        fc = fc.merge(normals.rename(columns={"tavg":"tavg_norm"}), on="Mese", how="left")
        fc = fc.merge(anomalies, on="Mese", how="left")
        fc["temp_anomaly"] = fc["temp_anomaly"].fillna(0.0)
        fc["tavg"] = fc["tavg_norm"] + fc["temp_anomaly"]
        fc = add_degree_days(fc, temp_col="tavg", base=base_temp)
        # get predictions
        Xf = fc[["Mese","HDD","CDD","Anno"]].astype(float)
        # If model exists, predict; else NaNs
        if models.get("Elettricità_kWh") is not None:
            fc["Elettricità_kWh_pred"] = models["Elettricità_kWh"].predict(Xf)
            low, high = bootstrap_pi(fc["Elettricità_kWh_pred"].values, resid["Elettricità_kWh"], n=600, alpha=0.2)
            fc["Elettricità_kWh_low"] = low
            fc["Elettricità_kWh_high"] = high
        else:
            fc["Elettricità_kWh_pred"] = np.nan
            fc["Elettricità_kWh_low"] = np.nan
            fc["Elettricità_kWh_high"] = np.nan

        if models.get("Gas_m3") is not None:
            fc["Gas_m3_pred"] = models["Gas_m3"].predict(Xf)
            lowg, highg = bootstrap_pi(fc["Gas_m3_pred"].values, resid["Gas_m3"], n=600, alpha=0.2)
            fc["Gas_m3_low"] = lowg
            fc["Gas_m3_high"] = highg
        else:
            fc["Gas_m3_pred"] = np.nan
            fc["Gas_m3_low"] = np.nan
            fc["Gas_m3_high"] = np.nan

        # plots
        safe_name = "".join(c for c in sede if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        out_img_e = os.path.join(OUTPUT_FOLDER, f"{safe_name}_elec.png")
        out_img_g = os.path.join(OUTPUT_FOLDER, f"{safe_name}_gas.png")

        plot_history_and_forecast(merged[["Anno","Mese","Elettricità_kWh"]], fc[["Anno","Mese","Elettricità_kWh_pred","Elettricità_kWh_low","Elettricità_kWh_high"]].rename(columns={
            "Elettricità_kWh_pred":"Elettricità_kWh_pred","Elettricità_kWh_low":"low","Elettricità_kWh_high":"high"
        }), "Elettricità_kWh", "Elettricità_kWh_pred", f"{sede} — Elettricità (kWh)", out_img_e)

        plot_history_and_forecast(merged[["Anno","Mese","Gas_m3"]], fc[["Anno","Mese","Gas_m3_pred","Gas_m3_low","Gas_m3_high"]].rename(columns={
            "Gas_m3_pred":"Gas_m3_pred","Gas_m3_low":"low","Gas_m3_high":"high"
        }), "Gas_m3", "Gas_m3_pred", f"{sede} — Gas (m³)", out_img_g)

        pdf_images.extend([out_img_e, out_img_g])

        # save per-sede results
        results[sede] = {
            "city": city,
            "latlon": (lat, lon),
            "metrics": metrics,
            "history": merged,
            "forecast": fc,
            "plots": {"elec": out_img_e, "gas": out_img_g}
        }

    # Save intermediate results in a temp pickle or in session? For simplicity, we keep a temporary file.
    temp_report_id = datetime.now().strftime("%Y%m%d%H%M%S")
    report_path = os.path.join(OUTPUT_FOLDER, f"report_{temp_report_id}.pdf")

    # generate PDF
    generate_pdf_report(report_path, results, base_temp)
    return render_template("results.html", report_url=url_for("download_report", filename=os.path.basename(report_path)), results=results)

@app.route("/download/<path:filename>", methods=["GET"])
def download_report(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        flash("Report non trovato.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=filename)

# -----------------------
# PDF generation
# -----------------------
def generate_pdf_report(output_pdf_path, results_dict, base_temp):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Report Consumi Energetici — Previsioni & Meteo", styles["Title"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Base per HDD/CDD: {base_temp} °C", styles["Normal"]))
    story.append(Spacer(1,12))

    for sede, info in results_dict.items():
        story.append(Paragraph(f"Sede: {sede}", styles["Heading2"]))
        story.append(Paragraph(f"Città: {info.get('city','-')}", styles["Normal"]))
        latlon = info.get("latlon", (None,None))
        story.append(Paragraph(f"Lat/Lon: {latlon[0]}, {latlon[1]}", styles["Normal"]))
        story.append(Spacer(1,8))

        # metrics table
        mets = info.get("metrics", {})
        data = [["Utenza","R²","MAE"]]
        for k,v in mets.items():
            if v["R2"] is None:
                data.append([k, "-", "-"])
            else:
                data.append([k, f"{v['R2']:.3f}", f"{v['MAE']:.2f}"])
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f2f2f2")),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")
        ]))
        story.append(t)
        story.append(Spacer(1,12))

        # Add images if present
        plots = info.get("plots", {})
        for key,imgp in plots.items():
            if imgp and os.path.exists(imgp):
                # scale image to page width
                iw = RLImage(imgp, width=500, height=200)
                story.append(iw)
                story.append(Spacer(1,8))

        # Add a small table of forecast (first few rows) for readability
        fc = info.get("forecast")
        if fc is not None and not fc.empty:
            df_show = fc[["Anno","Mese","tavg"]].copy()
            # Add preds columns if exist
            if "Elettricità_kWh_pred" in fc.columns:
                df_show["E_el_pred"] = fc["Elettricità_kWh_pred"].round(0)
            if "Gas_m3_pred" in fc.columns:
                df_show["Gas_pred"] = fc["Gas_m3_pred"].round(0)
            # Convert to Table
            tbl_data = [df_show.columns.tolist()] + df_show.values.tolist()
            t2 = Table(tbl_data, hAlign="LEFT")
            t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f8f8f8"))]))
            story.append(t2)
            story.append(Spacer(1,12))

        story.append(PageBreak())

    # Build
    doc.build(story)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
