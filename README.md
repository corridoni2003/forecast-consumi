# Energy & Meteo Forecaster (Flask)

## Setup
1. Crea virtualenv:
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. Avvia:
python app.py

3. Apri nel browser: http://127.0.0.1:8501

## Note
- L'app usa Meteostat e Open-Meteo per meteo e anomalie. Se ci sono problemi di rete o geocoding, l'app usa fallback sintetici.
- Il PDF viene generato con ReportLab e include grafici (PNG) e tabelle. I PNG sono salvati in `static/outputs`.
