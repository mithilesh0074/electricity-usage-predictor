import streamlit as st
import pandas as pd
import datetime
import numpy as np
import requests
import joblib
import os

# ================= CONFIG =================

API_KEY = "a46a70e05675b9a7e3073ab5486bbc9d"

MODEL_FILE = "electricity_model.pkl"
DATA_FILE = "usage_history.csv"

BILLING_DAYS = 60

# ================= LOAD MODEL =================

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("Run train_model.py first")
    st.stop()

# ================= WEATHER =================

def get_future_avg_temp(city):

    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url).json()

        temps = [entry["main"]["temp"] for entry in data["list"]]

        return np.mean(temps)

    except:
        return 30

# ================= FESTIVALS =================

@st.cache_data
def load_festivals():

    df = pd.read_csv("festivals.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df

festivals = load_festivals()

def count_future_events(today, remaining_days):

    weekends = 0
    fest = 0

    for i in range(remaining_days):

        d = today + datetime.timedelta(days=i)

        if d.weekday() >= 5:
            weekends += 1

        if d in festivals["date"].values:
            fest += 1

    return weekends, fest

# ================= TARIFF =================

SLABS = [
    (200, 4.95),
    (250, 6.65),
    (300, 8.80),
    (400, 9.95),
    (500, 11.05),
    (float("inf"), 12.15)
]

def calculate_bill(units):

    cost = 0
    prev_limit = 0

    for limit, rate in SLABS:

        if units > limit:
            cost += (limit - prev_limit) * rate
            prev_limit = limit
        else:
            cost += (units - prev_limit) * rate
            break

    return cost

def get_next_slab_limit(units):

    for limit, rate in SLABS:
        if units <= limit:
            return limit

    return None

# ================= SAVE DATA =================

def save_usage(features, predicted_daily):

    row = pd.DataFrame([[
        features[0][0],
        features[0][1],
        features[0][2],
        features[0][3],
        features[0][4],
        features[0][5],
        features[0][6],
        features[0][7],
        predicted_daily
    ]], columns=[
        "family_size",
        "rooms",
        "bedrooms",
        "bathrooms",
        "temperature",
        "weekends",
        "festivals",
        "remaining_days",
        "daily_units"
    ])

    row.to_csv(DATA_FILE, mode="a", header=False, index=False)

# ================= RETRAIN =================

def retrain_model():

    df = pd.read_csv(DATA_FILE)

    X = df.drop("daily_units", axis=1)
    y = df["daily_units"]

    new_model = joblib.load(MODEL_FILE)

    new_model.fit(X, y)

    joblib.dump(new_model, MODEL_FILE)

# ================= UI =================

st.title("Electricity Consumption & Tariff Predictor (ML Powered)")

city = st.text_input("City", "Chennai")

start_meter = st.number_input("Start meter reading", value=100.0)
current_meter = st.number_input("Current meter reading", value=238.0)

start_date = st.date_input("Billing start date")

family_size = st.slider("Family size", 1, 8, 4)

rooms = st.number_input("Rooms", value=3)
bedrooms = st.number_input("Bedrooms", value=2)
bathrooms = st.number_input("Bathrooms", value=2)

st.subheader("Appliances")

fans = st.number_input("Number of fans", value=4)
lights = st.number_input("Number of lights", value=6)

has_ev = st.checkbox("Do you own an EV?")

ev_battery = 0
ev_charge_per_week = 0

if has_ev:

    ev_battery = st.number_input("EV battery capacity (kWh)", value=40.0)
    ev_charge_per_week = st.number_input("Charges per week", value=2)

# ================= CALCULATIONS =================

today = datetime.date.today()

days_completed = (today - start_date).days

units_used = current_meter - start_meter

remaining_days = max(0, BILLING_DAYS - days_completed)

avg_temp = get_future_avg_temp(city)

weekends, fest = count_future_events(today, remaining_days)

# ================= ML PREDICTION =================

features = [[
    family_size,
    rooms,
    bedrooms,
    bathrooms,
    avg_temp,
    weekends,
    fest,
    remaining_days
]]

predicted_daily = model.predict(features)[0]

# ================= ACTUAL DAILY AVG =================

if days_completed > 0:
    actual_daily_avg = units_used / days_completed
else:
    actual_daily_avg = predicted_daily

# ================= EV LOAD =================

ev_daily_units = 0

if has_ev:
    ev_daily_units = (ev_battery * ev_charge_per_week) / 7

# ================= FINAL DAILY USAGE =================

final_daily_usage = actual_daily_avg + ev_daily_units

# ================= FUTURE UNITS =================

future_units = final_daily_usage * remaining_days

total_units = units_used + future_units

# ================= BILL =================

bill = calculate_bill(total_units)

# ================= SAFE LIMIT =================

next_limit = get_next_slab_limit(total_units)

if next_limit == float("inf"):
    safe_units = 0
else:
    safe_units = next_limit - total_units

# ================= OUTPUT =================

st.subheader("Results")

st.write("Units used so far:", round(units_used,2))

st.write("Predicted total usage:", round(total_units,2))

st.write("Estimated bill: ₹", round(bill,2))

# ================= SAFE LIMIT OUTPUT =================

if next_limit == float("inf"):

    st.warning("You are already in highest tariff slab (>500 units).")

else:

    st.success(f"You can still use {round(safe_units,2)} units before reaching {next_limit} slab.")

    safe_daily = safe_units / max(remaining_days,1)

    st.write(f"Safe extra usage per day: {round(safe_daily,2)} units/day")

# ================= SAVE & LEARN =================

save_usage(features, predicted_daily)

retrain_model()