import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("usage_history.csv")

X=df.drop("daily_units",axis=1)
y=df["daily_units"]

model=RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X,y)

joblib.dump(model,"electricity_model.pkl")

print("Model trained and saved successfully")
