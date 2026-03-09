import pandas as pd
import numpy as np

np.random.seed(42)

data=[]

for i in range(800):

    family=np.random.randint(1,8)
    rooms=np.random.randint(1,6)
    bedrooms=np.random.randint(1,5)
    bathrooms=np.random.randint(1,4)

    temp=np.random.uniform(20,40)

    weekends=np.random.randint(6,12)
    festivals=np.random.randint(0,4)

    remaining_days=np.random.randint(20,60)

    base=2+family*0.9+rooms*0.5+bedrooms*0.4

    temp_effect=(temp-25)*0.18

    weekend_effect=weekends*0.12

    festival_effect=festivals*0.25

    units=base+temp_effect+weekend_effect+festival_effect

    data.append([
        family,
        rooms,
        bedrooms,
        bathrooms,
        temp,
        weekends,
        festivals,
        remaining_days,
        units
    ])

df=pd.DataFrame(data,columns=[
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

df.to_csv("usage_history.csv",index=False)

print("Dataset created successfully")
