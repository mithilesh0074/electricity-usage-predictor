import pandas as pd

# Path to your uploaded file
excel_file = "2025-india-annual-calendar-holidays-02.xlsx"

# Read Excel
df = pd.read_excel(excel_file)

# IMPORTANT: adjust column names if needed after you print df.head()
print(df.columns)

# Assume columns like: Date, Holiday
df = df.rename(columns={
    df.columns[0]: "date",
    df.columns[1]: "festival"
})

# Convert date format
df["date"] = pd.to_datetime(df["date"]).dt.date

# Save to CSV
df.to_csv("festivals.csv", index=False)

print("festivals.csv created successfully")
