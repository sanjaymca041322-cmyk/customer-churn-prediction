import pandas as pd
import numpy as np
import random

# Number of customers
num_customers = 1000

np.random.seed(42)

data = []

for i in range(1, num_customers + 1):

    age = np.random.randint(18, 70)
    gender = random.choice(["Male", "Female"])
    tenure = np.random.randint(1, 73)

    monthly_charges = round(np.random.uniform(300, 3000), 2)

    contract = random.choice([
        "Monthly",
        "Quarterly",
        "Yearly"
    ])

    support_calls = np.random.randint(0, 11)

    total_spend = round(monthly_charges * tenure, 2)

    # Logic to generate churn
    churn_probability = 0.2

    if contract == "Monthly":
        churn_probability += 0.25

    if support_calls > 5:
        churn_probability += 0.20

    if tenure < 12:
        churn_probability += 0.15

    churn = "Yes" if random.random() < churn_probability else "No"

    data.append([
        f"C{i:04d}",
        age,
        gender,
        tenure,
        monthly_charges,
        total_spend,
        contract,
        support_calls,
        churn
    ])

columns = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "TotalSpend",
    "Contract",
    "SupportCalls",
    "Churn"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/customer_churn.csv", index=False)

print("Dataset created successfully!")
print(df.head())