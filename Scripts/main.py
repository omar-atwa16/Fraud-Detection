# %% [markdown]
# # Fraud Detection - Dimensional Warehouse & OLAP
#
# **Analytical question:** Where and when is fraud most likely, and where does it cost the most money?
#
# **Fact grain:** one row = one transaction
#
# **Measures:** TransactionAmt, isFraud, fraudAmt
#
# **Dimensions:**
# - dimProduct (ProductCD)
# - dimCard    (card6)
# - dimDevice  (DeviceType)
# - dimTime    (Month -> Day -> Hour)


# %% Imports
import os
import pandas as pd


# %% Load transactions (only the columns we need)
df = pd.read_csv(
    "../Data/train_transaction.csv",
    usecols=["TransactionID", "isFraud", "TransactionDT",
            "TransactionAmt", "ProductCD", "card6"],
)
print("Raw shape:", df.shape)


# %% Load identity and attach DeviceType (left join, one row per transaction)
ident = pd.read_csv("../Data/train_identity.csv",
                    usecols=["TransactionID", "DeviceType"])
assert ident["TransactionID"].is_unique, "Identity has duplicate TransactionIDs!"

df = df.merge(ident, on="TransactionID", how="left", validate="one_to_one")
print("Shape after identity join:", df.shape)


# %% Light cleaning + new columns
df["card6"] = df["card6"].fillna("Unknown")
df["DeviceType"] = df["DeviceType"].fillna("Unknown")   # no identity record

# TransactionDT = seconds from a start point
df["Day"] = df["TransactionDT"] // 86400
df["Month"] = df["Day"] // 30 + 1
df["Hour"] = (df["TransactionDT"] // 3600) % 24

# new measure: money lost to fraud
df["fraudAmt"] = df["TransactionAmt"] * df["isFraud"]


# %% Build dimensions
dimProduct = df[["ProductCD"]].drop_duplicates().reset_index(drop=True)
dimProduct["ProductKey"] = range(1, len(dimProduct) + 1)

dimCard = df[["card6"]].drop_duplicates().reset_index(drop=True)
dimCard["CardKey"] = range(1, len(dimCard) + 1)

dimDevice = df[["DeviceType"]].drop_duplicates().reset_index(drop=True)
dimDevice["DeviceKey"] = range(1, len(dimDevice) + 1)

dimTime = df[["Month", "Day", "Hour"]].drop_duplicates().reset_index(drop=True)
dimTime["TimeKey"] = range(1, len(dimTime) + 1)


# %% Build the fact table (many_to_one joins protect against fan-out)
rowsBefore = len(df)

fact = (
    df.merge(dimProduct, on="ProductCD", how="left", validate="many_to_one")
        .merge(dimCard, on="card6", how="left", validate="many_to_one")
        .merge(dimDevice, on="DeviceType", how="left", validate="many_to_one")
        .merge(dimTime[["Day", "Hour", "TimeKey"]], on=["Day", "Hour"], how="left", validate="many_to_one")
        [["TransactionID", "ProductKey", "CardKey", "DeviceKey", "TimeKey", "TransactionAmt", "isFraud", "fraudAmt"]]
)


# %% Validate: no fan-out, no missing keys
print("Rows before:", rowsBefore, "| after:", len(fact))
assert len(fact) == rowsBefore, "Fan-out detected"
assert fact["TransactionID"].is_unique, "Duplicate transactions"
assert fact[["ProductKey", "CardKey", "DeviceKey", "TimeKey"]].notna().all().all(), "Missing keys"
print("PASS")


# %% Cube = fact + dimensions
# Cube = fact + dimensions
cube = (
    fact.merge(dimProduct, on="ProductKey")
        .merge(dimCard, on="CardKey")
        .merge(dimDevice, on="DeviceKey")
        .merge(dimTime, on="TimeKey")
)

overallRate = cube["isFraud"].mean()
print(f"Overall fraud rate: {overallRate:.2%}")


# %% Helper: one summary table used by every OLAP operation
def summary(data, by):
    s = data.groupby(by).agg(
        transactions=("TransactionID", "count"),
        fraudCases=("isFraud", "sum"),
        fraudRate=("isFraud", "mean"),
        fraudAmt=("fraudAmt", "sum"),
    )
    s["lift"] = s["fraudRate"] / overallRate   # 2.0 = twice as risky as average
    s["share_of_fraud_$"] = s["fraudAmt"] / cube["fraudAmt"].sum()
    return s.round(3)


# %% Aggregation: which product is riskiest? which costs the most?
print(summary(cube, "ProductCD").sort_values("lift", ascending=False))


# %% Aggregation: card type
print(summary(cube, "card6").sort_values("lift", ascending=False))


# %% Aggregation: device type
print(summary(cube, "DeviceType").sort_values("lift", ascending=False))


# %% Roll-up: Day -> Month
print(summary(cube, "Month"))


# %% Drill-down: Month -> Day (one month)
print(summary(cube[cube["Month"] == 1], "Day"))


# %% Drill-down: Day -> Hour (all days together) - when does fraud happen?
print(summary(cube, "Hour"))


# %% Slice: ONE condition (ProductCD = C) - change to any product
print(summary(cube[cube["ProductCD"] == "C"], "Month"))


# %% Dice: MULTIPLE conditions
dice = cube[
    cube["ProductCD"].isin(["W", "C"])
    & (cube["card6"] == "credit")
    & cube["Month"].isin([1, 2, 3])
]
print(summary(dice, ["ProductCD", "Month"]))


# %% Pivot: product x month (fraud rate)
pivot = cube.pivot_table(
    index="ProductCD", columns="Month", values="isFraud", aggfunc="mean"
)
print(pivot.round(3))


# %% Pivot: product x card type (fraud rate)
print(cube.pivot_table(
    index="ProductCD", columns="card6", values="isFraud", aggfunc="mean"
).round(3))


# %% Pivot: product x device type (fraud rate)
print(cube.pivot_table(
    index="ProductCD", columns="DeviceType", values="isFraud", aggfunc="mean"
).round(3))


# %% Save tables (for a future dashboard)
# Save tables (for a future dashboard)
tablesPath = "../Tables/"
os.makedirs(tablesPath, exist_ok=True)

cube.to_csv(os.path.join(tablesPath, "fraud_cube.csv"), index=False)
fact.to_csv(os.path.join(tablesPath, "fact_transactions.csv"), index=False)
dimProduct.to_csv(os.path.join(tablesPath, "dimProduct.csv"), index=False)
dimCard.to_csv(os.path.join(tablesPath, "dimCard.csv"), index=False)
dimDevice.to_csv(os.path.join(tablesPath, "dimDevice.csv"), index=False)
dimTime.to_csv(os.path.join(tablesPath, "dimTime.csv"), index=False)