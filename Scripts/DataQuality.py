# %%
from Classess.DataAnalyzer import DataQualityAnalyzer
import pandas as pd
from pprint import pprint

# %%
df1 = pd.read_csv("../Data/train_transaction.csv")
df2 = pd.read_csv("../Data/train_identity.csv")

for df in (df1, df2):
    analyzer = DataQualityAnalyzer(df)
    report = analyzer.generate_quality_report()
    print(f"Processed {len(df)} rows with {len(df.columns)} columns.")
    pprint(report)


# %%
