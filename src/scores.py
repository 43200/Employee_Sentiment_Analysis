import pandas as pd
import numpy as np

def compute_monthly_scores(df, date_col="date", emp_col="employee_id", label_col="sentiment_label"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    # map labels to scores
    score_map = {"Positive":1, "Negative":-1, "Neutral":0}
    df["msg_score"] = df[label_col].map(score_map).fillna(0)
    monthly = df.groupby([emp_col, "month"]).msg_score.sum().reset_index()
    monthly = monthly.rename(columns={"msg_score":"monthly_score"})
    return monthly

def top_three_rankings(monthly_df, month):
    # month is a pandas.Timestamp (month start)
    subset = monthly_df[monthly_df["month"]==month]
    # top positive
    top_pos = subset.sort_values(["monthly_score","employee_id"], ascending=[False, True]).head(3)
    top_neg = subset.sort_values(["monthly_score","employee_id"], ascending=[True, True]).head(3)
    return top_pos, top_neg

def detect_flight_risk_rolling(df, date_col="date", emp_col="employee_id", label_col="sentiment_label", window_days=30, negative_label="Negative", threshold=4):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([emp_col, date_col])
    risks = set()
    # for each employee, sliding window count of negative messages
    for emp, g in df.groupby(emp_col):
        dates = g.loc[g[label_col]==negative_label, date_col].dropna().sort_values().astype('datetime64[ns]').values
        if len(dates) < threshold:
            continue
        # two pointers over sorted dates
        i = 0
        for j in range(len(dates)):
            while i<j and (dates[j] - dates[i]).astype('timedelta64[D]').astype(int) > (window_days-1):
                i += 1
            if (j - i + 1) >= threshold:
                risks.add(emp)
                break
    return sorted(list(risks))