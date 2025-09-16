import streamlit as st
import pandas as pd
import numpy as np
from dea_algorithm import dea_with_slacks

st.title("DEA Analysis with CRS/VRS and Slacks")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";", engine="python").dropna()

    st.write("### Raw Data")
    st.dataframe(df.head())

    # User picks which columns are inputs/outputs
    dmu_col = st.selectbox("Select DMU column", df.columns)
    input_cols = st.multiselect("Select input columns", [c for c in df.columns if c != dmu_col])
    output_cols = st.multiselect("Select output columns", [c for c in df.columns if c != dmu_col])

    orientation = st.radio("Select orientation", ["input", "output"])
    rts = st.radio("Select returns to scale", ["CRS", "VRS"])

    if input_cols and output_cols:
        dmus = df[dmu_col].astype(str).tolist()
        X = df[input_cols].values
        Y = df[output_cols].values

        scores, lambdas, slacks_in, slacks_out = dea_with_slacks(X, Y, orientation, rts)

        # Build results DataFrame
        results = pd.DataFrame({
            "DMU": dmus,
            "Efficiency": [float(s) for s in scores]
        })

        # Add slacks
        for i, col in enumerate(input_cols):
            results[f"Slack_in_{col}"] = [row[i] for row in slacks_in]

        for j, col in enumerate(output_cols):
            results[f"Slack_out_{col}"] = [row[j] for row in slacks_out]

        st.write("### DEA Results")
        st.dataframe(results)

        # Explain
        st.write("""
        - **Efficiency = 1 (or 100%)** → DMU is efficient.  
        - **Efficiency < 1 (input-oriented)** → DMU can proportionally reduce inputs.  
        - **Efficiency > 1 (output-oriented)** → DMU can proportionally increase outputs.  
        - **Slack_in** values > 0 → input overuse (can be reduced further).  
        - **Slack_out** values > 0 → output shortfalls (can be increased further).  
        """)
