import streamlit as st
import pandas as pd
import numpy as np
import io
from dea_algorithm import dea_with_slacks

st.title("DEA Analysis with CRS/VRS, Slacks, and Recommendations")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";", engine="python").dropna()

    st.write("### Raw Data")
    st.dataframe(df.head())

    # Select DMU, inputs, outputs
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

        # === Efficiency Results ===
        res_df = pd.DataFrame({
            "DMU": dmus,
            "Efficiency": [float(s) for s in scores]
        })

        # Slacks
        slacks_in_df = pd.DataFrame(slacks_in, columns=input_cols, index=dmus)
        slacks_out_df = pd.DataFrame(slacks_out, columns=output_cols, index=dmus)

        st.write("### DEA Results")
        st.dataframe(res_df)

        st.write("### Input Slacks (excess inputs)", slacks_in_df)
        st.write("### Output Slacks (shortfall outputs)", slacks_out_df)

        # Peer weights
        lambdas_df = pd.DataFrame(lambdas, index=dmus, columns=dmus)
        st.write("### Peer Weights (Reference Set)", lambdas_df)

        # === Recommendations ===
        recommendations = []
        for i, dmu in enumerate(dmus):
            rec_text = []
            if scores[i] < 1 and orientation == "input":
                rec_text.append(f"Reduce inputs proportionally by {(1 - scores[i]) * 100:.1f}%.")
            elif scores[i] > 1 and orientation == "output":
                rec_text.append(f"Increase outputs proportionally by {(scores[i] - 1) * 100:.1f}%.")
            
            # Add input slacks
            input_excess = slacks_in_df.iloc[i]
            excess_list = [f"{col} (-{val:.2f})" for col, val in input_excess.items() if val > 1e-6]
            if excess_list:
                rec_text.append("Reduce excess in: " + ", ".join(excess_list))

            # Add output slacks
            output_short = slacks_out_df.iloc[i]
            short_list = [f"{col} (+{val:.2f})" for col, val in output_short.items() if val > 1e-6]
            if short_list:
                rec_text.append("Increase outputs in: " + ", ".join(short_list))

            # Add benchmark peers
            peers = [dmus[j] for j, lam in enumerate(lambdas[i]) if lam > 1e-6]
            if peers:
                rec_text.append("Benchmark against: " + ", ".join(peers))

            recommendations.append({
                "DMU": dmu,
                "Recommendation": " | ".join(rec_text) if rec_text else "Already efficient"
            })

        recs_df = pd.DataFrame(recommendations)
        st.write("### Recommendations", recs_df)

        # === Interpretation Guide ===
        st.markdown("### Interpretation Guide")
        st.markdown("""
        - **Efficiency = 1** → DMU is efficient (on the frontier).  
        - **Efficiency < 1** (input-oriented) → reduce inputs proportionally.  
        - **Efficiency > 1** (output-oriented) → expand outputs proportionally.  
        - **Input slacks > 0** → reduce these inputs further.  
        - **Output slacks > 0** → increase these outputs further.  
        - **Peer Weights (λ values)** → show benchmark DMUs for each inefficient unit.  
        - **Recommendations** → actionable steps combining efficiency, slacks, and peers.
        """)

        # === Download Excel Report ===
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            res_df.to_excel(writer, sheet_name="Efficiency", index=False)
            slacks_in_df.to_excel(writer, sheet_name="Input_Slacks")
            slacks_out_df.to_excel(writer, sheet_name="Output_Slacks")
            lambdas_df.to_excel(writer, sheet_name="Peer_Weights")
            recs_df.to_excel(writer, sheet_name="Recommendations", index=False)

        excel_data = output.getvalue()
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name="DEA_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
