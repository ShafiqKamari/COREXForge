
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="COREXForge — Contribution via Removal & Explained Variability for MCDM Weighting Method", layout="wide")

APP_TITLE = "COREXForge — Contribution via Removal & Explained Variability for MCDM Weighting Method"
CORE_TYPES = ["Benefit", "Cost", "Target"]

# -----------------------------
# COREX helpers
# -----------------------------
def normalize_column(x: pd.Series, ctype: str, target: float = None) -> pd.Series:
    xmin, xmax = x.min(), x.max()
    if ctype == "Benefit":
        r = (x - xmin) / (xmax - xmin)
    elif ctype == "Cost":
        r = (xmax - x) / (xmax - xmin)
    elif ctype == "Target":
        d = (x - target).abs()
        Dmax = d.max()
        r = 1.0 - d / Dmax
    else:
        raise ValueError("Unknown criterion type")
    return r.astype(float)

def corex_pipeline(df_vals: pd.DataFrame, crit_types: dict, targets: dict, alpha: float = 0.5):
    cols = list(df_vals.columns)
    n = len(cols)

    # Step 2: Normalization of the Decision Matrix
    R = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        t = crit_types[c]
        if t == "Target":
            R[c] = normalize_column(df_vals[c], t, targets.get(c, 0.0))
        else:
            R[c] = normalize_column(df_vals[c], t)

    # Step 3: The Overall Performance Score (P)
    P = R.sum(axis=1) / n

    # Step 4: The Performance Score under Criterion Removal (P_minus)
    row_sums = R.sum(axis=1)
    P_minus = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        P_minus[c] = (row_sums - R[c]) / n

    # Step 5: Removal Impact Score (Rj)
    D = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        D[c] = (P - P_minus[c]).abs()
    Rj = D.sum(axis=0).rename("RemovalImpact")

    # Step 6: The Standard Deviation of Each Criterion (sigma)
    sigma = R.std(axis=0, ddof=1).rename("Sigma")

    # Step 7: The Sum of Absolute Correlations for Each Criterion
    corr = R.corr(method="pearson").fillna(0.0)
    sum_abs_corr = corr.abs().sum(axis=1).rename("SumAbsCorr")

    # Step 8: Explained Variability Score (Vj)
    Vj = (sigma / sum_abs_corr).rename("ExplainedVariability")

    # Step 9: COREX Weight Scores (W)
    Rbar = (Rj / Rj.sum()).rename("Rbar")
    Vbar = (Vj / Vj.sum()).rename("Vbar")
    W = (alpha * Rbar + (1.0 - alpha) * Vbar).rename("Weight")
    W = W / W.sum()

    summary = pd.concat([Rj, Vj, Rbar, Vbar, W], axis=1)
    summary.index.name = "Criterion"

    return {
        "R": R, "P": P, "P_minus": P_minus, "Rj": Rj, "sigma": sigma,
        "sum_abs_corr": sum_abs_corr, "Vj": Vj, "W": W, "summary": summary
    }

# -----------------------------
# Sample dataset (with Alternative column)
# -----------------------------
def make_sample_dataset():
    data = {
        "Alternative": [f"A{i+1}" for i in range(5)],
        "Benefit1": [70, 85, 90, 60, 75],
        "Benefit2": [150, 140, 160, 155, 145],
        "Cost1": [200, 180, 220, 210, 190],
        "Cost2": [15, 12, 18, 14, 13],
        "Target1": [50, 55, 52, 48, 60],
    }
    return pd.DataFrame(data)

# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("Configuration")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    alpha = st.slider("Blend parameter α", 0.0, 1.0, 0.5, 0.05)
    st.caption("α = 1 removal-only • α = 0 redundancy-only")
    use_sample = st.checkbox("Use sample dataset", value=False)

raw_df = None
if use_sample:
    raw_df = make_sample_dataset()
else:
    if file is not None:
        if file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)

if raw_df is None:
    st.info("Upload a file or use the sample dataset.")
    st.stop()

# Step 1: The Decision Matrix (original)
st.subheader("Step 1: The Decision Matrix")
st.dataframe(raw_df, use_container_width=True)

# Row identifiers — auto-select Alternative if available
with st.expander("Row identifiers"):
    options = ["<row number>"] + list(raw_df.columns)
    default_index = 0
    if "Alternative" in raw_df.columns:
        default_index = options.index("Alternative")
    idx_col = st.selectbox("Use this column as alternative names", options, index=default_index)
if idx_col != "<row number>":
    raw_df = raw_df.set_index(idx_col)

# Criteria selection
num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found.")
    st.stop()

with st.expander("Select criteria columns"):
    selected_cols = st.multiselect("Criteria", options=num_cols, default=num_cols)
if not selected_cols:
    st.error("Select at least one criterion.")
    st.stop()

df_vals = raw_df[selected_cols].copy()

# Criterion types and targets
st.subheader("Criterion types and targets")
if "crit_meta_excel" not in st.session_state or set(st.session_state["crit_meta_excel"].index) != set(selected_cols):
    st.session_state["crit_meta_excel"] = pd.DataFrame(
        {"Type": ["Benefit"]*len(selected_cols), "Target": [0.0]*len(selected_cols)},
        index=selected_cols
    )

meta = st.data_editor(
    st.session_state["crit_meta_excel"],
    column_config={
        "Type": st.column_config.SelectboxColumn(options=CORE_TYPES),
        "Target": st.column_config.NumberColumn(format="%.6f"),
    },
    use_container_width=True
)
st.session_state["crit_meta_excel"] = meta
crit_types = meta["Type"].to_dict()
targets = meta["Target"].astype(float).to_dict()

# Check assumptions
for c in df_vals.columns:
    x = df_vals[c]
    if crit_types[c] in ["Benefit", "Cost"]:
        if np.isclose(x.min(), x.max()):
            st.error(f"Criterion {c} has max equal to min.")
            st.stop()
    else:
        d = (x - targets.get(c, 0.0)).abs()
        if np.isclose(d.max(), 0.0):
            st.error(f"All alternatives hit the exact target for {c}.")
            st.stop()

if st.button("Compute COREXForge (show 9 steps)"):
    A = corex_pipeline(df_vals, crit_types, targets, alpha=alpha)
    st.success("COREXForge computed. See steps below.")

    with st.expander("Step 2: Normalization of the Decision Matrix", expanded=True):
        st.dataframe(A["R"].style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 3: The Overall Performance Score", expanded=True):
        st.dataframe(A["P"].to_frame("P").style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 4: The Performance Score under Criterion Removal", expanded=False):
        st.dataframe(A["P_minus"].style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 5: Removal Impact Score", expanded=True):
        st.dataframe(A["Rj"].to_frame().style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 6: The Standard Deviation of Each Criterion", expanded=True):
        st.dataframe(A["sigma"].to_frame().style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 7: The Sum of Absolute Correlations for Each Criterion", expanded=False):
        st.dataframe(A["sum_abs_corr"].to_frame().style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 8: Explained Variability Score", expanded=False):
        st.dataframe(A["Vj"].to_frame().style.format("{:.6f}"), use_container_width=True)

    with st.expander("Step 9: COREX Weight Scores", expanded=True):
        st.dataframe(A["W"].to_frame("Weight").style.format("{:.6f}"), use_container_width=True)
        fig = px.bar(A["W"].reset_index(), x="index", y="Weight")
        fig.update_layout(xaxis_title="Criterion", yaxis_title="Weight", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary table")
    st.dataframe(A["summary"].style.format("{:.6f}"), use_container_width=True)

    # -------- Single Excel Export with all steps + summary --------
    st.subheader("Download all results")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Step 1: Decision Matrix (full, with chosen index)
        raw_df.to_excel(writer, sheet_name="Step1_DecisionMatrix")
        # Step 2
        A["R"].to_excel(writer, sheet_name="Step2_Normalized_R")
        # Step 3
        A["P"].to_frame("P").to_excel(writer, sheet_name="Step3_P")
        # Step 4
        A["P_minus"].to_excel(writer, sheet_name="Step4_P_minus")
        # Step 5
        A["Rj"].to_frame().to_excel(writer, sheet_name="Step5_RemovalImpact_Rj")
        # Step 6
        A["sigma"].to_frame().to_excel(writer, sheet_name="Step6_Sigma")
        # Step 7
        A["sum_abs_corr"].to_frame().to_excel(writer, sheet_name="Step7_SumAbsCorr")
        # Step 8
        A["Vj"].to_frame().to_excel(writer, sheet_name="Step8_ExplainedVar_Vj")
        # Step 9
        A["W"].to_frame("Weight").to_excel(writer, sheet_name="Step9_Weights")
        # Summary
        A["summary"].to_excel(writer, sheet_name="Summary")

    st.download_button(
        label="Download all results (Excel)",
        data=buffer.getvalue(),
        file_name="corexforge_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
