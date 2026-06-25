import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="C-FRPD — Causal Flight Recovery Prioritization Dashboard",
    page_icon="✈️", layout="wide"
)

st.markdown("""
<style>
.metric-card{background:white;border-radius:8px;padding:16px;
  border-left:4px solid #2E75B6;box-shadow:0 1px 4px rgba(0,0,0,0.1);margin-bottom:8px;}
.metric-value{font-size:28px;font-weight:bold;color:#1F3864;}
.metric-label{font-size:13px;color:#595959;margin-top:4px;}
.high-risk{border-left-color:#d62728!important;}
.low-risk{border-left-color:#2ca02c!important;}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GLOBAL_ATE = 30.57

CARRIER_TYPE_MAP = {
    "AA":"Mainline","DL":"Mainline","UA":"Mainline","AS":"Mainline","HA":"Mainline","US":"Mainline",
    "WN":"LCC","B6":"LCC","VX":"LCC",
    "NK":"ULCC","F9":"ULCC","G4":"ULCC",
    "OO":"Regional","YV":"Regional","MQ":"Regional","OH":"Regional","YX":"Regional",
    "9E":"Regional","EV":"Regional","QX":"Regional",
}
CARRIER_NAMES = {
    "AA":"American","DL":"Delta","UA":"United","AS":"Alaska","HA":"Hawaiian",
    "WN":"Southwest","B6":"JetBlue","NK":"Spirit","F9":"Frontier","G4":"Allegiant",
    "OO":"SkyWest","YX":"Republic","9E":"Endeavor","MQ":"Envoy","OH":"PSA","YV":"Mesa",
}
TYPE_COLORS   = {"Mainline":"#1f77b4","LCC":"#2ca02c","ULCC":"#d62728","Regional":"#ff7f0e"}
TIME_ORDER    = ["Early morning (5-9)","Midday (10-14)","Afternoon (15-19)","Evening/night (20+)"]
TIER_ORDER    = ["Large","Medium","Small","NonHub"]
ROUTE_ORDER   = ["Hub-to-Hub","Hub-to-Spoke","Spoke-to-Spoke"]
METHOD_COLORS = {
    "C-FRPD (GA-calibrated)":    "#2ca02c",
    "SAW (data-driven, pre-GA)": "#1f77b4",
    "Longest delay first":       "#ff7f0e",
    "Earliest dep first":        "#9467bd",
    "Highest spillover":         "#8c564b",
    "Hub first":                 "#d62728",
    "CATE only":                 "#17becf",
    "Random":                    "#7f7f7f",
}
DISAGREE = {
    "AA":(32.72,45.62,"+12.90"),
    "DL":(34.32,46.68,"+12.36"),
    "UA":(31.84,42.09,"+10.25"),
    "AS":(32.90,49.08,"+16.18"),
    "WN":(34.88,47.38,"+12.49"),
}

# ── GA-calibrated weights (from Notebook 8, corr-seed arm) ───────────────────
GA_WEIGHTS = {
    "Global (GA-calibrated)": {
        "CATE":     0.3484,
        "SPILLOVER":0.0058,
        "STRATEGIC":0.0347,
        "WINDOW":   0.4128,
        "HIST_RATE":0.1983,
    },
    "Carrier: Mainline": {
        "CATE":     0.3441,
        "SPILLOVER":0.0520,
        "STRATEGIC":0.0219,
        "WINDOW":   0.4464,
        "HIST_RATE":0.1355,
    },
    "Carrier: LCC": {
        "CATE":     0.0959,
        "SPILLOVER":0.3305,
        "STRATEGIC":0.0055,
        "WINDOW":   0.5682,
        "HIST_RATE":0.0000,
    },
    "Carrier: ULCC": {
        "CATE":     0.3506,
        "SPILLOVER":0.0349,
        "STRATEGIC":0.0216,
        "WINDOW":   0.5899,
        "HIST_RATE":0.0030,
    },
    "Carrier: Regional": {
        "CATE":     0.0000,
        "SPILLOVER":0.0099,
        "STRATEGIC":0.0000,
        "WINDOW":   0.9308,
        "HIST_RATE":0.0593,
    },
    "Hub status: Hub-to-Hub": {
        "CATE":     0.4067,
        "SPILLOVER":0.0008,
        "STRATEGIC":0.0034,
        "WINDOW":   0.4410,
        "HIST_RATE":0.1481,
    },
    "Hub status: Hub-to-Spoke": {
        "CATE":     0.0702,
        "SPILLOVER":0.3161,
        "STRATEGIC":0.1224,
        "WINDOW":   0.3985,
        "HIST_RATE":0.0929,
    },
    "Hub status: Spoke-to-Spoke": {
        "CATE":     0.4117,
        "SPILLOVER":0.0007,
        "STRATEGIC":0.0001,
        "WINDOW":   0.5871,
        "HIST_RATE":0.0003,
    },
    "Time: Afternoon Peak (15-19)": {
        "CATE":     0.1336,
        "SPILLOVER":0.1060,
        "STRATEGIC":0.0152,
        "WINDOW":   0.6647,
        "HIST_RATE":0.0804,
    },
    "Time: Evening/Night (19+)": {
        "CATE":     0.0720,
        "SPILLOVER":0.4398,
        "STRATEGIC":0.0442,
        "WINDOW":   0.3084,
        "HIST_RATE":0.1356,
    },
    "Time: Midday (11-15)": {
        "CATE":     0.2931,
        "SPILLOVER":0.0244,
        "STRATEGIC":0.0072,
        "WINDOW":   0.1772,
        "HIST_RATE":0.4981,
    },
    "Time: Morning (5-11)": {
        "CATE":     0.3411,
        "SPILLOVER":0.0001,
        "STRATEGIC":0.0000,
        "WINDOW":   0.0009,
        "HIST_RATE":0.6580,
    },
}

# Pre-GA data-driven weights (kept for comparison in Tab 3)
DATA_DRIVEN_WEIGHTS = {
    "CATE":     0.0613,
    "SPILLOVER":0.3191,
    "STRATEGIC":0.0212,
    "WINDOW":   0.5204,
    "HIST_RATE":0.0780,
}

WEIGHTS_TABLE = pd.DataFrame([
    {"Policy": "Global (GA-calibrated)", "CATE": 0.348, "Spillover": 0.006, "Route value": 0.035,
     "Recovery window": 0.413, "Hist. rate": 0.198},
    {"Policy": "Data-driven (pre-GA)",   "CATE": 0.061, "Spillover": 0.319, "Route value": 0.021,
     "Recovery window": 0.520, "Hist. rate": 0.078},
])

CARRIER_DATA = pd.DataFrame([
    {"Carrier":"AA","Type":"Mainline","CATE":32.72,"CI_LB":24.1,"CI_UB":41.3,"Diff":2.15},
    {"Carrier":"DL","Type":"Mainline","CATE":34.32,"CI_LB":25.8,"CI_UB":42.8,"Diff":3.75},
    {"Carrier":"UA","Type":"Mainline","CATE":31.84,"CI_LB":23.2,"CI_UB":40.4,"Diff":1.27},
    {"Carrier":"AS","Type":"Mainline","CATE":32.90,"CI_LB":24.3,"CI_UB":41.5,"Diff":2.33},
    {"Carrier":"HA","Type":"Mainline","CATE":39.64,"CI_LB":31.0,"CI_UB":48.3,"Diff":9.07},
    {"Carrier":"WN","Type":"LCC",     "CATE":34.88,"CI_LB":26.4,"CI_UB":43.4,"Diff":4.31},
    {"Carrier":"B6","Type":"LCC",     "CATE":38.39,"CI_LB":29.8,"CI_UB":47.0,"Diff":7.82},
    {"Carrier":"NK","Type":"ULCC",    "CATE":39.73,"CI_LB":31.1,"CI_UB":48.4,"Diff":9.16},
    {"Carrier":"F9","Type":"ULCC",    "CATE":39.52,"CI_LB":30.9,"CI_UB":48.1,"Diff":8.95},
    {"Carrier":"G4","Type":"ULCC",    "CATE":42.66,"CI_LB":34.0,"CI_UB":51.3,"Diff":12.09},
    {"Carrier":"OO","Type":"Regional","CATE":43.91,"CI_LB":35.3,"CI_UB":52.5,"Diff":13.34},
    {"Carrier":"YX","Type":"Regional","CATE":40.96,"CI_LB":32.4,"CI_UB":49.5,"Diff":10.39},
    {"Carrier":"9E","Type":"Regional","CATE":44.02,"CI_LB":35.4,"CI_UB":52.6,"Diff":13.45},
    {"Carrier":"MQ","Type":"Regional","CATE":43.14,"CI_LB":34.5,"CI_UB":51.8,"Diff":12.57},
    {"Carrier":"OH","Type":"Regional","CATE":43.25,"CI_LB":34.6,"CI_UB":51.9,"Diff":12.68},
    {"Carrier":"YV","Type":"Regional","CATE":41.32,"CI_LB":32.7,"CI_UB":49.9,"Diff":10.75},
])

# ── Verified holdout results K=1-20 (per-scenario normalization, 2024-25 holdout) ──
# Source: notebook8_results/global_ga_k_full_1_20.csv
HOLDOUT_K_TABLE = {
    #  K:  (GA%,   PreGA%,  BestBench%,  BestBenchName,          Protected)
    1:  ( 3.60,   3.41,    3.46,  "Longest delay first",    38647),
    2:  ( 7.34,   7.05,    7.16,  "Longest delay first",    78849),
    3:  (11.06,  10.70,   10.77,  "Longest delay first",   118875),
    4:  (14.35,  13.94,   13.94,  "Highest spillover",     154224),
    5:  (17.30,  16.85,   16.82,  "Highest spillover",     185943),
    6:  (19.99,  19.52,   19.47,  "Earliest dep first",    214872),
    7:  (22.44,  21.97,   21.95,  "Earliest dep first",    241205),
    8:  (24.73,  24.26,   24.26,  "Earliest dep first",    265777),
    9:  (26.86,  26.39,   26.42,  "Earliest dep first",    288693),
    10: (28.85,  28.40,   28.44,  "Earliest dep first",    310090),
    11: (30.73,  30.30,   30.34,  "Earliest dep first",    330344),
    12: (32.50,  32.09,   32.14,  "Earliest dep first",    349300),
    13: (34.17,  33.77,   33.84,  "Earliest dep first",    367274),
    14: (35.77,  35.39,   35.45,  "Earliest dep first",    384479),
    15: (37.28,  36.93,   36.99,  "Earliest dep first",    400725),
    16: (38.73,  38.40,   38.48,  "Earliest dep first",    416271),
    17: (40.11,  39.81,   39.89,  "Earliest dep first",    431077),
    18: (41.41,  41.16,   41.25,  "Earliest dep first",    445121),
    19: (42.69,  42.47,   42.56,  "Earliest dep first",    458864),
    20: (43.91,  43.72,   43.81,  "Earliest dep first",    472012),
}

# ── Verified benchmark curves K=1-20 ─────────────────────────────────────────
# Source: captured_rate_sk() on holdout 2024-25, per-scenario normalization
BENCHMARK_METHODS_ALL = {
    "C-FRPD (GA-calibrated)":    [HOLDOUT_K_TABLE[k][0] for k in range(1, 21)],
    "SAW (data-driven, pre-GA)": [HOLDOUT_K_TABLE[k][1] for k in range(1, 21)],
    "Longest delay first":  [ 3.46, 7.16,10.77,13.92,16.72,19.27,21.60,23.75,25.74,27.60,29.34,31.00,32.58,34.04,35.46,36.80,38.10,39.32,40.51,41.66],
    "Earliest dep first":   [ 3.33, 6.93,10.60,13.84,16.77,19.47,21.95,24.26,26.42,28.44,30.34,32.14,33.84,35.45,36.99,38.48,39.89,41.25,42.56,43.81],
    "Highest spillover":    [ 3.44, 7.06,10.73,13.94,16.82,19.44,21.85,24.07,26.15,28.10,29.93,31.67,33.30,34.83,36.31,37.71,39.07,40.37,41.62,42.80],
    "CATE only":            [ 3.33, 6.61, 9.91,12.75,15.28,17.58,19.68,21.61,23.41,25.08,26.68,28.18,29.60,30.94,32.23,33.47,34.65,35.80,36.90,37.96],
    "Random":               [ 3.39, 6.77,10.17,13.10,15.71,18.07,20.24,22.24,24.09,25.81,27.44,28.99,30.43,31.80,33.13,34.37,35.58,36.74,37.86,38.94],
    "Hub first":            [ 3.19, 6.46, 9.79,12.70,15.31,17.67,19.84,21.83,23.68,25.43,27.07,28.62,30.08,31.48,32.80,34.06,35.28,36.45,37.57,38.66],
}

# ── SAW scoring (per-scenario normalization, matches Notebook 8) ──────────────
def saw_score(features_matrix, weights_dict, eps=1e-9):
    """
    Simple Additive Weighting (SAW) priority score per the C-FRPD methodology.
    features_matrix: np.array shape (n_flights, 5) columns in order:
        [CATE, SPILLOVER, STRATEGIC, WINDOW, HIST_RATE]
    weights_dict: dict with keys CATE, SPILLOVER, STRATEGIC, WINDOW, HIST_RATE
    Returns: priority score array (higher = higher priority)
    """
    X = features_matrix.astype(float)
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        rng = col.max() - col.min()
        X_norm[:, j] = (col - col.min()) / (rng + eps)
    w = np.array([
        weights_dict["CATE"],
        weights_dict["SPILLOVER"],
        weights_dict["STRATEGIC"],
        weights_dict["WINDOW"],
        weights_dict["HIST_RATE"],
    ])
    return X_norm @ w

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    return {
        "ct":       pd.read_csv("dash_cate_by_carrier_type.csv"),
        "rt":       pd.read_csv("dash_cate_by_route_type.csv"),
        "td":       pd.read_csv("dash_cate_by_time.csv"),
        "ot":       pd.read_csv("dash_cate_by_tier.csv"),
        "ct_td":    pd.read_csv("dash_cate_carrier_time.csv"),
        "carr_sum": pd.read_csv("dash_carrier_summary.csv"),
        "carr_heat":pd.read_csv("dash_carrier_heatmap.csv"),
        "carr_reg": pd.read_csv("dash_carrier_regime.csv"),
        "bench":    pd.read_csv("dash_benchmark_full.csv"),
        "queue":    pd.read_csv("dash_top_queue.csv"),
    }

with st.spinner("Loading C-FRPD dashboard..."):
    D = load_all()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ C-FRPD — Causal Flight Recovery Prioritization Dashboard")
st.markdown(
    "**GA-calibrated causal DSS** for U.S. domestic flight disruption recovery | "
    "66.5M BTS flights (2015–2025) | "
    "[Paper: Decision Support Systems (under review)]"
)

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl, cls in zip(
    [c1, c2, c3, c4],
    ["30.57 pp", "11.06%", "118,875", "13 / 16"],
    ["Global causal ATE [21.76, 39.38]",
     "Downstream disruptions captured (K=3, holdout)",
     "Downstream flights protected (K=3, holdout)",
     "Context policies validated (holdout)"],
    ["", "low-risk", "low-risk", "low-risk"]
):
    with col:
        st.markdown(
            f'<div class="metric-card {cls}">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>', unsafe_allow_html=True
        )

with st.expander("ℹ️ About this dashboard", expanded=False):
    st.markdown("""
**What is the C-FRPD?**

The Causal Flight Recovery Prioritization Dashboard (C-FRPD) is a causal, metaheuristic, and
multi-criteria decision support system for airline disruption recovery, built from 66.5 million
BTS On-Time Performance records (2015–2025).

**What is CATE?**
CATE stands for Conditional Average Treatment Effect — it measures how much a prior flight
disruption on the same aircraft rotation *causally* increases the probability that the next
flight is also disrupted. A CATE of 30.57 pp means a prior disruption raises the next flight's
disruption probability by 30.57 percentage points (from a baseline of ~19% to ~49%). Estimated
using a causal forest with airport-day confounding controls, validated by three placebo tests.

**What is the SAW priority score?**
Simple Additive Weighting (SAW) ranks disrupted flights using five criteria: CATE propagation risk,
downstream spillover, route strategic value, recovery window, and historical disruption rate.
Each criterion is min-max normalized per scenario, then weighted and summed. Weights are
calibrated by a genetic algorithm (GA) to maximize captured downstream disruptions. The GA
was seeded from two strategies (correlation-derived and causal-magnitude-derived) and both
converged to near-identical final weights, confirming a robust optimum.

**The four tabs:**
- **🌐 Global Causal Intelligence** — System-wide propagation effects and causal identification chain.
- **🏢 Airline Command View** — Carrier-specific CATE vs. system average; where global models mislead.
- **🚑 C-FRPD Prioritization Engine** — GA-calibrated SAW ranking vs. 6 benchmarks at any K.
- **🚨 Live Flight Triage** — Enter today's disrupted flights and get a real-time SAW priority ranking.

**Data source:** BTS On-Time Performance Database. Models trained on 2015–2019.
Evaluated on 2022–2025 Recovery period (entirely out of sample).
""")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Global Causal Intelligence",
    "🏢 Airline Command View",
    "🚑 C-FRPD Prioritization Engine",
    "🚨 Live Flight Triage",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Global Causal Intelligence
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Global Causal Intelligence")
    st.markdown(
        "System-wide causal propagation effects. "
        "Treatment T1: prior flight on same aircraft rotation was disrupted."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("CATE by Carrier Type")
        ct = D["ct"].sort_values("mean_cate")
        fig = go.Figure([go.Bar(
            y=ct["CARRIER_TYPE"].tolist(),
            x=(ct["mean_cate"]*100).tolist(),
            orientation="h",
            marker_color=[TYPE_COLORS.get(c,"#7f7f7f") for c in ct["CARRIER_TYPE"]],
            error_x=dict(type="data",
                array=[(ub-m)*100 for m,ub in zip(ct["mean_cate"],ct["ub"])],
                arrayminus=[(m-lb)*100 for m,lb in zip(ct["mean_cate"],ct["lb"])],
                visible=True)
        )])
        fig.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black",
                      annotation_text=f"Global {GLOBAL_ATE} pp")
        fig.update_layout(showlegend=False, height=280, xaxis_title="CATE (pp)",
                          margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("CATE by Route Type")
        rt = D["rt"].sort_values("mean_cate")
        rc = {"Hub-to-Hub":"#1f77b4","Hub-to-Spoke":"#ff7f0e","Spoke-to-Spoke":"#d62728"}
        fig2 = go.Figure([go.Bar(
            y=rt["ROUTE_TYPE"].tolist(),
            x=(rt["mean_cate"]*100).tolist(),
            orientation="h",
            marker_color=[rc.get(r,"#7f7f7f") for r in rt["ROUTE_TYPE"]],
            error_x=dict(type="data",
                array=[(ub-m)*100 for m,ub in zip(rt["mean_cate"],rt["ub"])],
                arrayminus=[(m-lb)*100 for m,lb in zip(rt["mean_cate"],rt["lb"])],
                visible=True)
        )])
        fig2.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig2.update_layout(showlegend=False, height=280, xaxis_title="CATE (pp)",
                           margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("CATE by Time of Day")
        td = D["td"].set_index("TIME_BIN").reindex(TIME_ORDER)
        fig3 = go.Figure(go.Bar(
            x=td.index.tolist(), y=(td["mean_cate"]*100).tolist(),
            marker_color=["#aec7e8","#ffbb78","#d62728","#9467bd"],
            text=[f"{v:.1f} pp" for v in td["mean_cate"]*100], textposition="outside"
        ))
        fig3.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global ATE")
        fig3.update_layout(height=280, yaxis_title="CATE (pp)",
                           margin=dict(l=0,r=20,t=20,b=80),
                           plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("CATE by Origin Airport Tier")
        ot = D["ot"].set_index("ORIGIN_TIER").reindex(TIER_ORDER)
        fig4 = go.Figure(go.Bar(
            x=ot.index.tolist(), y=(ot["mean_cate"]*100).tolist(),
            marker_color=["#aec7e8","#ffbb78","#98df8a","#ff9896"],
            text=[f"{v:.1f} pp" for v in ot["mean_cate"]*100], textposition="outside"
        ))
        fig4.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig4.update_layout(height=280, yaxis_title="CATE (pp)",
                           margin=dict(l=0,r=20,t=20,b=40),
                           plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Causal Identification Chain")
    id_labels = [
        "Naive difference",
        "V1 causal (no airport ctrl)",
        "V2 causal (with airport ctrl) ← headline",
        "Placebo V1 (future flight — contaminated)",
        "Placebo V2 (other tail, same ahd)",
        "Placebo V3 (other tail, same carrier+ahd)",
    ]
    id_vals   = [38.01, 36.82, 30.57, 26.46, 3.28, 6.02]
    id_colors = ["#d62728","#ff7f0e","#2ca02c","#9467bd","#1f77b4","#17becf"]
    fig5 = go.Figure(go.Bar(
        y=id_labels[::-1], x=id_vals[::-1], orientation="h",
        marker_color=id_colors[::-1],
        text=[f"{v:.2f} pp" for v in id_vals[::-1]], textposition="outside"
    ))
    fig5.add_vline(x=5, line_dash="dot", line_color="gray",
                   annotation_text="5 pp threshold")
    fig5.update_layout(height=340, xaxis_title="Average Treatment Effect (pp)",
                       margin=dict(l=0,r=90,t=20,b=40),
                       plot_bgcolor="white", showlegend=False, xaxis_range=[0,46])
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Placebo V2 (3.28 pp) and V3 (6.02 pp) near zero — rotation mechanism confirmed. "
        "All heterogeneity differences significant at p<0.001. "
        "94.5% of individual flight CATEs statistically significant at 95% level."
    )

    st.subheader("CATE Heatmap — Carrier Type × Time of Day")
    ct_td = D["ct_td"].pivot(index="CARRIER_TYPE", columns="TIME_BIN", values="mean_cate")
    ct_td = ct_td.reindex(columns=[t for t in TIME_ORDER if t in ct_td.columns]) * 100
    fig6 = go.Figure(go.Heatmap(
        z=ct_td.values.tolist(),
        x=[c.split("(")[0].strip() for c in ct_td.columns],
        y=ct_td.index.tolist(),
        colorscale="RdYlGn_r",
        text=[[f"{v:.1f} pp" for v in row] for row in ct_td.values],
        texttemplate="%{text}",
        colorbar=dict(title="CATE (pp)"), zmin=10, zmax=55
    ))
    fig6.update_layout(height=260, margin=dict(l=0,r=0,t=20,b=60))
    st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Airline Command View
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Airline Command View")
    st.markdown(
        "Carrier-specific CATE vs system average. "
        "Reveals where global rankings mislead recovery decisions."
    )
    carrier_list = sorted(CARRIER_DATA["Carrier"].tolist())
    selected = st.selectbox(
        "Select carrier", carrier_list,
        format_func=lambda x: f"{x} — {CARRIER_NAMES.get(x,x)} ({CARRIER_TYPE_MAP.get(x,'')})"
    )
    row  = CARRIER_DATA[CARRIER_DATA["Carrier"]==selected].iloc[0]
    diff = row["CATE"] - GLOBAL_ATE
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cls = "high-risk" if row["CATE"] > GLOBAL_ATE+3 else "low-risk"
        st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{row["CATE"]:.2f} pp</div><div class="metric-label">Carrier CATE (T1)</div></div>', unsafe_allow_html=True)
    with c2:
        sign = "+" if diff >= 0 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sign}{diff:.2f} pp</div><div class="metric-label">vs Global ATE ({GLOBAL_ATE} pp)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{row["CI_LB"]:.1f}–{row["CI_UB"]:.1f}</div><div class="metric-label">95% CI (pp)</div></div>', unsafe_allow_html=True)
    with c4:
        frag = "HIGH ⚠️" if diff > 3 else ("LOW ✅" if diff < -3 else "Average")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{frag}</div><div class="metric-label">{row["Type"]} carrier</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("All Carriers — CATE Ranking")
        cd = CARRIER_DATA.sort_values("CATE")
        fig7 = go.Figure(go.Bar(
            y=cd["Carrier"].tolist(), x=cd["CATE"].tolist(), orientation="h",
            marker_color=["#d62728" if c==selected else TYPE_COLORS.get(CARRIER_TYPE_MAP.get(c,""),"#7f7f7f") for c in cd["Carrier"]],
            text=[f"{v:.1f}" for v in cd["CATE"]], textposition="outside"
        ))
        fig7.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black", annotation_text="Global")
        fig7.update_layout(height=480, xaxis_title="CATE (pp)", margin=dict(l=0,r=60,t=20,b=40), plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)
    with col_r:
        st.subheader(f"{selected} — CATE by Route × Time")
        ch = D["carr_heat"]
        carr_ch = ch[ch["carrier"]==selected]
        if len(carr_ch) > 0:
            pivot = carr_ch.pivot(index="route_type", columns="time_bin", values="mean_cate")
            pivot = pivot.reindex(
                index=[r for r in ROUTE_ORDER if r in pivot.index],
                columns=[t for t in TIME_ORDER if t in pivot.columns]
            ) * 100
            fig8 = go.Figure(go.Heatmap(
                z=pivot.values.tolist(),
                x=[c.split("(")[0].strip() for c in pivot.columns],
                y=pivot.index.tolist(), colorscale="RdYlGn_r",
                text=[[f"{v:.1f} pp" if not np.isnan(v) else "n/a" for v in r] for r in pivot.values],
                texttemplate="%{text}", colorbar=dict(title="CATE (pp)"), zmin=10, zmax=55
            ))
            fig8.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=60))
            st.plotly_chart(fig8, use_container_width=True)
            st.caption("Red = high propagation risk. Above 38 pp warrants priority protocols.")
        else:
            st.info(f"No heatmap data for {selected}.")
    st.subheader("Where the Global View Misleads")
    if selected in DISAGREE:
        overall, aft_h2s, delta = DISAGREE[selected]
        st.warning(
            f"**{selected} ({CARRIER_NAMES.get(selected,selected)})** overall CATE: "
            f"**{overall:.2f} pp** — near average globally. "
            f"But on **afternoon hub-to-spoke routes**: **{aft_h2s:.2f} pp** "
            f"({delta} pp above own average, p<0.001). "
            f"A global recovery model would miss this vulnerability entirely."
        )
    else:
        st.info("Select AA, DL, UA, AS, or WN to see the global vs airline-specific disagreement.")
    st.subheader(f"{selected} — CATE across Regimes")
    cr = D["carr_reg"]
    carr_reg = cr[cr["OP_UNIQUE_CARRIER"]==selected]
    if len(carr_reg) > 0:
        fig9 = go.Figure(go.Bar(
            x=carr_reg["REGIME"].tolist(),
            y=(carr_reg["CATE_T1"]*100).tolist(),
            marker_color=["#1f77b4","#2ca02c"],
            text=[f"{v:.2f} pp" for v in carr_reg["CATE_T1"]*100],
            textposition="outside"
        ))
        fig9.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black", annotation_text="Global ATE")
        fig9.update_layout(height=250, yaxis_title="CATE (pp)", margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig9, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — C-FRPD Prioritization Engine
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("C-FRPD Prioritization Engine")
    st.markdown(
        "GA-calibrated SAW ranking evaluated on 5.04M disrupted flights (2022–2025 holdout). "
        "13 of 16 context-specific policies outperform the strongest operational benchmark."
    )

    col1, col2 = st.columns(2)
    with col1:
        k_val = st.slider("Intervention budget K (user-specified)", 1, 20, 3,
                          help="Number of flights to intervene on per airport-date scenario. "
                               "Set this to your actual available recovery capacity.")
    with col2:
        ct_filter = st.selectbox("Filter by carrier type", ["All","Mainline","LCC","ULCC","Regional"])

    # ── Metric cards for selected K (All carrier type only) ───────────────────
    if ct_filter == "All":
        ga_pct, prega_pct, best_pct, best_name, protected = HOLDOUT_K_TABLE[k_val]
        gap = round(ga_pct - best_pct, 2)
        sign = "+" if gap >= 0 else ""
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            cls = "low-risk" if gap >= 0 else "high-risk"
            st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{ga_pct:.2f}%</div><div class="metric-label">C-FRPD captured (K={k_val})</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sign}{gap:.2f} pp</div><div class="metric-label">vs best benchmark ({best_name})</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card low-risk"><div class="metric-value">{protected:,}</div><div class="metric-label">Downstream flights protected</div></div>', unsafe_allow_html=True)
        
