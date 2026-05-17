# Generate final corrected app.py
app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Flight Disruption Causal DSS",
    page_icon="✈️", layout="wide"
)
st.markdown("""
<style>
.metric-card {
    background:white; border-radius:8px; padding:16px;
    border-left:4px solid #2E75B6;
    box-shadow:0 1px 4px rgba(0,0,0,0.1); margin-bottom:8px;
}
.metric-value{font-size:28px;font-weight:bold;color:#1F3864;}
.metric-label{font-size:13px;color:#595959;margin-top:4px;}
.high-risk{border-left-color:#d62728!important;}
.low-risk{border-left-color:#2ca02c!important;}
</style>
""", unsafe_allow_html=True)

GLOBAL_ATE = 30.57
CARRIER_TYPE_MAP = {
    "AA":"Mainline","DL":"Mainline","UA":"Mainline","AS":"Mainline","HA":"Mainline",
    "WN":"LCC","B6":"LCC","VX":"LCC",
    "NK":"ULCC","F9":"ULCC","G4":"ULCC",
    "OO":"Regional","YV":"Regional","MQ":"Regional","OH":"Regional",
    "YX":"Regional","9E":"Regional","EV":"Regional","QX":"Regional",
}
CARRIER_NAMES = {
    "AA":"American","DL":"Delta","UA":"United","AS":"Alaska","HA":"Hawaiian",
    "WN":"Southwest","B6":"JetBlue","NK":"Spirit","F9":"Frontier",
    "G4":"Allegiant","OO":"SkyWest","YX":"Republic","9E":"Endeavor",
    "MQ":"Envoy","OH":"PSA","YV":"Mesa",
}
TYPE_COLORS = {"Mainline":"#1f77b4","LCC":"#2ca02c","ULCC":"#d62728","Regional":"#ff7f0e"}
TIME_ORDER  = ["Early morning (5-9)","Midday (10-14)","Afternoon (15-19)","Evening/night (20+)"]
TIER_ORDER  = ["Large","Medium","Small","NonHub"]
ROUTE_ORDER = ["Hub-to-Hub","Hub-to-Spoke","Spoke-to-Spoke"]
METHOD_COLORS = {
    "TOPSIS (DSS)":"#2ca02c","Longest delay first":"#ff7f0e",
    "Earliest dep first":"#1f77b4","Highest spillover":"#9467bd",
    "Hub first":"#d62728","CATE only":"#8c564b","Random":"#7f7f7f",
}
DISAGREE = {
    "AA":(32.72,45.62,"+12.90"),"DL":(34.32,46.68,"+12.36"),
    "UA":(31.84,42.09,"+10.25"),"AS":(32.90,49.08,"+16.18"),
    "WN":(34.88,47.38,"+12.49"),
}
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
WEIGHTS_DATA = pd.DataFrame([
    {"Criterion":"Recovery window","Weight (%)":52.04},
    {"Criterion":"Downstream spillover","Weight (%)":31.91},
    {"Criterion":"Historical disr rate","Weight (%)":7.80},
    {"Criterion":"CATE propagation risk","Weight (%)":6.13},
    {"Criterion":"Route strategic value","Weight (%)":2.12},
])

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

with st.spinner("Loading..."):
    D = load_all()

# Header
st.title("✈️ Flight Disruption Causal Decision Support System")
st.markdown("**Causal propagation analysis** of 66.5M U.S. domestic flights (2015–2025) | BTS")
c1,c2,c3,c4 = st.columns(4)
metrics = [
    ("30.57 pp","Global causal ATE [21.76, 39.38]",""),
    ("49.51%","Downstream disruption rate","high-risk"),
    ("66.5M","Flights analyzed (2015–2025)",""),
    ("0.81","Predictive model AUC","low-risk"),
]
for col,(val,lbl,cls) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(
            f'<div class="metric-card {cls}"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True)
st.markdown("---")

tab1,tab2,tab3 = st.tabs([
    "🌐  Global Causal Intelligence",
    "🏢  Airline Command View",
    "🚑  Recovery Prioritization Engine"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Global Causal Intelligence")

    def err_bar(df, mean_col, lb_col, ub_col):
        """Build error_x dict compatible with plotly v6."""
        return dict(
            type="data",
            array=[(row[ub_col]-row[mean_col])*100 for _,row in df.iterrows()],
            arrayminus=[(row[mean_col]-row[lb_col])*100 for _,row in df.iterrows()],
            visible=True
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
            error_x=dict(
                type="data",
                array=[(ub-m)*100 for m,ub in zip(ct["mean_cate"],ct["ub"])],
                arrayminus=[(m-lb)*100 for m,lb in zip(ct["mean_cate"],ct["lb"])],
                visible=True
            )
        )])
        fig.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black",
                      annotation_text=f"Global {GLOBAL_ATE} pp")
        fig.update_layout(showlegend=False, height=280,
            xaxis_title="CATE (pp)", margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white")
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("CATE by Route Type")
        rt = D["rt"].sort_values("mean_cate")
        route_colors = {"Hub-to-Hub":"#1f77b4","Hub-to-Spoke":"#ff7f0e","Spoke-to-Spoke":"#d62728"}
        fig2 = go.Figure([go.Bar(
            y=rt["ROUTE_TYPE"].tolist(),
            x=(rt["mean_cate"]*100).tolist(),
            orientation="h",
            marker_color=[route_colors.get(r,"#7f7f7f") for r in rt["ROUTE_TYPE"]],
            error_x=dict(
                type="data",
                array=[(ub-m)*100 for m,ub in zip(rt["mean_cate"],rt["ub"])],
                arrayminus=[(m-lb)*100 for m,lb in zip(rt["mean_cate"],rt["lb"])],
                visible=True
            )
        )])
        fig2.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig2.update_layout(showlegend=False, height=280,
            xaxis_title="CATE (pp)", margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white")
        st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("CATE by Time of Day")
        td = D["td"].set_index("TIME_BIN").reindex(TIME_ORDER)
        fig3 = go.Figure(go.Bar(
            x=td.index.tolist(), y=(td["mean_cate"]*100).tolist(),
            marker_color=["#aec7e8","#ffbb78","#d62728","#9467bd"],
            text=[f"{v:.1f} pp" for v in td["mean_cate"]*100],
            textposition="outside"
        ))
        fig3.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global ATE")
        fig3.update_layout(height=280, yaxis_title="CATE (pp)",
            margin=dict(l=0,r=20,t=20,b=80),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig3, width='stretch')

    with col_d:
        st.subheader("CATE by Origin Airport Tier")
        ot = D["ot"].set_index("ORIGIN_TIER").reindex(TIER_ORDER)
        fig4 = go.Figure(go.Bar(
            x=ot.index.tolist(), y=(ot["mean_cate"]*100).tolist(),
            marker_color=["#aec7e8","#ffbb78","#98df8a","#ff9896"],
            text=[f"{v:.1f} pp" for v in ot["mean_cate"]*100],
            textposition="outside"
        ))
        fig4.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig4.update_layout(height=280, yaxis_title="CATE (pp)",
            margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig4, width='stretch')

    st.subheader("Causal Identification Chain")
    id_labels = [
        "Naive difference",
        "V1 causal (no airport ctrl)",
        "V2 causal (with airport ctrl) ← headline",
        "Placebo V1 (future flight — contaminated)",
        "Placebo V2 (other tail, same ahd)",
        "Placebo V3 (other tail, same carrier+ahd)"
    ]
    id_vals   = [38.01, 36.82, 30.57, 26.46, 3.28, 6.02]
    id_colors = ["#d62728","#ff7f0e","#2ca02c","#9467bd","#1f77b4","#17becf"]
    fig5 = go.Figure(go.Bar(
        y=id_labels[::-1], x=id_vals[::-1], orientation="h",
        marker_color=id_colors[::-1],
        text=[f"{v:.2f} pp" for v in id_vals[::-1]],
        textposition="outside"
    ))
    fig5.add_vline(x=5, line_dash="dot", line_color="gray",
                   annotation_text="5 pp threshold")
    fig5.update_layout(height=340,
        xaxis_title="Average Treatment Effect (pp)",
        margin=dict(l=0,r=90,t=20,b=40),
        plot_bgcolor="white", showlegend=False, xaxis_range=[0,46])
    st.plotly_chart(fig5, width='stretch')
    st.caption(
        "Placebo V2 (3.28 pp) and V3 (6.02 pp) near zero — rotation mechanism confirmed. "
        "All heterogeneity differences significant at p<0.001."
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
    st.plotly_chart(fig6, width='stretch')

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Airline Command View")
    st.markdown("Carrier-specific CATE vs system average. Reveals where global rankings mislead.")

    carrier_list = sorted(CARRIER_DATA["Carrier"].tolist())
    selected = st.selectbox("Select carrier", carrier_list,
        format_func=lambda x: f"{x} — {CARRIER_NAMES.get(x,x)} ({CARRIER_TYPE_MAP.get(x,'')})")
    row = CARRIER_DATA[CARRIER_DATA["Carrier"]==selected].iloc[0]
    diff = row["CATE"] - GLOBAL_ATE

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        cls = "high-risk" if row["CATE"]>GLOBAL_ATE+3 else "low-risk"
        st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{row["CATE"]:.2f} pp</div>'
                    f'<div class="metric-label">Carrier CATE (T1)</div></div>', unsafe_allow_html=True)
    with c2:
        sign = "+" if diff>=0 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sign}{diff:.2f} pp</div>'
                    f'<div class="metric-label">vs Global ATE ({GLOBAL_ATE} pp)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{row["CI_LB"]:.1f}–{row["CI_UB"]:.1f}</div>'
                    f'<div class="metric-label">95% CI (pp)</div></div>', unsafe_allow_html=True)
    with c4:
        frag = "HIGH ⚠️" if diff>3 else ("LOW ✅" if diff<-3 else "Average")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{frag}</div>'
                    f'<div class="metric-label">{row["Type"]} carrier</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("All Carriers — CATE Ranking")
        cd = CARRIER_DATA.sort_values("CATE")
        fig7 = go.Figure(go.Bar(
            y=cd["Carrier"].tolist(), x=cd["CATE"].tolist(), orientation="h",
            marker_color=["#d62728" if c==selected else
                          TYPE_COLORS.get(CARRIER_TYPE_MAP.get(c,""),"#7f7f7f")
                          for c in cd["Carrier"]],
            text=[f"{v:.1f}" for v in cd["CATE"]],
            textposition="outside"
        ))
        fig7.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global")
        fig7.update_layout(height=480, xaxis_title="CATE (pp)",
            margin=dict(l=0,r=60,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig7, width='stretch')

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
                y=pivot.index.tolist(),
                colorscale="RdYlGn_r",
                text=[[f"{v:.1f} pp" if not np.isnan(v) else "n/a" for v in r]
                      for r in pivot.values],
                texttemplate="%{text}",
                colorbar=dict(title="CATE (pp)"), zmin=10, zmax=55
            ))
            fig8.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=60))
            st.plotly_chart(fig8, width='stretch')
            st.caption("Red = high propagation risk. Above 38 pp warrants priority protocols.")
        else:
            st.info(f"No heatmap data for {selected}.")

    st.subheader("Where the Global View Misleads")
    if selected in DISAGREE:
        overall, aft_h2s, delta = DISAGREE[selected]
        st.warning(
            f"**{selected} ({CARRIER_NAMES.get(selected,selected)})** overall CATE: "
            f"**{overall:.2f} pp** — near average globally. But on **afternoon "
            f"hub-to-spoke routes**: **{aft_h2s:.2f} pp** ({delta} pp above own average, "
            f"p<0.001). A global recovery model would miss this vulnerability entirely."
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
        fig9.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global ATE")
        fig9.update_layout(height=250, yaxis_title="CATE (pp)",
            margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig9, width='stretch')

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Recovery Prioritization Engine")
    st.markdown("TOPSIS ranking combining causal propagation risk with operational criteria. "
                "Evaluated on 5M disrupted flights (2022–2025).")

    col1, col2 = st.columns(2)
    with col1:
        k_val = st.slider("Interventions per scenario (K)", 1, 10, 3)
    with col2:
        ct_filter = st.selectbox("Filter by carrier type",
                                  ["All","Mainline","LCC","ULCC","Regional"])

    st.subheader(f"Benchmark Comparison at K={k_val}")
    bench_k = D["bench"][
        (D["bench"]["k"]==k_val) & (D["bench"]["carrier_type"]==ct_filter)
    ].sort_values("pct_captured", ascending=False)

    if len(bench_k) > 0:
        fig10 = go.Figure(go.Bar(
            x=bench_k["method"].tolist(), y=bench_k["pct_captured"].tolist(),
            marker_color=[METHOD_COLORS.get(m,"#7f7f7f") for m in bench_k["method"]],
            text=[f"{v:.2f}%" for v in bench_k["pct_captured"]],
            textposition="outside"
        ))
        fig10.update_layout(height=360,
            yaxis_title="Downstream disruptions captured (%)",
            margin=dict(l=0,r=20,t=20,b=100),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig10, width='stretch')

    st.subheader("K Sensitivity — All Methods")
    bench_ct = D["bench"][D["bench"]["carrier_type"]==ct_filter]
    fig11 = go.Figure()
    for method in bench_ct["method"].unique():
        m_data = bench_ct[bench_ct["method"]==method].sort_values("k")
        fig11.add_trace(go.Scatter(
            x=m_data["k"].tolist(), y=m_data["pct_captured"].tolist(),
            mode="lines+markers", name=method,
            line=dict(color=METHOD_COLORS.get(method,"#7f7f7f"),
                      width=3 if method=="TOPSIS (DSS)" else 1.5,
                      dash="solid" if method=="TOPSIS (DSS)" else "dot"),
            marker=dict(size=5)
        ))
    fig11.update_layout(height=340,
        xaxis_title="K", yaxis_title="Downstream disruptions captured (%)",
        margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white",
        xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig11, width='stretch')

    st.subheader("Top Priority Recovery Queue (TOPSIS)")
    queue = D["queue"].copy()
    queue["CATE_T1"]              = (queue["CATE_T1"]*100).round(1)
    queue["ROUTE_30DAY_DISR_RATE"]= (queue["ROUTE_30DAY_DISR_RATE"]*100).round(1)
    queue["TOPSIS_SCORE"]         = queue["TOPSIS_SCORE"].round(3)
    queue["RECOVERY_WINDOW"]      = queue["RECOVERY_WINDOW"].round(0).astype(int)
    display = {"OP_UNIQUE_CARRIER":"Carrier","ORIGIN_AIRPORT":"Origin",
               "DEST_AIRPORT":"Dest","CARRIER_TYPE":"Type","ROUTE_TYPE":"Route",
               "CATE_T1":"CATE (pp)","DOWNSTREAM_SPILLOVER":"Spillover",
               "RECOVERY_WINDOW":"Rec Window","TOPSIS_SCORE":"TOPSIS Score",
               "NEXT_DISRUPTED":"Cascade?"}
# Replace the queue dataframe block in your app.py with this:
    st.dataframe(
        queue[list(display.keys())].rename(columns=display),
        width='stretch', height=360
    )

    st.subheader("TOPSIS Criteria Weights (Data-Driven)")
    col_pie, col_tbl = st.columns([1,1])
    with col_pie:
        fig12 = go.Figure(go.Pie(
            labels=WEIGHTS_DATA["Criterion"].tolist(),
            values=WEIGHTS_DATA["Weight (%)"].tolist(),
            hole=0.4,
            marker_colors=["#2E75B6","#2ca02c","#ff7f0e","#d62728","#9467bd"]
        ))
        fig12.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig12, width='stretch')
    with col_tbl:
        st.dataframe(WEIGHTS_DATA, width='stretch', hide_index=True)
    st.caption(
        "Weights derived from Pearson correlation with downstream disruption outcomes. "
        "Robust across 13 weight scenarios (max variation 0.25%)."
    )
    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.info("**TOPSIS outperforms** Hub-first, CATE-only, and Random at p<0.001. "
                "CATE alone underperforms random (-2.65%) but necessary in combination (+4.73%).")
    with col_f2:
        st.warning("**Carrier-type-specific.** TOPSIS best for Mainline (+4.88% over CATE-only). "
                   "CATE-only best for Regional (+3.44% over TOPSIS). One size does not fit all.")
'''

with open('app.py', 'w') as f:
    f.write(app_code)
print("✅ Final corrected app.py written")
print("   - No external file dependencies for weights/carrier data (hardcoded)")
print("   - Plotly v6 compatible (array/arrayminus instead of plus/minus)")
print("   - Only needs the 10 dash_*.csv files + requirements.txt")
