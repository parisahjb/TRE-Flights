
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Disruption Causal DSS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: white; border-radius: 8px; padding: 16px;
        border-left: 4px solid #2E75B6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #1F3864; }
    .metric-label { font-size: 13px; color: #595959; margin-top: 4px; }
    .high-risk  { border-left-color: #d62728 !important; }
    .low-risk   { border-left-color: #2ca02c !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GLOBAL_ATE = 30.57
CARRIER_TYPE_MAP = {
    "AA":"Mainline","DL":"Mainline","UA":"Mainline","AS":"Mainline",
    "HA":"Mainline","US":"Mainline",
    "WN":"LCC","B6":"LCC","VX":"LCC",
    "NK":"ULCC","F9":"ULCC","G4":"ULCC",
    "OO":"Regional","YV":"Regional","MQ":"Regional",
    "OH":"Regional","YX":"Regional","9E":"Regional",
    "EV":"Regional","QX":"Regional",
}
CARRIER_NAMES = {
    "AA":"American","DL":"Delta","UA":"United","AS":"Alaska","HA":"Hawaiian",
    "WN":"Southwest","B6":"JetBlue","NK":"Spirit","F9":"Frontier",
    "G4":"Allegiant","OO":"SkyWest","YX":"Republic","9E":"Endeavor",
    "MQ":"Envoy","OH":"PSA","YV":"Mesa",
}
TYPE_COLORS = {
    "Mainline":"#1f77b4","LCC":"#2ca02c","ULCC":"#d62728","Regional":"#ff7f0e"
}
TIME_ORDER  = ["Early morning (5-9)","Midday (10-14)",
               "Afternoon (15-19)","Evening/night (20+)"]
TIER_ORDER  = ["Large","Medium","Small","NonHub"]
ROUTE_ORDER = ["Hub-to-Hub","Hub-to-Spoke","Spoke-to-Spoke"]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    d = {}
    d["ct"]           = pd.read_csv("dash_cate_by_carrier_type.csv")
    d["rt"]           = pd.read_csv("dash_cate_by_route_type.csv")
    d["td"]           = pd.read_csv("dash_cate_by_time.csv")
    d["ot"]           = pd.read_csv("dash_cate_by_tier.csv")
    d["ct_rt"]        = pd.read_csv("dash_cate_carrier_route.csv")
    d["ct_td"]        = pd.read_csv("dash_cate_carrier_time.csv")
    d["carr_sum"]     = pd.read_csv("dash_carrier_summary.csv")
    d["carr_heat"]    = pd.read_csv("dash_carrier_heatmap.csv")
    d["carr_regime"]  = pd.read_csv("dash_carrier_regime.csv")
    d["bench_full"]   = pd.read_csv("dash_benchmark_full.csv")
    d["queue"]        = pd.read_csv("dash_top_queue.csv")
    d["weights"]      = pd.read_csv("nb5_topsis_weights.csv")
    d["carr_results"] = pd.read_parquet("checkpoint_nb4_carrier_results.parquet")
    return d

with st.spinner("Loading dashboard data..."):
    D = load_all()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Disruption Causal Decision Support System")
st.markdown(
    "**Causal propagation analysis** of 66.5M U.S. domestic flights "
    "(2015–2025) | BTS On-Time Performance Database"
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('''<div class="metric-card"><div class="metric-value">30.57 pp</div>
    <div class="metric-label">Global causal ATE [21.76, 39.38]</div></div>''',
    unsafe_allow_html=True)
with c2:
    st.markdown('''<div class="metric-card high-risk"><div class="metric-value">49.51%</div>
    <div class="metric-label">Downstream disruption rate</div></div>''',
    unsafe_allow_html=True)
with c3:
    st.markdown('''<div class="metric-card"><div class="metric-value">66.5M</div>
    <div class="metric-label">Flights analyzed (2015–2025)</div></div>''',
    unsafe_allow_html=True)
with c4:
    st.markdown('''<div class="metric-card low-risk"><div class="metric-value">0.81</div>
    <div class="metric-label">Predictive model AUC</div></div>''',
    unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🌐  Global Causal Intelligence",
    "🏢  Airline Command View",
    "🚑  Recovery Prioritization Engine"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: GLOBAL CAUSAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Global Causal Intelligence")
    st.markdown(
        "System-wide causal propagation effects across the U.S. air network. "
        "Treatment T1: prior flight on same aircraft rotation was disrupted."
    )

    # Row 1: carrier type + route type
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("CATE by Carrier Type")
        ct = D["ct"].sort_values("mean_cate")
        fig = go.Figure()
        for _, row in ct.iterrows():
            fig.add_trace(go.Bar(
                y=[row["CARRIER_TYPE"]], x=[row["mean_cate"]*100],
                orientation="h", name=row["CARRIER_TYPE"],
                marker_color=TYPE_COLORS.get(row["CARRIER_TYPE"],"#7f7f7f"),
                error_x=dict(type="data",
                    plus=[(row["ub"]-row["mean_cate"])*100],
                    minus=[(row["mean_cate"]-row["lb"])*100],
                    visible=True)
            ))
        fig.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black",
                      annotation_text=f"Global ATE {GLOBAL_ATE} pp")
        fig.update_layout(showlegend=False, height=280,
            xaxis_title="CATE (percentage points)",
            margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("CATE by Route Type")
        rt = D["rt"].sort_values("mean_cate")
        route_colors = {"Hub-to-Hub":"#1f77b4","Hub-to-Spoke":"#ff7f0e",
                        "Spoke-to-Spoke":"#d62728"}
        fig2 = go.Figure()
        for _, row in rt.iterrows():
            fig2.add_trace(go.Bar(
                y=[row["ROUTE_TYPE"]], x=[row["mean_cate"]*100],
                orientation="h", name=row["ROUTE_TYPE"],
                marker_color=route_colors.get(row["ROUTE_TYPE"],"#7f7f7f"),
                error_x=dict(type="data",
                    plus=[(row["ub"]-row["mean_cate"])*100],
                    minus=[(row["mean_cate"]-row["lb"])*100],
                    visible=True)
            ))
        fig2.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig2.update_layout(showlegend=False, height=280,
            xaxis_title="CATE (percentage points)",
            margin=dict(l=0,r=20,t=20,b=40), plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: time of day + tier
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("CATE by Time of Day")
        td = D["td"].set_index("TIME_BIN").reindex(TIME_ORDER)
        time_colors = ["#aec7e8","#ffbb78","#d62728","#9467bd"]
        fig3 = go.Figure(go.Bar(
            x=td.index, y=td["mean_cate"]*100,
            marker_color=time_colors,
            text=[f"{v:.1f} pp" for v in td["mean_cate"]*100],
            textposition="outside"
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
        tier_colors = ["#aec7e8","#ffbb78","#98df8a","#ff9896"]
        fig4 = go.Figure(go.Bar(
            x=ot.index, y=ot["mean_cate"]*100,
            marker_color=tier_colors,
            text=[f"{v:.1f} pp" for v in ot["mean_cate"]*100],
            textposition="outside"
        ))
        fig4.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black")
        fig4.update_layout(height=280, yaxis_title="CATE (pp)",
            margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Identification chain
    st.subheader("Causal Identification Chain")
    id_labels = ["Naive difference","V1 causal (no airport ctrl)",
                 "V2 causal (with airport ctrl) ← headline",
                 "Placebo V1 (future flight — contaminated)",
                 "Placebo V2 (other tail, same ahd)",
                 "Placebo V3 (other tail, same carrier+ahd)"]
    id_values = [38.01, 36.82, 30.57, 26.46, 3.28, 6.02]
    id_colors = ["#d62728","#ff7f0e","#2ca02c","#9467bd","#1f77b4","#17becf"]
    fig5 = go.Figure(go.Bar(
        y=id_labels[::-1], x=id_values[::-1],
        orientation="h", marker_color=id_colors[::-1],
        text=[f"{v:.2f} pp" for v in id_values[::-1]],
        textposition="outside"
    ))
    fig5.add_vline(x=5, line_dash="dot", line_color="gray",
                   annotation_text="5 pp threshold")
    fig5.update_layout(height=340,
        xaxis_title="Average Treatment Effect (percentage points)",
        margin=dict(l=0,r=90,t=20,b=40),
        plot_bgcolor="white", showlegend=False, xaxis_range=[0,46])
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Placebo V2 (3.28 pp) and V3 (6.02 pp) are near zero, confirming the "
        "rotation mechanism. All five tests form a rigorous identification chain."
    )

    # Row 4: Cross-tab heatmap
    st.subheader("CATE Heatmap — Carrier Type × Time of Day")
    ct_td = D["ct_td"].pivot(index="CARRIER_TYPE", columns="TIME_BIN",
                              values="mean_cate")
    ct_td = ct_td.reindex(columns=[t for t in TIME_ORDER if t in ct_td.columns])
    ct_td_pp = ct_td * 100
    fig6 = go.Figure(go.Heatmap(
        z=ct_td_pp.values,
        x=[c.split("(")[0].strip() for c in ct_td_pp.columns],
        y=ct_td_pp.index.tolist(),
        colorscale="RdYlGn_r",
        text=[[f"{v:.1f} pp" for v in row] for row in ct_td_pp.values],
        texttemplate="%{text}",
        colorbar=dict(title="CATE (pp)"),
        zmin=10, zmax=55
    ))
    fig6.update_layout(height=260,
        margin=dict(l=0,r=0,t=20,b=60))
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: AIRLINE COMMAND VIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Airline Command View")
    st.markdown(
        "Carrier-specific CATE estimates vs system average. "
        "Reveals where the global view misleads recovery decisions."
    )

    carrier_list = sorted(D["carr_sum"]["OP_UNIQUE_CARRIER"].tolist())
    selected = st.selectbox(
        "Select carrier",
        carrier_list,
        format_func=lambda x: f"{x} — {CARRIER_NAMES.get(x,x)} "
                               f"({CARRIER_TYPE_MAP.get(x,'')})"
    )

    # Metrics
    cs = D["carr_sum"]
    row = cs[cs["OP_UNIQUE_CARRIER"]==selected].iloc[0]
    mean_pp = row["mean_cate"]*100
    diff_pp = mean_pp - GLOBAL_ATE
    lb_pp   = row["lb"]*100
    ub_pp   = row["ub"]*100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk = "high-risk" if mean_pp > GLOBAL_ATE+3 else "low-risk"
        st.markdown(f'''<div class="metric-card {risk}">
        <div class="metric-value">{mean_pp:.2f} pp</div>
        <div class="metric-label">Carrier CATE (T1)</div></div>''',
        unsafe_allow_html=True)
    with c2:
        sign = "+" if diff_pp >= 0 else ""
        st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{sign}{diff_pp:.2f} pp</div>
        <div class="metric-label">vs Global ATE ({GLOBAL_ATE} pp)</div></div>''',
        unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{lb_pp:.1f}–{ub_pp:.1f}</div>
        <div class="metric-label">95% CI (pp)</div></div>''',
        unsafe_allow_html=True)
    with c4:
        ctype = CARRIER_TYPE_MAP.get(selected,"")
        fragility = "HIGH ⚠️" if diff_pp > 3 else ("LOW ✅" if diff_pp < -3 else "Average")
        st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{fragility}</div>
        <div class="metric-label">{ctype} carrier</div></div>''',
        unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("All Carriers — CATE Ranking")
        cs_sorted = cs.sort_values("mean_cate").copy()
        cs_sorted["mean_pp"] = cs_sorted["mean_cate"]*100
        bar_colors = [
            "#d62728" if c==selected else
            TYPE_COLORS.get(CARRIER_TYPE_MAP.get(c,""),"#7f7f7f")
            for c in cs_sorted["OP_UNIQUE_CARRIER"]
        ]
        fig7 = go.Figure(go.Bar(
            y=cs_sorted["OP_UNIQUE_CARRIER"], x=cs_sorted["mean_pp"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.1f}" for v in cs_sorted["mean_pp"]],
            textposition="outside"
        ))
        fig7.add_vline(x=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global")
        fig7.update_layout(height=450, xaxis_title="CATE (pp)",
            margin=dict(l=0,r=60,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    with col_r:
        st.subheader(f"{selected} — CATE by Route × Time")
        ch = D["carr_heat"]
        carr_ch = ch[ch["carrier"]==selected].copy()
        if len(carr_ch) > 0:
            pivot = carr_ch.pivot(index="route_type", columns="time_bin",
                                   values="mean_cate")
            pivot = pivot.reindex(
                index=[r for r in ROUTE_ORDER if r in pivot.index],
                columns=[t for t in TIME_ORDER if t in pivot.columns]
            ) * 100
            fig8 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[c.split("(")[0].strip() for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="RdYlGn_r",
                text=[[f"{v:.1f} pp" if not np.isnan(v) else "n/a"
                       for v in row] for row in pivot.values],
                texttemplate="%{text}",
                colorbar=dict(title="CATE (pp)"),
                zmin=10, zmax=55
            ))
            fig8.update_layout(height=300,
                margin=dict(l=0,r=0,t=20,b=60))
            st.plotly_chart(fig8, use_container_width=True)
            st.caption("Red = high propagation risk. Values above 38 pp warrant priority protocols.")
        else:
            st.info(f"No heatmap data for {selected}")

    # Disagreement box
    st.subheader("Where the Global View Misleads")
    disagree_carriers = {
        "AA": (32.72, 45.62, "+12.90"),
        "DL": (34.32, 46.68, "+12.36"),
        "UA": (31.84, 42.09, "+10.25"),
        "AS": (32.90, 49.08, "+16.18"),
        "WN": (34.88, 47.38, "+12.49"),
    }
    if selected in disagree_carriers:
        overall, aft_h2s, delta = disagree_carriers[selected]
        st.warning(
            f"**{selected} ({CARRIER_NAMES.get(selected,selected)})** "
            f"has an overall CATE of **{overall:.2f} pp** — near average. "
            f"But on **afternoon hub-to-spoke routes**, CATE rises to "
            f"**{aft_h2s:.2f} pp** ({delta} pp above its own average, p<0.001). "
            f"A global recovery model would miss this vulnerability entirely."
        )
    else:
        st.info(
            "Select AA, DL, UA, AS, or WN to see the global vs "
            "airline-specific disagreement analysis."
        )

    # Regime comparison
    st.subheader(f"{selected} — CATE by Regime")
    cr = D["carr_regime"]
    carr_reg = cr[cr["OP_UNIQUE_CARRIER"]==selected].copy()
    if len(carr_reg) > 0:
        fig9 = go.Figure(go.Bar(
            x=carr_reg["REGIME"], y=carr_reg["CATE_T1"]*100,
            marker_color=["#1f77b4","#d62728","#2ca02c"],
            text=[f"{v:.2f} pp" for v in carr_reg["CATE_T1"]*100],
            textposition="outside"
        ))
        fig9.add_hline(y=GLOBAL_ATE, line_dash="dash", line_color="black",
                       annotation_text="Global ATE")
        fig9.update_layout(height=250, yaxis_title="CATE (pp)",
            margin=dict(l=0,r=20,t=20,b=40),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RECOVERY PRIORITIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Recovery Prioritization Engine")
    st.markdown(
        "TOPSIS-based ranking combining causal propagation risk with "
        "operational criteria. Evaluated on 5M disrupted flights (2022–2025)."
    )

    col1, col2 = st.columns(2)
    with col1:
        k_val = st.slider("Interventions per scenario (K)", 1, 10, 3)
    with col2:
        ct_filter = st.selectbox(
            "Filter by carrier type",
            ["All","Mainline","LCC","ULCC","Regional"]
        )

    # Benchmark chart
    st.subheader(f"Benchmark Comparison at K={k_val}")
    bench = D["bench_full"]
    bench_k = bench[
        (bench["k"]==k_val) & (bench["carrier_type"]==ct_filter)
    ].sort_values("pct_captured", ascending=False)

    if len(bench_k) == 0:
        st.warning("No data for this combination.")
    else:
        method_colors = {
            "TOPSIS (DSS)"       : "#2ca02c",
            "Longest delay first": "#ff7f0e",
            "Earliest dep first" : "#1f77b4",
            "Highest spillover"  : "#9467bd",
            "Hub first"          : "#d62728",
            "CATE only"          : "#8c564b",
            "Random"             : "#7f7f7f",
        }
        bar_colors = [method_colors.get(m,"#7f7f7f") for m in bench_k["method"]]
        fig10 = go.Figure(go.Bar(
            x=bench_k["method"], y=bench_k["pct_captured"],
            marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in bench_k["pct_captured"]],
            textposition="outside"
        ))
        fig10.update_layout(height=360,
            yaxis_title="Downstream disruptions captured (%)",
            margin=dict(l=0,r=20,t=20,b=100),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig10, use_container_width=True)

    # K sensitivity curve
    st.subheader("TOPSIS Advantage vs K (all methods)")
    bench_all = D["bench_full"][D["bench_full"]["carrier_type"]==ct_filter]
    method_list = bench_all["method"].unique()
    fig11 = go.Figure()
    for method in method_list:
        m_data = bench_all[bench_all["method"]==method].sort_values("k")
        fig11.add_trace(go.Scatter(
            x=m_data["k"], y=m_data["pct_captured"],
            mode="lines+markers", name=method,
            line=dict(
                color=method_colors.get(method,"#7f7f7f"),
                width=3 if method=="TOPSIS (DSS)" else 1.5,
                dash="solid" if method=="TOPSIS (DSS)" else "dot"
            ),
            marker=dict(size=6)
        ))
    fig11.update_layout(height=360,
        xaxis_title="K — Interventions per scenario",
        yaxis_title="Downstream disruptions captured (%)",
        margin=dict(l=0,r=20,t=20,b=40),
        plot_bgcolor="white",
        legend=dict(orientation="v", x=1.01),
        xaxis=dict(tickmode="linear", dtick=1)
    )
    st.plotly_chart(fig11, use_container_width=True)

    # Priority queue
    st.subheader("Top Priority Recovery Queue (TOPSIS)")
    queue = D["queue"].copy()
    queue["CATE_T1"] = (queue["CATE_T1"]*100).round(1)
    queue["ROUTE_30DAY_DISR_RATE"] = (queue["ROUTE_30DAY_DISR_RATE"]*100).round(1)
    queue["TOPSIS_SCORE"] = queue["TOPSIS_SCORE"].round(3)
    queue["RECOVERY_WINDOW"] = queue["RECOVERY_WINDOW"].round(0).astype(int)
    display_cols = {
        "OP_UNIQUE_CARRIER":"Carrier","ORIGIN_AIRPORT":"Origin",
        "DEST_AIRPORT":"Dest","CARRIER_TYPE":"Type",
        "ROUTE_TYPE":"Route","ORIGIN_TIER":"Tier",
        "CATE_T1":"CATE (pp)","DOWNSTREAM_SPILLOVER":"Spillover",
        "RECOVERY_WINDOW":"Rec Window","TOPSIS_SCORE":"TOPSIS Score",
        "NEXT_DISRUPTED":"Downstream disrupted?"
    }
    queue_display = queue[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        queue_display.style.background_gradient(
            subset=["CATE (pp)","TOPSIS Score"], cmap="RdYlGn_r"
        ),
        use_container_width=True, height=380
    )

    # Weights
    st.subheader("TOPSIS Criteria Weights (Data-Driven)")
    col_pie, col_tbl = st.columns([1,1])
    with col_pie:
        w = D["weights"]
        fig12 = go.Figure(go.Pie(
            labels=w["Criterion"], values=w["Normalized_weight"]*100,
            hole=0.4,
            marker_colors=["#2E75B6","#2ca02c","#ff7f0e","#d62728","#9467bd"]
        ))
        fig12.update_layout(height=280,
            margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig12, use_container_width=True)
    with col_tbl:
        w_display = w[["Criterion","Normalized_weight"]].copy()
        w_display["Weight (%)"] = (w_display["Normalized_weight"]*100).round(2)
        st.dataframe(
            w_display[["Criterion","Weight (%)"]],
            use_container_width=True, hide_index=True
        )
    st.caption(
        "Weights derived from Pearson correlation with downstream disruption outcomes "
        "(2015–2019 training period). Robust across 13 weight perturbation scenarios "
        "(max variation 0.25%)."
    )

    # Key findings
    st.markdown("---")
    st.subheader("Key Findings")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.info(
            "**TOPSIS significantly outperforms** Hub-first, CATE-only, and "
            "Random at p<0.001. CATE alone underperforms random (-2.65%) but "
            "is necessary in combination (+4.73%)."
        )
    with col_f2:
        st.warning(
            "**Optimal method is carrier-type-specific.** TOPSIS best for "
            "Mainline (+4.88% over CATE-only). CATE-only best for Regional "
            "(+3.44% over TOPSIS). One-size-fits-all ranking underserves "
            "specific segments."
        )
