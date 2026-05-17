import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Flight Disruption Causal DSS", page_icon="✈️", layout="wide")
st.markdown("""
<style>
.metric-card{background:white;border-radius:8px;padding:16px;border-left:4px solid #2E75B6;box-shadow:0 1px 4px rgba(0,0,0,0.1);margin-bottom:8px;}
.metric-value{font-size:28px;font-weight:bold;color:#1F3864;}
.metric-label{font-size:13px;color:#595959;margin-top:4px;}
.high-risk{border-left-color:#d62728!important;}
.low-risk{border-left-color:#2ca02c!important;}
</style>""", unsafe_allow_html=True)

GLOBAL_ATE=30.57
CARRIER_TYPE_MAP={"AA":"Mainline","DL":"Mainline","UA":"Mainline","AS":"Mainline","HA":"Mainline","US":"Mainline","WN":"LCC","B6":"LCC","VX":"LCC","NK":"ULCC","F9":"ULCC","G4":"ULCC","OO":"Regional","YV":"Regional","MQ":"Regional","OH":"Regional","YX":"Regional","9E":"Regional","EV":"Regional","QX":"Regional"}
CARRIER_NAMES={"AA":"American","DL":"Delta","UA":"United","AS":"Alaska","HA":"Hawaiian","WN":"Southwest","B6":"JetBlue","NK":"Spirit","F9":"Frontier","G4":"Allegiant","OO":"SkyWest","YX":"Republic","9E":"Endeavor","MQ":"Envoy","OH":"PSA","YV":"Mesa"}
TYPE_COLORS={"Mainline":"#1f77b4","LCC":"#2ca02c","ULCC":"#d62728","Regional":"#ff7f0e"}
TIME_ORDER=["Early morning (5-9)","Midday (10-14)","Afternoon (15-19)","Evening/night (20+)"]
TIER_ORDER=["Large","Medium","Small","NonHub"]
ROUTE_ORDER=["Hub-to-Hub","Hub-to-Spoke","Spoke-to-Spoke"]
METHOD_COLORS={"TOPSIS (DSS)":"#2ca02c","Longest delay first":"#ff7f0e","Earliest dep first":"#1f77b4","Highest spillover":"#9467bd","Hub first":"#d62728","CATE only":"#8c564b","Random":"#7f7f7f"}
DISAGREE={"AA":(32.72,45.62,"+12.90"),"DL":(34.32,46.68,"+12.36"),"UA":(31.84,42.09,"+10.25"),"AS":(32.90,49.08,"+16.18"),"WN":(34.88,47.38,"+12.49")}
CARRIER_DATA=pd.DataFrame([
    {"Carrier":"AA","Type":"Mainline","CATE":32.72,"CI_LB":24.1,"CI_UB":41.3,"Diff":2.15},
    {"Carrier":"DL","Type":"Mainline","CATE":34.32,"CI_LB":25.8,"CI_UB":42.8,"Diff":3.75},
    {"Carrier":"UA","Type":"Mainline","CATE":31.84,"CI_LB":23.2,"CI_UB":40.4,"Diff":1.27},
    {"Carrier":"AS","Type":"Mainline","CATE":32.90,"CI_LB":24.3,"CI_UB":41.5,"Diff":2.33},
    {"Carrier":"HA","Type":"Mainline","CATE":39.64,"CI_LB":31.0,"CI_UB":48.3,"Diff":9.07},
    {"Carrier":"WN","Type":"LCC","CATE":34.88,"CI_LB":26.4,"CI_UB":43.4,"Diff":4.31},
    {"Carrier":"B6","Type":"LCC","CATE":38.39,"CI_LB":29.8,"CI_UB":47.0,"Diff":7.82},
    {"Carrier":"NK","Type":"ULCC","CATE":39.73,"CI_LB":31.1,"CI_UB":48.4,"Diff":9.16},
    {"Carrier":"F9","Type":"ULCC","CATE":39.52,"CI_LB":30.9,"CI_UB":48.1,"Diff":8.95},
    {"Carrier":"G4","Type":"ULCC","CATE":42.66,"CI_LB":34.0,"CI_UB":51.3,"Diff":12.09},
    {"Carrier":"OO","Type":"Regional","CATE":43.91,"CI_LB":35.3,"CI_UB":52.5,"Diff":13.34},
    {"Carrier":"YX","Type":"Regional","CATE":40.96,"CI_LB":32.4,"CI_UB":49.5,"Diff":10.39},
    {"Carrier":"9E","Type":"Regional","CATE":44.02,"CI_LB":35.4,"CI_UB":52.6,"Diff":13.45},
    {"Carrier":"MQ","Type":"Regional","CATE":43.14,"CI_LB":34.5,"CI_UB":51.8,"Diff":12.57},
    {"Carrier":"OH","Type":"Regional","CATE":43.25,"CI_LB":34.6,"CI_UB":51.9,"Diff":12.68},
    {"Carrier":"YV","Type":"Regional","CATE":41.32,"CI_LB":32.7,"CI_UB":49.9,"Diff":10.75},
])
WEIGHTS_DATA=pd.DataFrame([
    {"Criterion":"Recovery window","Weight (%)":52.04},
    {"Criterion":"Downstream spillover","Weight (%)":31.91},
    {"Criterion":"Historical disr rate","Weight (%)":7.80},
    {"Criterion":"CATE propagation risk","Weight (%)":6.13},
    {"Criterion":"Route strategic value","Weight (%)":2.12},
])

@st.cache_data
def load_all():
    return {
        "ct":pd.read_csv("dash_cate_by_carrier_type.csv"),
        "rt":pd.read_csv("dash_cate_by_route_type.csv"),
        "td":pd.read_csv("dash_cate_by_time.csv"),
        "ot":pd.read_csv("dash_cate_by_tier.csv"),
        "ct_td":pd.read_csv("dash_cate_carrier_time.csv"),
        "carr_sum":pd.read_csv("dash_carrier_summary.csv"),
        "carr_heat":pd.read_csv("dash_carrier_heatmap.csv"),
        "carr_reg":pd.read_csv("dash_carrier_regime.csv"),
        "bench":pd.read_csv("dash_benchmark_full.csv"),
        "queue":pd.read_csv("dash_top_queue.csv"),
    }

with st.spinner("Loading..."):
    D=load_all()

st.title("✈️ Flight Disruption Causal Decision Support System")
st.markdown("**Causal propagation analysis** of 66.5M U.S. domestic flights (2015–2025) | BTS On-Time Performance Database")
c1,c2,c3,c4=st.columns(4)
for col,val,lbl,cls in zip([c1,c2,c3,c4],["30.57 pp","49.51%","66.5M","0.81"],["Global causal ATE [21.76, 39.38]","Downstream disruption rate","Flights analyzed (2015–2025)","Predictive model AUC"],["","high-risk","","low-risk"]):
    with col:
        st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>',unsafe_allow_html=True)
with st.expander("ℹ️  About this dashboard", expanded=False):
    st.markdown("""
**What is this?**
A causal decision support system (DSS) for U.S. domestic flight disruption analysis and recovery prioritization,
built from 66.5 million BTS On-Time Performance records (2015–2025).

**What does CATE mean?**
CATE stands for Conditional Average Treatment Effect — it measures how much a prior flight disruption on the
same aircraft rotation *causally* increases the probability that the next flight is also disrupted.
A CATE of 30 pp means a prior disruption raises the next flight's disruption probability by 30 percentage points
(from a baseline of ~19% to ~49%). Estimated using a causal forest with airport-day confounding controls and
validated by three placebo tests.

**The four tabs:**
- **🌐 Global Causal Intelligence** — System-wide propagation effects by carrier type, route, time of day, and airport tier. Includes the full causal identification chain.
- **🏢 Airline Command View** — Select any carrier to see their CATE relative to the system average, a route × time heatmap, and where the global view misleads.
- **🚑 Recovery Prioritization Engine** — Compare seven ranking methods (including TOPSIS) for prioritizing which disrupted flights to recover first, at any intervention capacity K.
- **🚨 Live Flight Triage** — Enter today's disrupted flights and get a real-time TOPSIS priority ranking.

**Data source:** BTS On-Time Performance Database. No external data used.
**Models trained on:** 2015–2019. Evaluated on 2022–2025 (recovery regime).
""")

st.markdown("---")

tab1,tab2,tab3,tab4=st.tabs(["🌐  Global Causal Intelligence","🏢  Airline Command View","🚑  Recovery Prioritization Engine","🚨  Live Flight Triage"])

# TAB 1
with tab1:
    st.header("Global Causal Intelligence")
    st.markdown("System-wide causal propagation effects. Treatment T1: prior flight on same aircraft rotation was disrupted.")
    col_a,col_b=st.columns(2)
    with col_a:
        st.subheader("CATE by Carrier Type")
        ct=D["ct"].sort_values("mean_cate")
        fig=go.Figure([go.Bar(
            y=ct["CARRIER_TYPE"].tolist(),x=(ct["mean_cate"]*100).tolist(),orientation="h",
            marker_color=[TYPE_COLORS.get(c,"#7f7f7f") for c in ct["CARRIER_TYPE"]],
            error_x=dict(type="data",array=[(ub-m)*100 for m,ub in zip(ct["mean_cate"],ct["ub"])],arrayminus=[(m-lb)*100 for m,lb in zip(ct["mean_cate"],ct["lb"])],visible=True)
        )])
        fig.add_vline(x=GLOBAL_ATE,line_dash="dash",line_color="black",annotation_text=f"Global {GLOBAL_ATE} pp")
        fig.update_layout(showlegend=False,height=280,xaxis_title="CATE (pp)",margin=dict(l=0,r=20,t=20,b=40),plot_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with col_b:
        st.subheader("CATE by Route Type")
        rt=D["rt"].sort_values("mean_cate")
        rc={"Hub-to-Hub":"#1f77b4","Hub-to-Spoke":"#ff7f0e","Spoke-to-Spoke":"#d62728"}
        fig2=go.Figure([go.Bar(
            y=rt["ROUTE_TYPE"].tolist(),x=(rt["mean_cate"]*100).tolist(),orientation="h",
            marker_color=[rc.get(r,"#7f7f7f") for r in rt["ROUTE_TYPE"]],
            error_x=dict(type="data",array=[(ub-m)*100 for m,ub in zip(rt["mean_cate"],rt["ub"])],arrayminus=[(m-lb)*100 for m,lb in zip(rt["mean_cate"],rt["lb"])],visible=True)
        )])
        fig2.add_vline(x=GLOBAL_ATE,line_dash="dash",line_color="black")
        fig2.update_layout(showlegend=False,height=280,xaxis_title="CATE (pp)",margin=dict(l=0,r=20,t=20,b=40),plot_bgcolor="white")
        st.plotly_chart(fig2,use_container_width=True)
    col_c,col_d=st.columns(2)
    with col_c:
        st.subheader("CATE by Time of Day")
        td=D["td"].set_index("TIME_BIN").reindex(TIME_ORDER)
        fig3=go.Figure(go.Bar(x=td.index.tolist(),y=(td["mean_cate"]*100).tolist(),marker_color=["#aec7e8","#ffbb78","#d62728","#9467bd"],text=[f"{v:.1f} pp" for v in td["mean_cate"]*100],textposition="outside"))
        fig3.add_hline(y=GLOBAL_ATE,line_dash="dash",line_color="black",annotation_text="Global ATE")
        fig3.update_layout(height=280,yaxis_title="CATE (pp)",margin=dict(l=0,r=20,t=20,b=80),plot_bgcolor="white",showlegend=False)
        st.plotly_chart(fig3,use_container_width=True)
    with col_d:
        st.subheader("CATE by Origin Airport Tier")
        ot=D["ot"].set_index("ORIGIN_TIER").reindex(TIER_ORDER)
        fig4=go.Figure(go.Bar(x=ot.index.tolist(),y=(ot["mean_cate"]*100).tolist(),marker_color=["#aec7e8","#ffbb78","#98df8a","#ff9896"],text=[f"{v:.1f} pp" for v in ot["mean_cate"]*100],textposition="outside"))
        fig4.add_hline(y=GLOBAL_ATE,line_dash="dash",line_color="black")
        fig4.update_layout(height=280,yaxis_title="CATE (pp)",margin=dict(l=0,r=20,t=20,b=40),plot_bgcolor="white",showlegend=False)
        st.plotly_chart(fig4,use_container_width=True)
    st.subheader("Causal Identification Chain")
    id_labels=["Naive difference","V1 causal (no airport ctrl)","V2 causal (with airport ctrl) ← headline","Placebo V1 (future flight — contaminated)","Placebo V2 (other tail, same ahd)","Placebo V3 (other tail, same carrier+ahd)"]
    id_vals=[38.01,36.82,30.57,26.46,3.28,6.02]
    id_colors=["#d62728","#ff7f0e","#2ca02c","#9467bd","#1f77b4","#17becf"]
    fig5=go.Figure(go.Bar(y=id_labels[::-1],x=id_vals[::-1],orientation="h",marker_color=id_colors[::-1],text=[f"{v:.2f} pp" for v in id_vals[::-1]],textposition="outside"))
    fig5.add_vline(x=5,line_dash="dot",line_color="gray",annotation_text="5 pp threshold")
    fig5.update_layout(height=340,xaxis_title="Average Treatment Effect (pp)",margin=dict(l=0,r=90,t=20,b=40),plot_bgcolor="white",showlegend=False,xaxis_range=[0,46])
    st.plotly_chart(fig5,use_container_width=True)
    st.caption("Placebo V2 (3.28 pp) and V3 (6.02 pp) near zero — rotation mechanism confirmed. All heterogeneity differences significant at p<0.001.")
    st.subheader("CATE Heatmap — Carrier Type × Time of Day")
    ct_td=D["ct_td"].pivot(index="CARRIER_TYPE",columns="TIME_BIN",values="mean_cate")
    ct_td=ct_td.reindex(columns=[t for t in TIME_ORDER if t in ct_td.columns])*100
    fig6=go.Figure(go.Heatmap(z=ct_td.values.tolist(),x=[c.split("(")[0].strip() for c in ct_td.columns],y=ct_td.index.tolist(),colorscale="RdYlGn_r",text=[[f"{v:.1f} pp" for v in row] for row in ct_td.values],texttemplate="%{text}",colorbar=dict(title="CATE (pp)"),zmin=10,zmax=55))
    fig6.update_layout(height=260,margin=dict(l=0,r=0,t=20,b=60))
    st.plotly_chart(fig6,use_container_width=True)

# TAB 2
with tab2:
    st.header("Airline Command View")
    st.markdown("Carrier-specific CATE vs system average. Reveals where global rankings mislead recovery decisions.")
    carrier_list=sorted(CARRIER_DATA["Carrier"].tolist())
    selected=st.selectbox("Select carrier",carrier_list,format_func=lambda x:f"{x} — {CARRIER_NAMES.get(x,x)} ({CARRIER_TYPE_MAP.get(x,'')})")
    row=CARRIER_DATA[CARRIER_DATA["Carrier"]==selected].iloc[0]
    diff=row["CATE"]-GLOBAL_ATE
    c1,c2,c3,c4=st.columns(4)
    with c1:
        cls="high-risk" if row["CATE"]>GLOBAL_ATE+3 else "low-risk"
        st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{row["CATE"]:.2f} pp</div><div class="metric-label">Carrier CATE (T1)</div></div>',unsafe_allow_html=True)
    with c2:
        sign="+" if diff>=0 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sign}{diff:.2f} pp</div><div class="metric-label">vs Global ATE ({GLOBAL_ATE} pp)</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{row["CI_LB"]:.1f}–{row["CI_UB"]:.1f}</div><div class="metric-label">95% CI (pp)</div></div>',unsafe_allow_html=True)
    with c4:
        frag="HIGH ⚠️" if diff>3 else ("LOW ✅" if diff<-3 else "Average")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{frag}</div><div class="metric-label">{row["Type"]} carrier</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    col_l,col_r=st.columns(2)
    with col_l:
        st.subheader("All Carriers — CATE Ranking")
        cd=CARRIER_DATA.sort_values("CATE")
        fig7=go.Figure(go.Bar(y=cd["Carrier"].tolist(),x=cd["CATE"].tolist(),orientation="h",marker_color=["#d62728" if c==selected else TYPE_COLORS.get(CARRIER_TYPE_MAP.get(c,""),"#7f7f7f") for c in cd["Carrier"]],text=[f"{v:.1f}" for v in cd["CATE"]],textposition="outside"))
        fig7.add_vline(x=GLOBAL_ATE,line_dash="dash",line_color="black",annotation_text="Global")
        fig7.update_layout(height=480,xaxis_title="CATE (pp)",margin=dict(l=0,r=60,t=20,b=40),plot_bgcolor="white",showlegend=False)
        st.plotly_chart(fig7,use_container_width=True)
    with col_r:
        st.subheader(f"{selected} — CATE by Route × Time")
        ch=D["carr_heat"]
        carr_ch=ch[ch["carrier"]==selected]
        if len(carr_ch)>0:
            pivot=carr_ch.pivot(index="route_type",columns="time_bin",values="mean_cate")
            pivot=pivot.reindex(index=[r for r in ROUTE_ORDER if r in pivot.index],columns=[t for t in TIME_ORDER if t in pivot.columns])*100
            fig8=go.Figure(go.Heatmap(z=pivot.values.tolist(),x=[c.split("(")[0].strip() for c in pivot.columns],y=pivot.index.tolist(),colorscale="RdYlGn_r",text=[[f"{v:.1f} pp" if not np.isnan(v) else "n/a" for v in r] for r in pivot.values],texttemplate="%{text}",colorbar=dict(title="CATE (pp)"),zmin=10,zmax=55))
            fig8.update_layout(height=300,margin=dict(l=0,r=0,t=20,b=60))
            st.plotly_chart(fig8,use_container_width=True)
            st.caption("Red = high propagation risk. Above 38 pp warrants priority protocols.")
        else:
            st.info(f"No heatmap data for {selected}.")
    st.subheader("Where the Global View Misleads")
    if selected in DISAGREE:
        overall,aft_h2s,delta=DISAGREE[selected]
        st.warning(f"**{selected} ({CARRIER_NAMES.get(selected,selected)})** overall CATE: **{overall:.2f} pp** — near average globally. But on **afternoon hub-to-spoke routes**: **{aft_h2s:.2f} pp** ({delta} pp above own average, p<0.001). A global recovery model would miss this vulnerability entirely.")
    else:
        st.info("Select AA, DL, UA, AS, or WN to see the global vs airline-specific disagreement.")
    st.subheader(f"{selected} — CATE across Regimes")
    cr=D["carr_reg"]
    carr_reg=cr[cr["OP_UNIQUE_CARRIER"]==selected]
    if len(carr_reg)>0:
        fig9=go.Figure(go.Bar(x=carr_reg["REGIME"].tolist(),y=(carr_reg["CATE_T1"]*100).tolist(),marker_color=["#1f77b4","#2ca02c"],text=[f"{v:.2f} pp" for v in carr_reg["CATE_T1"]*100],textposition="outside"))
        fig9.add_hline(y=GLOBAL_ATE,line_dash="dash",line_color="black",annotation_text="Global ATE")
        fig9.update_layout(height=250,yaxis_title="CATE (pp)",margin=dict(l=0,r=20,t=20,b=40),plot_bgcolor="white",showlegend=False)
        st.plotly_chart(fig9,use_container_width=True)

# TAB 3
with tab3:
    st.header("Recovery Prioritization Engine")
    st.markdown("TOPSIS ranking combining causal propagation risk with operational criteria. Evaluated on 5M disrupted flights (2022–2025).")
    col1,col2=st.columns(2)
    with col1:
        k_val=st.slider("Interventions per scenario (K)",1,10,3)
    with col2:
        ct_filter=st.selectbox("Filter by carrier type",["All","Mainline","LCC","ULCC","Regional"])
    st.subheader(f"Benchmark Comparison at K={k_val}")
    bench_k=D["bench"][(D["bench"]["k"]==k_val)&(D["bench"]["carrier_type"]==ct_filter)].sort_values("pct_captured",ascending=False)
    if len(bench_k)>0:
        fig10=go.Figure(go.Bar(x=bench_k["method"].tolist(),y=bench_k["pct_captured"].tolist(),marker_color=[METHOD_COLORS.get(m,"#7f7f7f") for m in bench_k["method"]],text=[f"{v:.2f}%" for v in bench_k["pct_captured"]],textposition="outside"))
        fig10.update_layout(height=360,yaxis_title="Downstream disruptions captured (%)",margin=dict(l=0,r=20,t=20,b=100),plot_bgcolor="white",showlegend=False)
        st.plotly_chart(fig10,use_container_width=True)
    st.subheader("K Sensitivity — All Methods")
    bench_ct=D["bench"][D["bench"]["carrier_type"]==ct_filter]
    fig11=go.Figure()
    for method in bench_ct["method"].unique():
        m_data=bench_ct[bench_ct["method"]==method].sort_values("k")
        fig11.add_trace(go.Scatter(x=m_data["k"].tolist(),y=m_data["pct_captured"].tolist(),mode="lines+markers",name=method,line=dict(color=METHOD_COLORS.get(method,"#7f7f7f"),width=3 if method=="TOPSIS (DSS)" else 1.5,dash="solid" if method=="TOPSIS (DSS)" else "dot"),marker=dict(size=5)))
    fig11.update_layout(height=340,xaxis_title="K",yaxis_title="Downstream disruptions captured (%)",margin=dict(l=0,r=20,t=20,b=40),plot_bgcolor="white",xaxis=dict(tickmode="linear",dtick=1))
    st.plotly_chart(fig11,use_container_width=True)
    st.subheader("Top Priority Recovery Queue (TOPSIS)")
    queue=D["queue"].copy()
    queue["CATE_T1"]=(queue["CATE_T1"]*100).round(1)
    queue["ROUTE_30DAY_DISR_RATE"]=(queue["ROUTE_30DAY_DISR_RATE"]*100).round(1)
    queue["TOPSIS_SCORE"]=queue["TOPSIS_SCORE"].round(3)
    queue["RECOVERY_WINDOW"]=queue["RECOVERY_WINDOW"].round(0).astype(int)
    display={"OP_UNIQUE_CARRIER":"Carrier","ORIGIN_AIRPORT":"Origin","DEST_AIRPORT":"Dest","CARRIER_TYPE":"Type","ROUTE_TYPE":"Route","CATE_T1":"CATE (pp)","DOWNSTREAM_SPILLOVER":"Spillover","RECOVERY_WINDOW":"Rec Window","TOPSIS_SCORE":"TOPSIS Score","NEXT_DISRUPTED":"Cascade?"}
    st.dataframe(queue[list(display.keys())].rename(columns=display),use_container_width=True,height=360)
    st.subheader("TOPSIS Criteria Weights (Data-Driven)")
    col_pie,col_tbl=st.columns([1,1])
    with col_pie:
        fig12=go.Figure(go.Pie(labels=WEIGHTS_DATA["Criterion"].tolist(),values=WEIGHTS_DATA["Weight (%)"].tolist(),hole=0.4,marker_colors=["#2E75B6","#2ca02c","#ff7f0e","#d62728","#9467bd"]))
        fig12.update_layout(height=280,margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig12,use_container_width=True)
    with col_tbl:
        st.dataframe(WEIGHTS_DATA,use_container_width=True,hide_index=True)
    st.caption("Weights derived from Pearson correlation with downstream disruption outcomes. Robust across 13 weight scenarios (max variation 0.25%).")
    st.markdown("---")
    col_f1,col_f2=st.columns(2)
    with col_f1:
        st.info("**TOPSIS outperforms** Hub-first, CATE-only, and Random at p<0.001. CATE alone underperforms random (-2.65%) but necessary in combination (+4.73%).")
    with col_f2:
        st.warning("**Carrier-type-specific.** TOPSIS best for Mainline (+4.88% over CATE-only). CATE-only best for Regional (+3.44% over TOPSIS). One size does not fit all.")

# TAB 4
with tab4:
    st.header("Live Flight Triage")
    st.markdown("Enter disrupted flights for a given airport and date. The system computes a real-time TOPSIS priority ranking using causal propagation estimates and operational criteria.")

    with st.expander("📖  How to use this tab", expanded=False):
        st.markdown("""
**Scenario:** You are an airline ops manager at a hub airport. Several flights have been disrupted.
You have limited ground crew or gate resources and need to decide which flights to intervene on first.

**Step-by-step:**

1. **Add each disrupted flight** using the form below. For each flight you need:
   - **Carrier and airports** — the operating carrier, origin, and destination IATA codes (e.g. ATL, ORD)
   - **Route type** — Hub-to-Hub (both airports are large/medium hubs), Hub-to-Spoke (one hub, one smaller airport), or Spoke-to-Spoke (both small/non-hub)
   - **Origin tier** — Large hub (e.g. ATL, ORD, LAX), Medium hub (e.g. RDU, MSY), Small hub, or Non-hub
   - **Departure hour** — the scheduled departure hour (0–23)
   - **Current delay** — how many minutes the flight is currently delayed
   - **Downstream flights at risk** — how many subsequent flights on this aircraft rotation today could be affected
   - **Route 30-day disruption rate** — approximate percentage of days this route has been disrupted in the past month (check your OCC system or use 20% as a default)
   - **Route annual departures** — approximate annual frequency of this OD pair (use 2000 as a default if unknown)

2. **CATE is auto-estimated** — you do not need to enter it. The system looks it up from carrier × route × time patterns trained on 5 years of BTS data.

3. **Click Add Flight** for each disrupted flight, then review the TOPSIS priority ranking.

**What the ranking means:**
The flight ranked 🔴 CRITICAL has the highest composite priority score — it combines causal propagation risk, downstream spillover, recovery window, and historical burden. Intervening on this flight first is expected to prevent the most downstream disruptions.

**Tip:** Add at least 3–5 flights to make the ranking meaningful. With only 2 flights the comparison is limited.
""")

    st.info("**CATE auto-estimated** from carrier × route × time heatmap (2015–2019 training data). Enter all other fields from today's schedule.")
    W={"CATE":0.0613,"SPILLOVER":0.3191,"STRATEGIC":0.0212,"WINDOW":0.5204,"HIST_RATE":0.0780}

    @st.cache_data
    def get_cate_lookup():
        return pd.read_csv("dash_carrier_heatmap.csv")
    cate_lookup=get_cate_lookup()

    def lookup_cate(carrier,route_type,hour):
        if hour<=9: time_bin="Early morning (5-9)"
        elif hour<=14: time_bin="Midday (10-14)"
        elif hour<=19: time_bin="Afternoon (15-19)"
        else: time_bin="Evening/night (20+)"
        match=cate_lookup[(cate_lookup["carrier"]==carrier)&(cate_lookup["route_type"]==route_type)&(cate_lookup["time_bin"]==time_bin)]
        if len(match)>0:
            return match["mean_cate"].iloc[0]*100,time_bin
        carr_avg=cate_lookup[cate_lookup["carrier"]==carrier]["mean_cate"].mean()
        return (carr_avg*100 if not np.isnan(carr_avg) else 30.57),time_bin

    if "flights" not in st.session_state:
        st.session_state.flights=[]

    st.subheader("Step 1 — Enter Disrupted Flights")
    st.markdown("Add up to 10 disrupted flights for a single airport-date scenario.")
    with st.form("flight_input_form"):
        st.markdown("**Add a flight:**")
        col1,col2,col3=st.columns(3)
        with col1:
            f_carrier=st.selectbox("Carrier",sorted(CARRIER_DATA["Carrier"].tolist()),format_func=lambda x:f"{x} — {CARRIER_NAMES.get(x,x)}")
            f_origin=st.text_input("Origin airport (IATA)","ATL")
            f_dest=st.text_input("Destination airport (IATA)","ORD")
        with col2:
            f_route=st.selectbox("Route type",["Hub-to-Hub","Hub-to-Spoke","Spoke-to-Spoke"])
            f_tier=st.selectbox("Origin tier",["Large","Medium","Small","NonHub"])
            f_hour=st.slider("Scheduled departure hour",0,23,14)
        with col3:
            f_delay=st.number_input("Current delay (min)",0,600,45)
            f_spillover=st.number_input("Downstream flights at risk",0,20,3)
            f_hist_rate=st.slider("Route 30-day disruption rate (%)",0,100,22)
        f_strategic=st.number_input("Route annual departures (approx)",100,20000,2000,help="Approximate annual departure count for this OD pair")
        submitted=st.form_submit_button("➕ Add Flight",type="primary")
        if submitted:
            if len(st.session_state.flights)>=10:
                st.warning("Maximum 10 flights per scenario.")
            else:
                cate_val,time_bin=lookup_cate(f_carrier,f_route,f_hour)
                rec_window=max(0,19*60-f_hour*60)
                st.session_state.flights.append({"Carrier":f_carrier,"Origin":f_origin.upper(),"Dest":f_dest.upper(),"Route type":f_route,"Tier":f_tier,"Hour":f_hour,"Time bin":time_bin,"Delay (min)":f_delay,"CATE (pp)":round(cate_val,2),"Spillover":f_spillover,"Rec Window":rec_window,"Hist Rate":f_hist_rate/100,"Strategic":f_strategic})
                st.success(f"Flight {f_carrier} {f_origin.upper()}→{f_dest.upper()} added.")

    col_clear,_=st.columns([1,4])
    with col_clear:
        if st.button("🗑️ Clear all flights"):
            st.session_state.flights=[]

    if len(st.session_state.flights)>0:
        st.subheader(f"Step 2 — Entered Flights ({len(st.session_state.flights)})")
        flights_df=pd.DataFrame(st.session_state.flights)
        st.dataframe(flights_df[["Carrier","Origin","Dest","Route type","Tier","Hour","CATE (pp)","Spillover","Rec Window","Delay (min)"]],use_container_width=True,hide_index=True)
        st.subheader("Step 3 — TOPSIS Priority Ranking")
        if len(st.session_state.flights)<2:
            st.warning("Add at least 2 flights to compute a ranking.")
        else:
            X=np.array([[f["CATE (pp)"]/100,f["Spillover"],f["Strategic"],f["Rec Window"],f["Hist Rate"]] for f in st.session_state.flights],dtype=float)
            n,m=X.shape
            col_norms=np.sqrt((X**2).sum(axis=0))
            col_norms[col_norms==0]=1
            X_norm=X/col_norms
            w=np.array([W["CATE"],W["SPILLOVER"],W["STRATEGIC"],W["WINDOW"],W["HIST_RATE"]])
            X_w=X_norm*w
            ideal_best=X_w.max(axis=0)
            ideal_worst=X_w.min(axis=0)
            d_best=np.sqrt(((X_w-ideal_best)**2).sum(axis=1))
            d_worst=np.sqrt(((X_w-ideal_worst)**2).sum(axis=1))
            denom=d_best+d_worst
            denom[denom==0]=1e-10
            scores=d_worst/denom
            ranks=len(scores)-scores.argsort().argsort()
            results=flights_df.copy()
            results["TOPSIS Score"]=scores.round(4)
            results["Priority Rank"]=ranks
            results=results.sort_values("Priority Rank")
            def priority_color(r):
                if r==1: return "🔴 CRITICAL"
                elif r==2: return "🟠 HIGH"
                elif r==3: return "🟡 MEDIUM"
                else: return "🟢 LOWER"
            results["Priority"]=results["Priority Rank"].apply(priority_color)
            st.dataframe(results[["Priority","Carrier","Origin","Dest","Route type","CATE (pp)","Spillover","Rec Window","TOPSIS Score"]].reset_index(drop=True),use_container_width=True,hide_index=True,height=380)
            top=results.iloc[0]
            st.markdown("---")
            st.subheader(f"🔴 Top Priority: {top['Carrier']} {top['Origin']}→{top['Dest']}")
            c1,c2,c3,c4=st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card high-risk"><div class="metric-value">{top["CATE (pp)"]:.1f} pp</div><div class="metric-label">Causal propagation risk</div></div>',unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(top["Spillover"])}</div><div class="metric-label">Downstream flights at risk</div></div>',unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(top["Rec Window"])} min</div><div class="metric-label">Recovery window remaining</div></div>',unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card low-risk"><div class="metric-value">{top["TOPSIS Score"]:.3f}</div><div class="metric-label">TOPSIS priority score</div></div>',unsafe_allow_html=True)
            top_pos=list(flights_df.index).index(results.index[0])
            X_scaled=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)+1e-10)*100
            top_vals=X_scaled[top_pos].tolist()
            criteria_names=["Causal Risk","Downstream Spillover","Strategic Value","Recovery Window","Historical Rate"]
            fig_radar=go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=top_vals+[top_vals[0]],theta=criteria_names+[criteria_names[0]],fill="toself",fillcolor="rgba(214,39,40,0.2)",line=dict(color="#d62728",width=2),name=f"{top['Carrier']} {top['Origin']}→{top['Dest']}"))
            fig_radar.add_trace(go.Scatterpolar(r=[50]*6,theta=criteria_names+[criteria_names[0]],fill="toself",fillcolor="rgba(31,119,180,0.1)",line=dict(color="#1f77b4",width=1.5,dash="dot"),name="Scenario average"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=True,height=350,margin=dict(l=40,r=40,t=40,b=40))
            col_radar,col_note=st.columns([1,1])
            with col_radar:
                st.plotly_chart(fig_radar,use_container_width=True)
            with col_note:
                st.markdown("**Why this flight is top priority:**")
                reasons=[]
                if top_vals[0]>60: reasons.append(f"High causal propagation risk ({top['CATE (pp)']:.1f} pp)")
                if top_vals[1]>60: reasons.append(f"Large downstream spillover ({int(top['Spillover'])} flights at risk)")
                if top_vals[3]>60: reasons.append(f"Long recovery window ({int(top['Rec Window'])} min remaining)")
                if top_vals[4]>60: reasons.append(f"Chronically disrupted route ({top['Hist Rate']*100:.1f}% 30-day rate)")
                if not reasons: reasons.append("Highest composite TOPSIS score across all criteria")
                for r in reasons:
                    st.markdown(f"• {r}")
                st.caption("CATE estimated from carrier × route × time heatmap (2015–2019 training data).")
    else:
        st.info("No flights entered yet. Use the form above to add disrupted flights.")
        st.markdown("**Example scenario:** ATL on a weekday afternoon with 5 disrupted flights competing for 2 available ground crews. Enter each flight's details above to see which should be prioritized.")
