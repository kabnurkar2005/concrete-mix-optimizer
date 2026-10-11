import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from mix_design import (MixInputs, calculate_mix, conventional_mix, optimize_mixes, rank_mixes, PRESETS,
                        durability_limits, valid_grades, estimate_water, coarse_fraction, curve_for, wc_for_strength,
                        target_mean_strength, sd_for_grade, calibration_factor, design_water, EXPOSURES, WORKABILITY, WATER_BASE,
                        AGG_SHAPE_REDUCTION, CA_VOLUME, SCM_LIMITS, SCM_SG, SCM_RATE_RANGE, SCM_K, CURVES, CURVE_WC,
                        MAX_CEMENT, SC, CC, CEM, CO2K)

st.set_page_config(page_title="Concrete Mix Optimizer", page_icon="🏗️", layout="wide")
st.title("🏗️ Concrete Mix Optimizer")
st.caption("IS 10262:2019 / IS 456 based mix design — finds cheaper, greener mixes and proves the strength is still adequate.")

# =============================================================== SIDEBAR (options narrow as you go)
with st.sidebar:
    st.header("1. Concrete & exposure")
    ctype = st.radio("Concrete type", ["RCC", "PCC"], horizontal=True)
    exposure = st.selectbox("Exposure condition", EXPOSURES, index=1)
    max_wc, min_c, min_grade = durability_limits(ctype, exposure)
    st.info(f"🔒 Fixed by IS 456 Table 5 — max w/c **{max_wc:.2f}**, min cementitious **{min_c} kg/m³**, min grade **{min_grade}**")
    grades = valid_grades(ctype, exposure)
    grade = st.selectbox("Concrete grade (only valid grades shown)", grades, index=grades.index("M25") if "M25" in grades else 0)
    site = st.radio("Site quality control", ["Good", "Fair"], horizontal=True, help="Fair adds 1 N/mm² to the assumed standard deviation (IS 10262 Table 2 note).")
    work = st.radio("Workability", list(WORKABILITY), index=1, horizontal=True)
    s_lo, s_hi, s_def = WORKABILITY[work]
    slump = st.slider("Target slump (mm)", s_lo, s_hi, s_def, 5)
    agg_size = st.selectbox("Nominal max aggregate size (mm)", [10, 20, 40], index=1)
    fine_zone = st.selectbox("Fine aggregate grading zone", ["Zone I", "Zone II", "Zone III", "Zone IV"], index=1)
    agg_shape = st.selectbox("Coarse aggregate type", list(AGG_SHAPE_REDUCTION))
    pumping = st.checkbox("Pumped / congested reinforcement", help="IS 10262 cl. 4.4.1: coarse aggregate may be reduced up to 10%.")
    wred = st.slider("Water reduction by chemical admixture (%)", 0, 30, 0, 1, help="IS 10262: superplasticiser may cut water up to 30%. Admixture cost is not included.")

    st.divider()
    st.header("2. Cement & SCM")
    cement_choice = st.selectbox("Cement type & grade", ["OPC 33", "OPC 43", "OPC 53", "PPC"], index=1)
    cement_type = "PPC" if cement_choice == "PPC" else "OPC"
    opc_grade = cement_choice.split()[-1] if cement_type == "OPC" else "-"
    st.caption(f"Strength curve used: IS 10262 Fig. 1 **curve {curve_for(cement_type, opc_grade)}**")
    if cement_type == "PPC":
        scm_opts = ["None"]
        st.caption("PPC already contains fly ash, so extra SCM is not offered. Choose OPC to use an SCM.")
    else:
        scm_opts = ["None", "Fly ash", "GGBS", "Silica fume"]
    scm_type = st.selectbox("SCM type", scm_opts)
    scm_range, scm_sg, scm_rate, scm_k = None, 2.2, 0.0, 0.0
    if scm_type != "None":
        lo, hi = SCM_LIMITS[scm_type]
        scm_range = st.slider(f"{scm_type} replacement range to search (% of binder)", lo, hi, (lo, hi), 5,
                              help="Limits follow typical IS 456/IS 1489/IS 16714/IS 15388 ranges.")
        scm_sg = st.number_input(f"{scm_type} specific gravity", 1.5, 4.0, SCM_SG[scm_type], 0.05)
        mode = st.radio("Supply form", ["Bulk", "Bagged"], horizontal=True)
        r_lo, r_hi = SCM_RATE_RANGE[scm_type][mode]
        scm_rate = st.number_input(f"{scm_type} rate (₹/kg)", 0.0, 200.0, round((r_lo + r_hi) / 2, 2), 0.1,
                                   help=f"Typical {mode.lower()} range: ₹{r_lo}–{r_hi}/kg")
        with st.expander("Advanced: SCM strength efficiency"):
            scm_k = st.number_input("28-day efficiency factor k", 0.0, 3.0, SCM_K[scm_type], 0.05,
                                    help="Assumption (not an IS value): 1 kg SCM counts as k kg cement for strength. Verify with trials.")

    st.divider()
    st.header("3. Aggregates & moisture")
    st.caption("Enter your lab-measured values. Typical ranges are shown as hints only.")
    fine_sg = st.number_input("Fine aggregate SG", 2.0, 3.2, 2.60, 0.01, help="Typical 2.55–2.65")
    coarse_sg = st.number_input("Coarse aggregate SG", 2.0, 3.4, 2.70, 0.01, help="Typical 2.60–2.90")
    cement_sg = st.number_input("Cement SG", 2.8, 3.3, 3.15, 0.01)
    fa_moist = st.number_input("Fine aggregate moisture (%)", 0.0, 15.0, 2.0, 0.1, help="Typical 1–3")
    ca_moist = st.number_input("Coarse aggregate moisture (%)", 0.0, 10.0, 1.0, 0.1, help="Typical 0.5–2")
    fa_abs = st.number_input("Fine aggregate absorption (%)", 0.0, 10.0, 1.0, 0.1)
    ca_abs = st.number_input("Coarse aggregate absorption (%)", 0.0, 10.0, 0.5, 0.1)

    st.divider()
    st.header("4. Local rates")
    bag = st.number_input("Cement (₹ per 50 kg bag)", 100.0, 1000.0, 400.0, 10.0, help="Typical ₹300–500")
    fa_ton = st.number_input("Fine aggregate (₹/tonne)", 100.0, 6000.0, 2000.0, 50.0, help="Typical ₹1500–2500")
    ca_ton = st.number_input("Coarse aggregate (₹/tonne)", 100.0, 6000.0, 2000.0, 50.0, help="Typical ₹1000–3200")
    water_rate = st.number_input("Water (₹/kg)", 0.0, 10.0, 1.35, 0.05, help="Typical ₹1.20–1.50")

    st.divider()
    st.header("5. Optimization goal")
    preset = st.radio("Ranking priority", list(PRESETS), help="Weights for cost / strength / CO₂ in the ranking score.")
    max_loss = st.slider("Max strength drop vs conventional (%)", 0, 20, 5,
                         help="Mixes must also reach the target mean strength.")

    st.divider()
    st.header("6. Lab calibration (optional)")
    use_cal = st.checkbox("I tested my conventional mix (no SCM)",
                          help="Enter your measured 28-day cube strength so predictions use your own materials instead of the generic IS curve.")
    cal_pairs = []
    if use_cal:
        st.caption("Cement-only mix, same cement type/grade as selected above. Use the average of 3 cubes.")
        w1 = st.number_input("Tested mix 1 – w/c", 0.25, 0.80, 0.50, 0.01)
        f1 = st.number_input("Tested mix 1 – 28-day strength (MPa)", 5.0, 100.0, 35.0, 0.5)
        cal_pairs = [(w1, f1)]
        with st.expander("Add more tested mixes (optional)"):
            for n in (2, 3):
                wn = st.number_input(f"Tested mix {n} – w/c (0 = unused)", 0.0, 0.80, 0.0, 0.01)
                fn = st.number_input(f"Tested mix {n} – strength (MPa)", 0.0, 100.0, 0.0, 0.5)
                if wn > 0 and fn > 0:
                    cal_pairs.append((wn, fn))
cal_factor = calibration_factor(curve_for(cement_type, opc_grade), cal_pairs)

inputs = MixInputs(grade=grade, concrete_type=ctype, exposure=exposure, site_control=site, slump_mm=slump,
                   agg_size_mm=agg_size, fine_zone=fine_zone, agg_shape=agg_shape, pumping=pumping,
                   water_reduction_pct=wred, cement_type=cement_type, opc_grade=opc_grade, scm_type=scm_type,
                   scm_k=scm_k, fine_sg=fine_sg, coarse_sg=coarse_sg, cement_sg=cement_sg, scm_sg=scm_sg,
                   fine_moisture_pct=fa_moist, coarse_moisture_pct=ca_moist, fine_absorption_pct=fa_abs,
                   coarse_absorption_pct=ca_abs, cement_rate=bag / 50.0, scm_rate=scm_rate,
                   fa_rate=fa_ton / 1000.0, ca_rate=ca_ton / 1000.0, water_rate=water_rate, strength_factor=cal_factor)

target = target_mean_strength(grade, site)
curve = curve_for(cement_type, opc_grade)
water = design_water(inputs)

# ------------------------------------------------------------------ locked code values
st.subheader("Code-derived values (locked)")
k = st.columns(5)
k[0].metric("Max w/c (IS 456)", f"{max_wc:.2f}")
k[1].metric("Min cementitious", f"{min_c} kg/m³")
k[2].metric("Target mean strength", f"{target:.2f} MPa", f"fck + 1.65×{sd_for_grade(grade, site):.1f}", delta_color="off")
k[3].metric("Design water", f"{water:.0f} kg/m³")
k[4].metric("Strength basis", f"×{cal_factor:.2f}", "your lab data" if cal_pairs else "IS default", delta_color="off")

if target > max(CURVES[curve]):
    st.error(f"Target {target:.1f} MPa is above what IS Fig. 1 curve {curve} reaches ({max(CURVES[curve]):.0f} MPa). Choose a higher cement grade or a lower concrete grade.")

if cal_pairs and not 0.7 <= cal_factor <= 1.4:
    st.warning(f"Your lab result is {cal_factor:.2f}× the IS curve — unusually far from it. Re-check the w/c, cement grade and cube results entered.")
conv = conventional_mix(inputs)
if conv[CEM] > MAX_CEMENT:
    st.warning(f"Conventional mix needs {conv[CEM]:.0f} kg/m³ cement, above the {MAX_CEMENT:.0f} kg/m³ limit. Use a higher-grade cement, SCM, or water-reducing admixture.")


def build_pdf(conv, opt=None, df=None):
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    sty = getSampleStyleSheet()
    story = [Paragraph("Concrete Mix Optimization – Report", sty["Title"]),
             Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", sty["Normal"]), Spacer(1, 8),
             Paragraph("Inputs", sty["Heading2"]),
             Paragraph(f"{ctype}, {grade} (target mean {target:.2f} MPa), exposure {exposure}, max w/c {max_wc:.2f}, min cementitious {min_c} kg/m3; "
                       f"slump {slump} mm; {agg_size} mm aggregate, {fine_zone}; {cement_choice}; SCM: {scm_type}. Priority: {preset}. "
                       f"Strength basis: " + (f"IS curve x {cal_factor:.2f} (calibrated with lab result)." if cal_pairs else "generic IS Fig. 1 curve."), sty["BodyText"]),
             Spacer(1, 6), Paragraph("Conclusion", sty["Heading2"])]
    if opt is not None:
        dc, ds = opt["Cost change vs conventional (%)"], opt["Strength vs conventional (%)"]
        story.append(Paragraph(f"Best optimized mix vs conventional: cost {dc:+.1f}%, cement {opt[CEM] - conv[CEM]:+.0f} kg/m3, strength {ds:+.1f}% "
                               f"({opt[SC]:.1f} MPa; target {target:.1f} MPa).", sty["BodyText"]))
    else:
        story.append(Paragraph(f"Conventional mix (IS 10262 procedure): w/c {conv['W/C ratio']:.2f}, cement {conv[CEM]:.0f} kg/m3, predicted strength "
                               f"{conv[SC]:.1f} MPa (target {target:.1f} MPa), cost Rs {conv[CC]:,.0f}/m3. No SCM selected.", sty["BodyText"]))
    story.append(Spacer(1, 6))
    head = ["Parameter", "Conventional"] + (["Optimized"] if opt is not None else [])
    data = [head]
    for lab, key in [("w/c", "W/C ratio"), ("Water kg/m3", "Water (kg/m³)"), ("Cement kg/m3", CEM), ("SCM kg/m3", "SCM (kg/m³)"),
                     ("Fine agg. (SSD) kg/m3", "Fine aggregate (SSD kg/m³)"), ("Coarse agg. (SSD) kg/m3", "Coarse aggregate (SSD kg/m³)"),
                     ("Predicted 28-day MPa", SC), ("Cost Rs/m3", CC), ("CO2 kg/m3", CO2K)]:
        data.append([lab, f"{conv[key]:.2f}"] + ([f"{opt[key]:.2f}"] if opt is not None else []))
    tb = Table(data, repeatRows=1, colWidths=[190, 130, 150][:len(head)])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445C")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(tb)
    if df is not None and not df.empty:
        td = [["Rank", "w/c", "SCM %", "Cement", "Strength", "Cost", "Score"]] + [
            [int(r["Rank"]), f"{r['W/C ratio']:.2f}", f"{r['SCM replacement (%)']:.0f}", f"{r[CEM]:.0f}", f"{r[SC]:.1f}", f"{r[CC]:.0f}", f"{r['Score (0-100)']:.1f}"]
            for _, r in df.head(5).iterrows()]
        t5 = Table(td, repeatRows=1)
        t5.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF0"))]))
        story += [Spacer(1, 8), Paragraph("Top 5 ranked mixes", sty["Heading2"]), t5]
    story += [Spacer(1, 8), Paragraph("Limitations", sty["Heading2"]),
              Paragraph("Strength is read from digitised IS 10262:2019 Fig. 1 curves with assumed SCM efficiency (about +/-10%). This is a screening tool; "
                        "verify against the licensed IS codes and confirm with laboratory trial mixes and qualified review before field use.", sty["BodyText"])]
    doc.build(story)
    return buff.getvalue()


tR, tC, tV, tM = st.tabs(["🏆 Results & ranking", "📐 Conventional mix", "✅ IS validation", "ℹ️ Method & limits"])

# ------------------------------------------------------------------ helpers
def pair_chart(title, a, b, unit, target_line=None):
    fig = go.Figure(go.Bar(x=["Conventional", "Optimized"], y=[a, b], text=[f"{a:,.1f}", f"{b:,.1f}"],
                           textposition="outside", marker_color=["#8a959e", "#1f9d55"]))
    d = (b / a - 1) * 100 if a else 0
    top = max(a, b, target_line or 0) * 1.2
    fig.update_layout(title=f"{title}  ({d:+.1f}%)", yaxis_title=unit, showlegend=False, height=340, margin=dict(t=55, b=20))
    fig.update_yaxes(range=[0, top])
    if target_line:
        fig.add_hline(y=target_line, line_dash="dash", line_color="red", annotation_text=f"Target {target_line:.1f}")
    return fig


def curve_chart(conv, opt):
    fig = go.Figure()
    for c in (1, 2, 3):
        sel = c == curve
        fig.add_trace(go.Scatter(x=CURVE_WC, y=[v * (cal_factor if sel else 1.0) for v in CURVES[c]], mode="lines",
                                 name=f"Curve {c}" + (" (selected" + (", lab-calibrated)" if cal_pairs else ")") if sel else ""),
                                 line=dict(width=4 if sel else 1.5, color="#1f77b4" if sel else "#c0c7cc", dash="solid" if sel else "dash")))
    fig.add_trace(go.Scatter(x=[conv["Effective w/b for strength"]], y=[conv[SC]], mode="markers+text", text=["Conventional"],
                             textposition="top right", marker=dict(size=14, color="#555", symbol="diamond"), name="Conventional"))
    fig.add_trace(go.Scatter(x=[opt["Effective w/b for strength"]], y=[opt[SC]], mode="markers+text", text=["Optimized"],
                             textposition="bottom left", marker=dict(size=15, color="#1f9d55", symbol="star"), name="Optimized"))
    if cal_pairs:
        fig.add_trace(go.Scatter(x=[p[0] for p in cal_pairs], y=[p[1] for p in cal_pairs], mode="markers", name="Your lab test",
                                 marker=dict(size=12, color="#d62728", symbol="x")))
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"Target mean {target:.1f} MPa")
    fig.update_layout(title="IS 10262 Fig. 1 — strength vs effective water/binder ratio", xaxis_title="Effective w/b (cement + k·SCM)",
                      yaxis_title="28-day strength (MPa)", height=420)
    return fig


# ================================================================= RESULTS
with tR:
    if scm_type == "None":
        st.info("No SCM selected, so the conventional mix is already the minimum-cement mix that meets the target. "
                "Pick an SCM (fly ash / GGBS / silica fume) in the sidebar to search for cheaper or greener alternatives.")
        st.dataframe(pd.DataFrame({"Item": ["w/c", "Cement (kg/m³)", "Predicted strength (MPa)", "Cost (₹/m³)"],
                                   "Conventional": [conv["W/C ratio"], conv[CEM], conv[SC], conv[CC]]}).round(2),
                     width="stretch", hide_index=True)
        st.download_button("📄 Download PDF report", build_pdf(conv), "concrete_mix_report.pdf", "application/pdf")
    else:
        raw = optimize_mixes(inputs, scm_range)
        df = rank_mixes(raw, conv, PRESETS[preset], target, max_loss)
        if raw.empty:
            st.warning("No mix satisfies the durability and cement limits. Try a higher-grade cement, a different SCM or a water-reducing admixture.")
        elif df.empty:
            st.warning(f"{len(raw)} mixes were generated but none reached the target mean strength ({target:.1f} MPa) within a {max_loss}% drop from conventional. "
                       "Raise the allowed drop, use a higher cement grade or a more efficient SCM.")
        else:
            opt = df.iloc[0].to_dict()
            dc, ds = opt["Cost change vs conventional (%)"], opt["Strength vs conventional (%)"]
            dcem = opt[CEM] - conv[CEM]
            msg = (f"**Best mix vs conventional:** cost **{dc:+.1f}%**, cement **{dcem:+.0f} kg/m³**, CO₂ **{-opt['CO2 saved vs conventional (%)']:+.1f}%**, "
                   f"strength **{ds:+.1f}%** ({opt[SC]:.1f} MPa vs target {target:.1f} MPa).")
            (st.success if dc < 0 else st.warning)(msg + ("" if dc < 0 else " At these rates no passing SCM mix is cheaper — check SCM rates or the ranking priority."))
            m = st.columns(4)
            m[0].metric("Cost / m³", f"₹{opt[CC]:,.0f}", f"{dc:+.1f}% vs conv", delta_color="inverse")
            m[1].metric("Predicted strength", f"{opt[SC]:.1f} MPa", f"{ds:+.1f}% vs conv")
            m[2].metric("Cement", f"{opt[CEM]:.0f} kg/m³", f"{dcem:+.0f} kg", delta_color="inverse")
            m[3].metric("Embodied CO₂", f"{opt[CO2K]:.0f} kg/m³", f"{-opt['CO2 saved vs conventional (%)']:+.1f}%", delta_color="inverse")
            if cal_pairs:
                st.caption(f"Strength is calibrated with your lab result (IS curve × {cal_factor:.2f}). SCM efficiency is still an assumption — cube-test the top-ranked mixes before use.")
            else:
                st.caption("Strength is read from the generic IS Fig. 1 curve (about ±10%). For more reliable results, test your conventional mix and enter it in section 6 of the sidebar.")

            st.subheader("Conventional vs optimized")
            g = st.columns(3)
            g[0].plotly_chart(pair_chart("Strength", conv[SC], opt[SC], "MPa", target), width="stretch")
            g[1].plotly_chart(pair_chart("Cost", conv[CC], opt[CC], "₹/m³"), width="stretch")
            g[2].plotly_chart(pair_chart("Cement consumption", conv[CEM], opt[CEM], "kg/m³"), width="stretch")
            st.plotly_chart(curve_chart(conv, opt), width="stretch")

            st.subheader("🏅 Ranked mixes")
            topn = st.slider("Show top N", 3, 20, 8)
            cols = ["Rank", "W/C ratio", "SCM replacement (%)", CEM, "SCM (kg/m³)", "Water (kg/m³)", SC, "Strength vs conventional (%)",
                    CC, "Cost change vs conventional (%)", "Cement saved vs conventional (%)", "CO2 saved vs conventional (%)", "Score (0-100)"]
            top = df.head(topn)[cols]
            fmt = {c: "{:.1f}" for c in cols if c not in ("Rank", "W/C ratio")}
            fmt["W/C ratio"] = "{:.2f}"
            st.dataframe(top.style.format(fmt), width="stretch", hide_index=True)

            st.subheader("Mix proportions per m³ (best mix)")
            rows = [("w/c", "W/C ratio", 2), ("Water", "Water (kg/m³)", 1), ("Cement", CEM, 1), ("SCM", "SCM (kg/m³)", 1),
                    ("Fine aggregate (SSD)", "Fine aggregate (SSD kg/m³)", 1), ("Coarse aggregate (SSD)", "Coarse aggregate (SSD kg/m³)", 1),
                    ("Fine aggregate (as batched)", "Fine aggregate (batch kg/m³)", 1), ("Coarse aggregate (as batched)", "Coarse aggregate (batch kg/m³)", 1),
                    ("Batch water (moisture-corrected)", "Batch water (kg/m³)", 1)]
            st.dataframe(pd.DataFrame({"Material": [r[0] for r in rows],
                                       "Conventional": [round(conv[r[1]], r[2]) for r in rows],
                                       "Optimized": [round(opt[r[1]], r[2]) for r in rows]}), width="stretch", hide_index=True)

            vol = st.number_input("Project concrete volume (m³)", 1.0, 100000.0, 50.0, 1.0)
            st.info(f"For {vol:,.0f} m³ versus conventional: cost **₹{(conv[CC] - opt[CC]) * vol:,.0f}** saved, "
                    f"cement **{(conv[CEM] - opt[CEM]) * vol / 1000:,.1f} t** saved, CO₂ **{(conv[CO2K] - opt[CO2K]) * vol / 1000:,.1f} t** saved.")

            with st.expander("All passing candidates: cost vs strength"):
                s = df.sort_values(CC)
                best, fx, fy = -1, [], []
                for _, r in s.iterrows():
                    if r[SC] > best:
                        best = r[SC]; fx.append(r[CC]); fy.append(r[SC])
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=df[CC], y=df[SC], mode="markers", name="Passing mixes", marker=dict(size=6, color="#9db7d5")))
                f2.add_trace(go.Scatter(x=fx, y=fy, mode="lines", name="Best trade-off", line=dict(color="#1f9d55", width=3)))
                f2.add_trace(go.Scatter(x=[conv[CC]], y=[conv[SC]], mode="markers", name="Conventional", marker=dict(size=13, color="#333", symbol="diamond")))
                f2.add_hline(y=target, line_dash="dash", line_color="red")
                f2.update_layout(xaxis_title="Cost (₹/m³)", yaxis_title="Predicted strength (MPa)", height=400)
                st.plotly_chart(f2, width="stretch")

            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Ranked mixes (CSV)", df.to_csv(index=False).encode(), "ranked_mixes.csv", "text/csv")
            d2.download_button("📄 Download PDF report", build_pdf(conv, opt, df), "concrete_mix_report.pdf", "application/pdf")

# ================================================================= CONVENTIONAL
with tC:
    st.subheader("Conventional (IS 10262 procedure) mix")
    st.write(f"w/c taken from Fig. 1 for the target mean strength, then capped by IS 456 durability. **Governed by:** {conv['Governed by']}.")
    show = {k_: v for k_, v in conv.items() if k_ != "Governed by"}
    st.dataframe(pd.DataFrame({"Value": show}).round(3), width="stretch")
    st.caption(f"Coarse aggregate fraction = Table 3 value ({CA_VOLUME[agg_size][fine_zone]:.2f}) adjusted ±0.01 per 0.05 change in w/c" + (", then −10% for pumping." if pumping else "."))

# ================================================================= VALIDATION
with tV:
    st.subheader("Check against the IS 10262 worked example (M40, OPC 43, 20 mm, 100 mm slump, severe RCC)")
    ex = MixInputs(grade="M40", concrete_type="RCC", exposure="Severe", slump_mm=100, agg_size_mm=20, fine_zone="Zone I",
                   cement_type="OPC", opc_grade="43", water_reduction_pct=29, pumping=True)
    exm = calculate_mix(ex, 0.40, 0.0)
    ex_wc, ex_minc, _ = durability_limits("RCC", "Severe")
    checks = [("Target mean strength (MPa)", target_mean_strength("M40"), 48.25, 0.01),
              ("Max w/c (IS 456 Table 5)", ex_wc, 0.45, 0.001),
              ("Min cement (kg/m³)", ex_minc, 320, 0.5),
              ("Water for 100 mm slump (kg/m³)", estimate_water(20, 100, "Angular", 0), 197, 0.5),
              ("Water after 29% reduction (kg/m³)", design_water(ex), 140, 0.5),
              ("Cement at w/c 0.40 (kg/m³)", exm[CEM], 350, 1.0)]
    vt = pd.DataFrame([{"Item": a, "This tool": round(b, 2), "IS example": c, "Status": "✅" if abs(b - c) <= tol else "❌"} for a, b, c, tol in checks])
    st.dataframe(vt, width="stretch", hide_index=True)
    st.write(f"Coarse aggregate fraction (Zone I, 20 mm, w/c 0.40): Table 3 + adjustment = **{coarse_fraction(20, 'Zone I', 0.40, False):.3f}**; "
             f"with 10% pumping reduction = **{coarse_fraction(20, 'Zone I', 0.40, True):.3f}**.")
    st.write(f"IS Fig. 1 (curve 2) gives w/c ≈ **{wc_for_strength(2, 48.25):.2f}** for 48.25 MPa; the example adopted 0.40 'based on experience' (with superplasticiser).")
    st.caption("The ❌/✅ test compares code arithmetic only. Fig. 1 values are digitised from the figure (reading error about ±3 MPa).")
    fig = go.Figure()
    for c_ in (1, 2, 3):
        fig.add_trace(go.Scatter(x=CURVE_WC, y=CURVES[c_], mode="lines+markers", name=f"Curve {c_}"))
    fig.update_layout(title="Digitised IS Fig. 1 curves used (fixed)", xaxis_title="Free w/c", yaxis_title="28-day strength (MPa)", height=380)
    st.plotly_chart(fig, width="stretch")

# ================================================================= METHOD
with tM:
    st.markdown("""
**How it works**
1. Durability (max w/c, min cementitious, min grade) comes from IS 456 Table 5 and is locked once type and exposure are chosen.
2. Target mean strength = fck + 1.65 s (s from IS 10262 Table 2, +1 for fair site control).
3. Water from IS 10262 Table 4, reduced for aggregate type (10 / 15 / 20 kg), raised/lowered 3% per 25 mm slump from 50 mm, and optionally cut by an admixture.
4. Coarse aggregate volume from Table 3 (size × sand zone), ±0.01 per 0.05 change in w/c, −10% for pumping. Volumes are on SSD basis; as-batched masses use your moisture/absorption.
5. **Lab calibration (optional):** if you enter the measured strength of your tested cement-only mix, the strength curve is scaled by measured ÷ IS value for your materials.
6. **Conventional mix:** w/c from Fig. 1 for the target strength, capped by durability, no SCM.
7. **Optimized mixes:** every w/c (0.01 steps) and SCM % (5% steps, within typical limits) that meets durability, the 450 kg/m³ cement cap, the target mean strength and your maximum strength drop. Ranked by a weighted cost / strength / CO₂ score.

**Limitations**
- Calibration fixes the strength level for your materials but not the SCM contribution: cube-test the top mixes.
- Fig. 1 is for cement; strength of SCM mixes uses an assumed efficiency factor k (editable). Treat as ±10%.
- CO₂ factors are indicative. Admixture cost and transport are not included.
- IS 10262 requires trial mixes: confirm any mix with cubes before use. Verify limits against the licensed IS 456 / IS 10262 documents.
""")
    st.markdown("[Bureau of Indian Standards](https://www.bis.gov.in/know-your-standard/?lang=en)")
