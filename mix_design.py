"""Concrete mix design + optimisation engine (IS 10262:2019 / IS 456 tables).

Educational screening tool. Code tables are fixed constants; material data and
rates are user inputs. Strength comes from fixed digitised IS 10262:2019 Fig. 1
curves (reading error about +/-3 MPa) - always confirm with trial cubes.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- code tables
GRADES = [f"M{x}" for x in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)]
MAX_CEMENT = 450.0  # kg/m3, IS 456 cl. 8.2.4.2 (excl. SCM)

# IS 456 Table 5: (max free w/c, min cementitious kg/m3, min grade)
DURABILITY = {
    "PCC": {"Mild": (0.60, 220, "M10"), "Moderate": (0.60, 240, "M15"), "Severe": (0.50, 250, "M20"),
            "Very severe": (0.45, 260, "M20"), "Extreme": (0.40, 280, "M25")},
    "RCC": {"Mild": (0.55, 300, "M20"), "Moderate": (0.50, 300, "M25"), "Severe": (0.45, 320, "M30"),
            "Very severe": (0.45, 340, "M35"), "Extreme": (0.40, 360, "M40")},
}
EXPOSURES = list(DURABILITY["RCC"].keys())

# Workability degree -> (slump min, slump max, default) mm  (IS 456 cl. 7.1)
WORKABILITY = {"Low": (25, 75, 50), "Medium": (50, 100, 75), "High": (100, 150, 125)}

# IS 10262 Table 4 water (kg/m3) for 50 mm slump, angular aggregate
WATER_BASE = {10: 208.0, 20: 186.0, 40: 165.0}
# water reduction (kg) for other coarse aggregate types
AGG_SHAPE_REDUCTION = {"Angular": 0.0, "Sub-angular": 10.0,
                       "Gravel with some crushed particles": 15.0, "Rounded gravel": 20.0}
AIR_CONTENT = {10: 0.015, 20: 0.010, 40: 0.008}  # entrapped air, fraction of volume
# Coarse aggregate volume per unit total aggregate volume (w/c 0.50)
CA_VOLUME = {10: {"Zone IV": 0.50, "Zone III": 0.48, "Zone II": 0.46, "Zone I": 0.44},
             20: {"Zone IV": 0.66, "Zone III": 0.64, "Zone II": 0.62, "Zone I": 0.60},
             40: {"Zone IV": 0.75, "Zone III": 0.73, "Zone II": 0.71, "Zone I": 0.69}}

# SCM: typical replacement limits (% of total binder) and indicative price ranges (Rs/kg)
SCM_LIMITS = {"Fly ash": (15, 35), "GGBS": (25, 70), "Silica fume": (5, 10)}
SCM_SG = {"Fly ash": 2.2, "GGBS": 2.9, "Silica fume": 2.2}
SCM_RATE_RANGE = {"Fly ash": {"Bulk": (0.8, 1.5), "Bagged": (1.2, 2.4)},
                  "GGBS": {"Bulk": (4.5, 6.5), "Bagged": (25.0, 35.0)},
                  "Silica fume": {"Bulk": (4.8, 8.0), "Bagged": (25.0, 35.0)}}
# assumed 28-day cementing efficiency of SCM (editable in the app; assumption, not an IS value)
SCM_K = {"None": 0.0, "Fly ash": 0.5, "GGBS": 0.8, "Silica fume": 2.0}
CO2 = {"cement": 0.90, "Fly ash": 0.02, "GGBS": 0.07, "Silica fume": 0.02, "fa": 0.005, "ca": 0.008}  # kg/kg indicative

# IS 10262:2019 Fig. 1 - digitised, FIXED (curve 1: OPC33, 2: OPC43/PPC, 3: OPC53)
CURVE_WC = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
CURVES = {1: [64.5, 55.0, 47.0, 40.0, 34.0, 28.5, 24.0, 19.5, 14.7],
          2: [70.5, 61.5, 53.0, 46.0, 40.0, 34.5, 29.5, 24.5, 19.3],
          3: [74.0, 66.0, 59.0, 52.5, 46.5, 40.5, 35.0, 29.8, 24.7]}

WC_STEP = 0.01      # internal search resolution (not user-facing)
SCM_STEP = 5.0

SC = "Predicted 28-day strength (MPa)"
CC = "Estimated material cost (₹/m³)"
CEM = "Cement (kg/m³)"
CO2K = "Embodied CO2 (kg/m³)"


# ----------------------------------------------------------------- helpers
def durability_limits(concrete_type, exposure):
    """(max w/c, min cementitious kg/m3, min grade)."""
    return DURABILITY[concrete_type][exposure]


def valid_grades(concrete_type, exposure):
    mn = int(durability_limits(concrete_type, exposure)[2][1:])
    return [g for g in GRADES if int(g[1:]) >= mn]


def sd_for_grade(grade, site_control="Good"):
    f = int(grade[1:])
    s = 3.5 if f <= 15 else 4.0 if f <= 25 else 5.0
    return s + (1.0 if site_control == "Fair" else 0.0)


def target_mean_strength(grade, site_control="Good"):
    return float(grade[1:]) + 1.65 * sd_for_grade(grade, site_control)


def estimate_water(agg_size, slump, shape="Angular", reduction_pct=0.0):
    w = WATER_BASE[int(agg_size)] - AGG_SHAPE_REDUCTION[shape]
    w *= 1 + 0.03 * (float(slump) - 50.0) / 25.0
    return w * (1 - float(reduction_pct) / 100.0)


def coarse_fraction(agg_size, zone, wc, pumping=False):
    cf = CA_VOLUME[int(agg_size)][zone] + 0.01 * (0.50 - wc) / 0.05
    return cf * (0.9 if pumping else 1.0)


def curve_for(cement_type, opc_grade="43"):
    if cement_type == "PPC":
        return 2
    return {"33": 1, "43": 2, "53": 3}[str(opc_grade)]


def curve_strength(curve, wb):
    return float(np.interp(wb, CURVE_WC, CURVES[curve]))


def wc_for_strength(curve, target):
    return float(np.interp(target, CURVES[curve][::-1], CURVE_WC[::-1]))


def calibration_factor(curve, pairs):
    """Lab calibration: mean of measured / IS-curve strength over tested cement-only mixes [(w/c, MPa), ...]."""
    ratios = [f / curve_strength(curve, w) for w, f in pairs if w > 0 and f > 0]
    return float(sum(ratios) / len(ratios)) if ratios else 1.0


def frange(a, b, step):
    out, x = [], float(a)
    while x <= float(b) + 1e-9:
        out.append(round(x, 4))
        x += step
    return out or [round(float(a), 4)]


@dataclass
class MixInputs:
    grade: str = "M25"
    concrete_type: str = "RCC"
    exposure: str = "Moderate"
    site_control: str = "Good"
    slump_mm: int = 75
    agg_size_mm: int = 20
    fine_zone: str = "Zone II"
    agg_shape: str = "Angular"
    pumping: bool = False
    water_reduction_pct: float = 0.0
    cement_type: str = "OPC"
    opc_grade: str = "43"
    scm_type: str = "None"
    scm_k: float = 0.5
    fine_sg: float = 2.60
    coarse_sg: float = 2.70
    cement_sg: float = 3.15
    scm_sg: float = 2.20
    fine_moisture_pct: float = 2.0
    coarse_moisture_pct: float = 1.0
    fine_absorption_pct: float = 1.0
    coarse_absorption_pct: float = 0.5
    cement_rate: float = 8.0   # Rs/kg
    scm_rate: float = 0.0
    fa_rate: float = 2.0
    ca_rate: float = 2.0
    water_rate: float = 1.35
    strength_factor: float = 1.0   # lab calibration of IS Fig. 1 (1.0 = IS default)


def design_water(i: MixInputs):
    return estimate_water(i.agg_size_mm, i.slump_mm, i.agg_shape, i.water_reduction_pct)


def calculate_mix(i: MixInputs, wc: float, scm_pct: float = 0.0) -> dict:
    if i.scm_type == "None":
        scm_pct = 0.0
    if not 0.2 <= wc <= 0.8:
        raise ValueError("w/c out of range")
    water = design_water(i)
    binder = water / wc
    scm = binder * scm_pct / 100.0
    cement = binder - scm
    air = AIR_CONTENT[int(i.agg_size_mm)]
    av = 1 - cement / (i.cement_sg * 1000) - (scm / (i.scm_sg * 1000) if scm else 0) - water / 1000 - air
    if av <= 0.3:
        raise ValueError("aggregate volume too small")
    cf = coarse_fraction(i.agg_size_mm, i.fine_zone, wc, i.pumping)
    ca = av * cf * i.coarse_sg * 1000
    fa = av * (1 - cf) * i.fine_sg * 1000
    # SSD design masses -> as-batched masses (moisture on dry mass)
    fa_dry = fa / (1 + i.fine_absorption_pct / 100)
    ca_dry = ca / (1 + i.coarse_absorption_pct / 100)
    fa_batch = fa_dry * (1 + i.fine_moisture_pct / 100)
    ca_batch = ca_dry * (1 + i.coarse_moisture_pct / 100)
    free = fa_dry * (i.fine_moisture_pct - i.fine_absorption_pct) / 100 + ca_dry * (i.coarse_moisture_pct - i.coarse_absorption_pct) / 100
    eff = cement + (i.scm_k if i.scm_type != "None" else 0.0) * scm
    wb_eff = water / eff
    fc = curve_strength(curve_for(i.cement_type, i.opc_grade), wb_eff) * i.strength_factor
    cost = cement * i.cement_rate + scm * (i.scm_rate if scm else 0) + fa * i.fa_rate + ca * i.ca_rate + water * i.water_rate
    co2 = cement * CO2["cement"] + scm * CO2.get(i.scm_type, 0) + fa * CO2["fa"] + ca * CO2["ca"]
    return {"W/C ratio": wc, "Water (kg/m³)": water, CEM: cement, "SCM (kg/m³)": scm, "Total binder (kg/m³)": binder,
            "SCM replacement (%)": scm_pct, "Fine aggregate (SSD kg/m³)": fa, "Coarse aggregate (SSD kg/m³)": ca,
            "Fine aggregate (batch kg/m³)": fa_batch, "Coarse aggregate (batch kg/m³)": ca_batch,
            "Batch water (kg/m³)": water - free, "Coarse aggregate fraction": cf,
            "Effective w/b for strength": wb_eff, SC: fc, CC: cost,
            "Cost per MPa (₹/m³/MPa)": cost / max(fc, 1e-9), CO2K: co2}


def conventional_mix(i: MixInputs) -> dict:
    """IS 10262 procedure: w/c from Fig.1 for target strength, checked against durability, no SCM."""
    max_wc, min_c, _ = durability_limits(i.concrete_type, i.exposure)
    target = target_mean_strength(i.grade, i.site_control)
    wc_t = math.floor(wc_for_strength(curve_for(i.cement_type, i.opc_grade), target / i.strength_factor) * 100 + 1e-9) / 100
    wc, why = wc_t, "Strength (IS Fig. 1)"
    if wc > max_wc:
        wc, why = max_wc, "Durability max w/c"
    water = design_water(i)
    if water / wc < min_c:
        wc, why = math.floor(water / min_c * 1000) / 1000, "Durability min cement"
    r = calculate_mix(i, wc, 0.0)
    r["Governed by"] = why
    return r


def optimize_mixes(i: MixInputs, scm_range=None) -> pd.DataFrame:
    """All candidates satisfying durability + cement limits (no strength filter yet)."""
    max_wc, min_c, _ = durability_limits(i.concrete_type, i.exposure)
    water = design_water(i)
    if i.scm_type == "None":
        scms = [0.0]
    else:
        lo, hi = scm_range or SCM_LIMITS[i.scm_type]
        scms = frange(lo, hi, SCM_STEP)
    rows = []
    for repl in scms:
        for wc in frange(0.25, max_wc, WC_STEP):
            if water / wc < min_c - 1e-9:
                continue
            try:
                r = calculate_mix(i, wc, repl)
            except ValueError:
                continue
            if r[CEM] > MAX_CEMENT:
                continue
            rows.append(r)
    return pd.DataFrame(rows)


PRESETS = {"Balanced": (0.35, 0.40, 0.25), "Lowest cost": (0.70, 0.20, 0.10),
           "Strength-focused": (0.20, 0.65, 0.15), "Eco-friendly": (0.25, 0.25, 0.50)}  # cost, strength, CO2


def rank_mixes(df, conv, weights, target, max_loss_pct):
    """Keep mixes meeting target mean strength and limited strength drop; rank by weighted 0-100 score."""
    if df.empty:
        return df
    d = df.copy()
    d["Strength vs conventional (%)"] = (d[SC] / conv[SC] - 1) * 100
    d["Cost change vs conventional (%)"] = (d[CC] / conv[CC] - 1) * 100
    d["Cement saved vs conventional (%)"] = (1 - d[CEM] / conv[CEM]) * 100
    d["CO2 saved vs conventional (%)"] = (1 - d[CO2K] / conv[CO2K]) * 100
    d = d[(d[SC] >= target) & (d["Strength vs conventional (%)"] >= -max_loss_pct)].copy()
    if d.empty:
        return d

    def norm(x, high_good):
        r = x.max() - x.min()
        n = (x - x.min()) / r if r > 1e-9 else x * 0 + 1.0
        return n if high_good else 1 - n

    wc_, ws_, we_ = weights
    s_eff = d[SC].clip(upper=conv[SC])  # no extra credit for strength above the conventional mix (avoids over-design)
    d["Score (0-100)"] = 100 * (wc_ * norm(d[CC], False) + ws_ * norm(s_eff, True) + we_ * norm(d[CO2K], False)) / (wc_ + ws_ + we_)
    d = d.sort_values(["Score (0-100)", CC], ascending=[False, True]).reset_index(drop=True)
    d.insert(0, "Rank", range(1, len(d) + 1))
    return d
