import re
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split

DATA_PATH = "NEW-DATA-1.T15.txt"

COMFORT_LOW = 21.0   # comfort band lower bound
COMFORT_HIGH = 24.0  # comfort band upper bound
OCC_TH = 0.5         # occupancy threshold for HVAC ON

def pct_range(a, lo=1, hi=99, pad=0.05):
    """
    Helper to get [lo, hi] percentile range with padding.
    """
    lo_v, hi_v = np.percentile(a, [lo, hi])
    span = hi_v - lo_v
    return lo_v - pad * span, hi_v + pad * span


def load_and_prepare():
    """
    Load NEW-DATA-1.T15.txt, build Datetime, target T_future_15, and Delta15.
    Returns a cleaned DataFrame.
    """
    raw = pd.read_csv(DATA_PATH, header=None)
    header_line = raw.iloc[0, 0]

    # Extract column names from weird header line
    names = re.findall(r"\d+:(\S+)", header_line)

    # Real data: skip first header line
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        names=names,
    )

    # Build Datetime
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    # Keep only relevant columns
    df = df[
        [
            "Datetime",
            "Temperature_Comedor_Sensor",
            "Temperature_Exterior_Sensor",
            "CO2_Comedor_Sensor",
            "Lighting_Comedor_Sensor",
        ]
    ].copy()

    # 15-min ahead temperature target
    df["T_future_15"] = df["Temperature_Comedor_Sensor"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    # Delta15 = T(t+15) - T(t)
    df["Delta15"] = df["T_future_15"] - df["Temperature_Comedor_Sensor"]

    return df

def build_occupancy_system(df):
    """
    Build fuzzy system: (CO2, Lighting) → occ_fuzzy in [0,1].
    """
    co2 = df["CO2_Comedor_Sensor"].values
    light = df["Lighting_Comedor_Sensor"].values

    co2_lo, co2_hi = pct_range(co2, 1, 99, 0.05)
    light_lo, light_hi = pct_range(light, 1, 99, 0.05)

    co2_var = ctrl.Antecedent(np.linspace(co2_lo, co2_hi, 401), "co2")
    light_var = ctrl.Antecedent(np.linspace(light_lo, light_hi, 401), "light")
    occ_var = ctrl.Consequent(np.linspace(0, 1, 201), "occ")

    # CO2: low / medium / high
    co2_var["low"] = fuzz.trimf(
        co2_var.universe,
        [co2_lo, co2_lo, (co2_lo + co2_hi) / 2],
    )
    co2_var["medium"] = fuzz.trimf(
        co2_var.universe,
        [co2_lo, (co2_lo + co2_hi) / 2, co2_hi],
    )
    co2_var["high"] = fuzz.trimf(
        co2_var.universe,
        [(co2_lo + co2_hi) / 2, co2_hi, co2_hi],
    )

    # Lighting: dark / dim / bright
    light_var["dark"] = fuzz.trimf(
        light_var.universe,
        [light_lo, light_lo, (light_lo + light_hi) / 2],
    )
    light_var["dim"] = fuzz.trimf(
        light_var.universe,
        [light_lo, (light_lo + light_hi) / 2, light_hi],
    )
    light_var["bright"] = fuzz.trimf(
        light_var.universe,
        [(light_lo + light_hi) / 2, light_hi, light_hi],
    )

    # Occupancy: low / medium / high (0–1)
    occ_var["low"] = fuzz.trimf(occ_var.universe, [0.0, 0.0, 0.4])
    occ_var["medium"] = fuzz.trimf(occ_var.universe, [0.2, 0.5, 0.8])
    occ_var["high"] = fuzz.trimf(occ_var.universe, [0.6, 1.0, 1.0])

    # Rules for occupancy
    rules_occ = [
        ctrl.Rule(light_var["dark"] & co2_var["low"], occ_var["low"]),
        ctrl.Rule(light_var["bright"] & co2_var["low"], occ_var["medium"]),
        ctrl.Rule(light_var["dim"] & co2_var["medium"], occ_var["medium"]),
        ctrl.Rule(light_var["bright"] & co2_var["medium"], occ_var["high"]),
        ctrl.Rule(light_var["bright"] & co2_var["high"], occ_var["high"]),
        ctrl.Rule(light_var["dim"] & co2_var["high"], occ_var["high"]),
    ]

    system = ctrl.ControlSystem(rules_occ)
    return system


def compute_occ_fuzzy(df, occ_system):
    """
    Run occ_system over all rows in df and add 'occ_fuzzy' column.
    """
    sim = ctrl.ControlSystemSimulation(occ_system)
    occ_vals = np.zeros(len(df))

    for i in range(len(df)):
        sim.input["co2"] = float(df.iloc[i]["CO2_Comedor_Sensor"])
        sim.input["light"] = float(df.iloc[i]["Lighting_Comedor_Sensor"])
        sim.compute()
        occ_vals[i] = sim.output["occ"]

    df = df.copy()
    df["occ_fuzzy"] = occ_vals
    return df

def build_delta_system(df_with_occ):
    """
    Build Mamdani FIS: (indoor T, outdoor T, occ_fuzzy) → Delta15.
    Returns (system, (d_lo, d_hi)).
    """
    X = df_with_occ[
        [
            "Temperature_Comedor_Sensor",
            "Temperature_Exterior_Sensor",
            "occ_fuzzy",
        ]
    ]
    y = df_with_occ["T_future_15"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    indo_train = X_train["Temperature_Comedor_Sensor"].values
    out_train = X_train["Temperature_Exterior_Sensor"].values
    occ_train = X_train["occ_fuzzy"].values
    Delta15_train = y_train.values - indo_train

    # Universe ranges
    indo_lo, indo_hi = pct_range(indo_train, 1, 99, 0.05)
    out_lo, out_hi = pct_range(out_train, 1, 99, 0.05)
    occ_lo, occ_hi = pct_range(occ_train, 1, 99, 0.05)
    d_lo, d_hi = pct_range(Delta15_train, 5, 95, 0.03)

    indoor = ctrl.Antecedent(np.linspace(indo_lo, indo_hi, 401), "indoor")
    outdoor = ctrl.Antecedent(np.linspace(out_lo, out_hi, 401), "outdoor")
    occupancy = ctrl.Antecedent(np.linspace(occ_lo, occ_hi, 201), "occupancy")
    delta = ctrl.Consequent(np.linspace(d_lo, d_hi, 401), "delta")

    # --- membership helpers (same as your notebook) ---
    def five_terms(var, lo, hi, names=("very_cold", "cold", "normal", "warm", "hot")):
        a, b, c, d, e = np.linspace(lo, hi, 5)
        var[names[0]] = fuzz.trapmf(var.universe, [lo, lo, a, b])
        var[names[1]] = fuzz.trimf(var.universe, [a, (a + b) / 2, c])
        var[names[2]] = fuzz.trimf(var.universe, [b, (b + c) / 2, d])
        var[names[3]] = fuzz.trimf(var.universe, [c, (c + d) / 2, e])
        var[names[4]] = fuzz.trapmf(var.universe, [d, e, hi, hi])

    def three_terms(var, lo, hi, names=("low", "medium", "high")):
        a, b, c = np.linspace(lo, hi, 3)
        var[names[0]] = fuzz.trapmf(var.universe, [lo, lo, a, b])
        var[names[1]] = fuzz.trimf(var.universe, [a, (a + b) / 2, c])
        var[names[2]] = fuzz.trapmf(var.universe, [b, c, hi, hi])

    def five_terms_delta(var, lo, hi):
        a, b, c, d, e = np.linspace(lo, hi, 5)
        var["strong_down"] = fuzz.trapmf(var.universe, [lo, lo, a, b])
        var["slight_down"] = fuzz.trimf(var.universe, [a, (a + b) / 2, c])
        var["stable"] = fuzz.trimf(var.universe, [b, (b + c) / 2, d])
        var["slight_up"] = fuzz.trimf(var.universe, [c, (c + d) / 2, e])
        var["strong_up"] = fuzz.trapmf(var.universe, [d, e, hi, hi])

    # --- apply membership functions ---
    five_terms(indoor, indo_lo, indo_hi)
    five_terms(outdoor, out_lo, out_hi)
    three_terms(occupancy, occ_lo, occ_hi)
    five_terms_delta(delta, d_lo, d_hi)

    occ_low = occupancy["low"]
    occ_med = occupancy["medium"]
    occ_high = occupancy["high"]

    # --- rules (your 18 rules) ---
    rules = [
        ctrl.Rule(indoor["very_cold"] & outdoor["hot"] & (occ_med | occ_high), delta["strong_up"]),
        ctrl.Rule(indoor["cold"] & outdoor["hot"] & (occ_med | occ_high), delta["strong_up"]),
        ctrl.Rule(indoor["very_cold"] & outdoor["warm"] & occ_high, delta["slight_up"]),
        ctrl.Rule(indoor["cold"] & outdoor["warm"] & occ_high, delta["slight_up"]),
        ctrl.Rule(indoor["normal"] & outdoor["warm"] & occ_low, delta["slight_up"]),
        ctrl.Rule(indoor["normal"] & outdoor["warm"] & occ_high, delta["slight_up"]),
        ctrl.Rule(indoor["normal"] & outdoor["hot"] & occ_med, delta["slight_up"]),
        ctrl.Rule(indoor["normal"] & outdoor["normal"] & occ_low, delta["stable"]),
        ctrl.Rule(indoor["normal"] & outdoor["normal"] & occ_high, delta["slight_up"]),
        ctrl.Rule(indoor["warm"] & outdoor["warm"] & (occ_low | occ_med), delta["stable"]),
        ctrl.Rule(indoor["cold"] & outdoor["cold"] & (occ_low | occ_med), delta["stable"]),
        ctrl.Rule(indoor["warm"] & outdoor["normal"] & occ_low, delta["slight_down"]),
        ctrl.Rule(indoor["warm"] & outdoor["normal"] & occ_high, delta["stable"]),
        ctrl.Rule(indoor["hot"] & outdoor["normal"] & occ_low, delta["slight_down"]),
        ctrl.Rule(indoor["hot"] & outdoor["cold"] & occ_low, delta["strong_down"]),
        ctrl.Rule(indoor["hot"] & outdoor["cold"] & occ_high, delta["slight_down"]),
        ctrl.Rule(indoor["very_cold"] & outdoor["very_cold"] & (occ_low | occ_med), delta["stable"]),
        ctrl.Rule(indoor["hot"] & outdoor["hot"] & (occ_low | occ_med), delta["stable"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, (d_lo, d_hi)


# --- Build everything once on import ---

DF = load_and_prepare()
OCC_SYSTEM = build_occupancy_system(DF)
DF_WITH_OCC = compute_occ_fuzzy(DF, OCC_SYSTEM)
DELTA_SYSTEM, DELTA_RANGE = build_delta_system(DF_WITH_OCC)


def single_step(indoor_t, outdoor_t, co2, lighting):
    """
    One-shot fuzzy inference:
    Inputs:
        indoor_t  (°C)
        outdoor_t (°C)
        co2       (ppm)
        lighting  (arbitrary units)
    Returns:
        occ       (0–1)
        delta_t   (°C change next 15 min)
        t_future  (°C predicted)
        action    ("HEAT_ON" | "COOL_ON" | "IDLE" | "OFF")
    """
    # 1) Occupancy
    occ_sim = ctrl.ControlSystemSimulation(OCC_SYSTEM)
    occ_sim.input["co2"] = float(co2)
    occ_sim.input["light"] = float(lighting)
    occ_sim.compute()
    occ = float(occ_sim.output["occ"])

    # 2) ΔT
    d_lo, d_hi = DELTA_RANGE
    sim = ctrl.ControlSystemSimulation(DELTA_SYSTEM)
    sim.input["indoor"] = float(indoor_t)
    sim.input["outdoor"] = float(outdoor_t)
    sim.input["occupancy"] = float(occ)
    sim.compute()

    # sometimes no rule fires → 'delta' missing, so fall back safely
    if "delta" in sim.output:
        raw_delta = sim.output["delta"]
    else:
        # log something to HF logs for debugging
        print(
            f"[WARN] No 'delta' output for inputs: "
            f"indoor={indoor_t}, outdoor={outdoor_t}, occ={occ}"
        )
        raw_delta = 0.0  # assume no change as a safe default

    delta_t = float(np.clip(raw_delta, d_lo, d_hi))
    t_future = indoor_t + delta_t

    # 3) HVAC action
    if occ > OCC_TH:
        if t_future < COMFORT_LOW:
            action = "HEAT_ON"
        elif t_future > COMFORT_HIGH:
            action = "COOL_ON"
        else:
            action = "IDLE"
    else:
        action = "OFF"

    return occ, delta_t, t_future, action

