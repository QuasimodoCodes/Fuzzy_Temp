import os
import re
import traceback

import gradio as gr
from openai import OpenAI

from fuzzy_backend import single_step

# =========================================================
#  OpenAI client (optional, only if key is available)
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENABLED = OPENAI_API_KEY is not None

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_ENABLED else None

# =========================================================
#  Helper text
# =========================================================

HELP = (
    "I need **4 numbers** in this order:\n"
    "  indoor_temp  outdoor_temp  CO₂_ppm  lighting\n\n"
    "Examples:\n"
    "  `21 12 400 60`\n"
    "  `indoor=21, outdoor=12, co2=400, light=60`"
)

# Short system prompt summarising your whole notebook
SYSTEM_PROMPT = """
You are the **Smart HVAC Fuzzy Assistant**.

Your core is a 2-stage fuzzy logic model trained on the SML2010 building dataset
(NEW-DATA-1.T15, March–April 2012, 15-minute sampling).

High-level design:
- Stage 1: fuzzy occupancy estimator
  - Inputs: CO2_Comedor_Sensor (ppm), Lighting_Comedor_Sensor
  - Fuzzy sets:
    - CO2: low / medium / high
    - Lighting: dark / dim / bright
    - Occupancy output (0–1): low / medium / high
  - Rules (examples):
    - dark & low CO2  → low occupancy
    - bright & medium CO2 → high occupancy
    - dim & high CO2 → high occupancy
  - Output: occ_fuzzy in [0,1] and a binary flag occupied_flag_fuzzy (threshold ~0.4–0.5).

- Stage 2: 3-input Mamdani FIS for 15-min temperature forecast
  - Inputs:
    - Indoor temperature: Temperature_Comedor_Sensor
    - Outdoor temperature: Temperature_Exterior_Sensor
    - Fuzzy occupancy: occ_fuzzy
  - Output: Delta15 = T(t+15) - T(t), with fuzzy sets:
    - strong_down, slight_down, stable, slight_up, strong_up
  - Indoor/outdoor each use 5 fuzzy sets: very_cold, cold, normal, warm, hot.
  - Occupancy input uses 3 fuzzy sets: low, medium, high.
  - Around 18 interpretable rules encode heating / cooling / stable behaviour
    depending on indoor/outdoor relationship and occupancy.
  - Universes and membership ranges are derived **only from TRAIN data** to avoid leakage.

Datasets and splitting:
- Source: real building measurements (cafeteria/comedor), 30 days, 15-min steps (~2760 rows).
- Features used:
  - Indoor & outdoor temperatures
  - CO2 & lighting (for occupancy)
  - Derived: T_future_15, Delta15, occ_fuzzy, occupied_flag_fuzzy
- Train/test: 80% early data for training, 20% last days for testing (shuffle=False).

Baselines for comparison:
- Persistence: T(t+15) = T(t)
- Linear regression using [indoor, outdoor, occ_fuzzy]
- Fuzzy system (3-input Mamdani):
  - Typically: RMSE ≈ 0.11 °C, MAE ≈ 0.10 °C
  - Better than persistence; slightly worse than linear regression but **interpretable and occupancy-aware**.

HVAC decision layer:
- Comfort band: 21°C ≤ T ≤ 24°C
- Occupancy threshold: occ_fuzzy > 0.5 → occupied
- Actions:
  - If occupied & T_pred < 21 → HEAT_ON
  - If occupied & T_pred > 24 → COOL_ON
  - If occupied & 21 ≤ T_pred ≤ 24 → IDLE
  - If not occupied → OFF (energy saving)

Your job when answering questions is:
- Explain how this fuzzy system works, how it was trained, what rules and features it uses, and how it compares to baselines.
- Use clear, non-technical language if the user sounds non-expert.
- If asked “who are you” or “what can you do”, say that you are a smart HVAC assistant that:
  - predicts indoor temperature 15 minutes ahead,
  - estimates occupancy from CO2 and lighting,
  - and suggests HEAT / COOL / IDLE / OFF decisions to save energy and keep comfort.
- Do **not** claim to use deep learning or neural networks; the core is fuzzy logic + simple ML baselines.
- If the user wants a prediction, tell them to send 4 numbers in the format:
  indoor_temp outdoor_temp CO2_ppm lighting.
- If the question is completely unrelated to HVAC, fuzzy logic, or the project, politely say you are focused on this HVAC fuzzy system only.
"""


# =========================================================
#  Helpers
# =========================================================

def parse_message(msg: str):
    """
    Extract first 4 numeric values from the user message.
    Returns (indoor, outdoor, co2, light) as floats.
    Raises ValueError if not enough numbers.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+", msg)
    if len(nums) < 4:
        raise ValueError("I need 4 numbers: indoor, outdoor, CO₂, lighting.")
    indoor, outdoor, co2, light = map(float, nums[:4])
    return indoor, outdoor, co2, light


def use_llm(message: str) -> str:
    """
    Use the OpenAI Responses API to answer conceptual questions
    about the fuzzy system, dataset, strategy, etc.
    """
    if not OPENAI_ENABLED:
        return (
                "I can only run the fuzzy HVAC model right now – the external LLM "
                "API key is not configured on this deployment.\n\n"
                + HELP
        )

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )

        # Extract text from the new Responses API structure
        # (first output, first content block)
        first_out = resp.output[0]
        first_content = first_out.content[0]
        return first_content.text

    except Exception as e:
        tb = traceback.format_exc()
        return (
                "There was an error when calling the LLM:\n\n"
                f"{e}\n\n```python\n{tb}\n```\n\n"
                "I can still help with numeric fuzzy predictions. " + HELP
        )


# =========================================================
#  Chat function
# =========================================================

def chat_fn(message, history):
    """
    Dual-mode behaviour:

    - If the message contains 4 or more numbers:
        → run fuzzy_backend.single_step and explain the result.
    - Otherwise:
        → use OpenAI LLM to answer questions about the fuzzy system.
    """
    # Try numeric path first
    try:
        indoor, outdoor, co2, light = parse_message(message)
        numeric_mode = True
    except Exception:
        numeric_mode = False

    # ---------- 1) Numeric → run fuzzy model ----------
    if numeric_mode:
        try:
            occ, dT, t_future, action = single_step(indoor, outdoor, co2, light)
        except Exception as e:
            tb = traceback.format_exc()
            return (
                "Internal error in fuzzy_backend:\n\n"
                f"{e}\n\n```python\n{tb}\n```"
            )

        # Short label for occupancy
        if occ >= 0.7:
            occ_label = "I’m quite sure the room is occupied."
        elif occ >= 0.4:
            occ_label = "The room might be occupied, but it’s not very clear."
        else:
            occ_label = "It looks like the room is mostly empty."

        # Short text for the HVAC action
        action_text = {
            "HEAT_ON": "Heat now (room will be too cold soon).",
            "COOL_ON": "Cool now (room will be too warm soon).",
            "IDLE": "Comfortable range – no action needed.",
            "OFF": "Room probably empty – turn HVAC OFF to save energy.",
        }.get(action, "")

        # Turn-ON/OFF recommendation in simple terms
        if action == "HEAT_ON":
            recommendation = "Turn **ON heating**."
        elif action == "COOL_ON":
            recommendation = "Turn **ON cooling**."
        elif action == "IDLE":
            recommendation = "Keep the HVAC **ON but idle** (no heating or cooling needed)."
        else:
            recommendation = "Turn the HVAC **OFF** to save energy."

        # Story-style explanation so it feels “AI-like”
        story = (
            " **Smart Climate Assistant Insight**\n"
            "I analysed your room using indoor/outdoor temperature, CO₂ and lighting. "
            "From these I inferred how likely the room is occupied and how the temperature "
            "will move in the next 15 minutes. Based on that, I chose the HVAC action "
            "that balances comfort and energy use."
        )

        details = (
            f"- Fuzzy occupancy `{occ:.2f}` → {occ_label}\n"
            f"- Predicted change ΔT ≈ `{dT:+.3f} °C` over 15 minutes.\n"
            f"- Forecast indoor temperature ≈ `{t_future:.2f} °C`."
        )

        return (
            f"**Inputs**\n"
            f"- Indoor T: `{indoor:.2f} °C`\n"
            f"- Outdoor T: `{outdoor:.2f} °C`\n"
            f"- CO₂: `{co2:.1f} ppm`\n"
            f"- Lighting: `{light:.1f}`\n\n"
            f"**Fuzzy occupancy**: `{occ:.2f}` (0–1)\n"
            f"ΔT next 15 min: `{dT:+.3f} °C`\n"
            f"Predicted T in 15 min: `{t_future:.2f} °C`\n\n"
            f"**HVAC suggestion**: **{action}**\n"
            f"{action_text}\n\n"
            f"---\n"
            f"{story}\n\n"
            f"{details}\n\n"
            f"---\n"
            f"###  Final Recommendation\n"
            f"**{recommendation}**"
        )

    # ---------- 2) Non-numeric → let LLM explain the project ----------
    return use_llm(message)


# =========================================================
#  Gradio interface
# =========================================================

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Smart HVAC Fuzzy Assistant",
    description=(
        "Ask questions like **'who are you?'** or **'how were you trained?'**, "
        "or send `21 12 400 60` (indoor, outdoor, CO₂, lighting) and I will:\n\n"
        "- infer occupancy from CO₂ + lighting,\n"
        "- predict temperature in 15 minutes,\n"
        "- and suggest HEAT / COOL / IDLE / OFF with an explanation.\n\n"
        "For numeric input I run a fuzzy logic model; for other text I use an LLM "
        "to explain how the system works."
    ),
)

if __name__ == "__main__":
    demo.launch()