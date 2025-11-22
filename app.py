import os
import re
import traceback
import gradio as gr
from openai import OpenAI
from fuzzy_backend import single_step

# Read API key from environment (HF "Secrets")
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

HELP = (
    "I need 4 numbers: indoor, outdoor, CO₂, lighting.\n\n"
    "Please provide **4 numbers** in this order:\n"
    "  indoor_temp  outdoor_temp  CO2_ppm  lighting\n\n"
    "Examples:\n"
    "  21 12 400 60\n"
    "  indoor=21, outdoor=12, co2=400, light=60"
)


def parse_message(msg: str):
    """
    Extract first 4 numeric values from the user message.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+", msg)
    if len(nums) < 4:
        raise ValueError("I need 4 numbers: indoor, outdoor, CO2, lighting.")
    indoor, outdoor, co2, light = map(float, nums[:4])
    return indoor, outdoor, co2, light


def chat_fn(message, history):
    # 1) Parse user input
    try:
        indoor, outdoor, co2, light = parse_message(message)
    except Exception as e:
        return f"{e}\n\n{HELP}"

    # 2) Call fuzzy backend
    try:
        occ, dT, t_future, action = single_step(indoor, outdoor, co2, light)
    except Exception as e:
        tb = traceback.format_exc()
        return (
            "Internal error in fuzzy_backend:\n\n"
            f"{e}\n\n```python\n{tb}\n```"
        )

    # --- short text for the action ---
    action_text = {
        "HEAT_ON": "Heat now (room will be too cold soon).",
        "COOL_ON": "Cool now (room will be too warm soon).",
        "IDLE": "Comfortable range – no action needed.",
        "OFF": "Room probably empty – turn HVAC OFF to save energy.",
    }.get(action, "")

    # --- interpretation of occupancy ---
    if occ >= 0.7:
        occ_label = "I’m quite sure the room is occupied."
    elif occ >= 0.4:
        occ_label = "The room might be occupied, but it’s not very clear."
    else:
        occ_label = "It looks like the room is mostly empty."

    # --- smart assistant storytelling ---
    story = (
        "🏠 **Smart Climate Assistant Insight**\n"
        "I analyzed your room using indoor/outdoor temperature, CO₂ levels, lighting, "
        "and fuzzy occupancy detection. Based on the predicted temperature 15 minutes "
        "from now and whether the room seems occupied, I calculated the best HVAC action "
        "to balance comfort and energy savings.\n\n"
        f"- Fuzzy occupancy: **{occ:.2f}** → {occ_label}\n"
        f"- Predicted temperature in 15 minutes: **{t_future:.2f} °C** "
        f"(change of {dT:+.3f} °C from now)."
    )

    # --- turn ON / OFF recommendation ---
    if action == "HEAT_ON":
        recommendation = "Turn **ON heating**."
    elif action == "COOL_ON":
        recommendation = "Turn **ON cooling**."
    elif action == "IDLE":
        recommendation = "Keep the HVAC **ON but idle** (no heating or cooling needed)."
    else:  # OFF
        recommendation = "Turn the HVAC **OFF** to save energy."

    # --- LLM-generated explanation (optional, safe if unavailable) ---
    if client is None:
        explanation = "(LLM explanation unavailable: missing `OPENAI_API_KEY` secret.)"
    else:
        prompt = (
            "You are an intelligent HVAC assistant for a smart building.\n"
            "Explain in 2–3 simple sentences why this is the right decision.\n\n"
            f"Indoor temperature: {indoor:.2f} °C\n"
            f"Outdoor temperature: {outdoor:.2f} °C\n"
            f"CO₂: {co2:.1f} ppm\n"
            f"Lighting: {light:.1f}\n"
            f"Fuzzy occupancy (0–1): {occ:.2f}\n"
            f"Predicted 15-min temperature: {t_future:.2f} °C\n"
            f"Temperature change (ΔT): {dT:+.3f} °C\n"
            f"Chosen HVAC action: {action}\n"
        )
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a smart building climate assistant. Be clear, concise, and non-technical.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.4,
            )
            explanation = completion.choices[0].message.content
        except Exception as e:
            explanation = f"(LLM explanation unavailable: {e})"

    # --- numeric summary + story + final recommendation + LLM explanation ---
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
        f"---\n"
        f"### ✅ Final Recommendation\n"
        f"**{recommendation}**\n\n"
        f"---\n"
        f"### 🤖 AI Explanation\n"
        f"{explanation}"
    )


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Smart HVAC Fuzzy Assistant",
    description=(
        "Type: `21 12 400 60` (indoor, outdoor, CO₂, lighting) and I will "
        "predict T in 15 minutes, infer occupancy, choose HEAT / COOL / IDLE / OFF, "
        "and explain the decision using an AI assistant."
    ),
)

if __name__ == "__main__":
    demo.launch()