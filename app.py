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
    # ------------------------------------------------------
    # CASE 1: message does NOT have 4 numbers → pure LLM chat
    # ------------------------------------------------------
    nums = re.findall(r"[-+]?\d*\.?\d+", message)
    if len(nums) < 4:
        if client is None:
            return (
                    "Hi, I’m your **Smart HVAC Assistant** 🤖\n\n"
                    "To make a prediction, I need 4 numbers: indoor temperature, "
                    "outdoor temperature, CO₂ and lighting.\n\n"
                    + HELP
            )

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly smart building assistant. "
                            "Explain things in simple language, like to someone with no technical background."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
                max_tokens=200,
                temperature=0.6,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"(AI chat unavailable: {e})\n\n" + HELP

    # ------------------------------------------------------
    # CASE 2: message has numbers → fuzzy + LLM mode
    # ------------------------------------------------------
    try:
        indoor, outdoor, co2, light = parse_message(message)
    except Exception as e:
        return f"{e}\n\n{HELP}"

    try:
        occ, dT, t_future, action = single_step(indoor, outdoor, co2, light)
    except Exception as e:
        tb = traceback.format_exc()
        return (
            "Internal error in fuzzy_backend:\n\n"
            f"{e}\n\n```python\n{tb}\n```"
        )

    # --- short text for the action ---
    if action == "HEAT_ON":
        action_label = "turn on heating"
        action_text = "The room will get too cold soon, so it’s best to start heating."
    elif action == "COOL_ON":
        action_label = "turn on cooling"
        action_text = "The room will get too warm soon, so it’s best to start cooling."
    elif action == "IDLE":
        action_label = "keep the system on, but not heating or cooling"
        action_text = "The temperature is already comfortable, so no change is needed."
    else:  # OFF
        action_label = "turn the HVAC off"
        action_text = "The room looks empty, so you can turn the system off to save energy."

    # --- simple occupancy wording ---
    if occ >= 0.7:
        occ_label = "The room probably has people inside."
    elif occ >= 0.4:
        occ_label = "The room might have some people, but I’m not fully sure."
    else:
        occ_label = "The room seems mostly empty."

    # --- quick human-friendly summary ---
    summary = (
        "### 🧾 Simple summary\n"
        f"- **Now:** about **{indoor:.1f} °C** inside\n"
        f"- **In 15 minutes:** I expect about **{t_future:.1f} °C**\n"
        f"- **Room use:** {occ_label}\n\n"
        f"👉 So, my suggestion is to **{action_label}**.\n"
        f"{action_text}\n"
    )

    # --- LLM explanation in very simple words ---
    if client is None:
        explanation = "(AI explanation unavailable: missing `OPENAI_API_KEY` secret.)"
    else:
        prompt = (
            "You are an intelligent HVAC assistant for a smart home.\n"
            "Explain this decision in 2–3 very simple sentences. "
            "Imagine you are talking to someone with no technical background. "
            "Avoid jargon like 'fuzzy logic', 'ΔT', 'occupancy index'.\n\n"
            f"Indoor temperature: {indoor:.2f} °C\n"
            f"Outdoor temperature: {outdoor:.2f} °C\n"
            f"CO₂: {co2:.1f} ppm\n"
            f"Lighting: {light:.1f}\n"
            f"Occupancy score (0–1): {occ:.2f}\n"
            f"Predicted temperature in 15 minutes: {t_future:.2f} °C\n"
            f"Chosen action: {action} ({action_label})\n"
        )
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly, non-technical assistant. "
                            "Use short, clear sentences and everyday words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.4,
            )
            explanation = completion.choices[0].message.content
        except Exception as e:
            explanation = f"(AI explanation unavailable: {e})"

    # --- technical details (for engineers) ---
    details = (
        "### 🔧 Technical details (optional)\n"
        f"- Indoor T now: `{indoor:.2f} °C`\n"
        f"- Outdoor T: `{outdoor:.2f} °C`\n"
        f"- CO₂: `{co2:.1f} ppm`\n"
        f"- Lighting: `{light:.1f}`\n"
        f"- Occupancy score: `{occ:.2f}` (0–1)\n"
        f"- Estimated change in 15 minutes: `{dT:+.3f} °C`\n"
        f"- Predicted T in 15 minutes: `{t_future:.2f} °C`\n"
        f"- Fuzzy HVAC action: **{action}**\n"
    )

    return (
            summary
            + "\n---\n"
            + "### 🤖 AI explanation\n"
            + explanation
            + "\n\n---\n"
            + details
    )



demo = gr.ChatInterface(
    fn=chat_fn,
    title="Smart HVAC Fuzzy Assistant",
    description=(
        "Ask questions like 'who are you?' or send `21 12 400 60` "
        "(indoor, outdoor, CO₂, lighting). "
        "For numeric input I will run fuzzy logic + LLM; for other text I will reply as a smart HVAC assistant."
    ),
)

if __name__ == "__main__":
    demo.launch()