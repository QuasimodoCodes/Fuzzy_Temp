import re
import os
import gradio as gr
from openai import OpenAI
from fuzzy_backend import single_step

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

HELP = (
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
    try:
        indoor, outdoor, co2, light = parse_message(message)
    except Exception as e:
        return f"{e}\n\n{HELP}"

    import traceback
    try:
        occ, dT, t_future, action = single_step(indoor, outdoor, co2, light)
    except Exception as e:
        tb = traceback.format_exc()
        # show the real error inside the chat, no red "Error" bubble
        return f"Internal error in fuzzy_backend:\n\n{e}\n\n```python\n{tb}\n```"

    action_text = {
        "HEAT_ON": "Heat now (room will be too cold soon).",
        "COOL_ON": "Cool now (room will be too warm soon).",
        "IDLE": "Comfortable range – no action needed.",
        "OFF": "Room probably empty – turn HVAC OFF to save energy.",
    }.get(action, "")

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
        f"{action_text}"
    )

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Smart HVAC Fuzzy Assistant",
    description="Type: `21 12 400 60` (indoor, outdoor, CO₂, lighting)…",
)

if __name__ == "__main__":
    demo.launch()