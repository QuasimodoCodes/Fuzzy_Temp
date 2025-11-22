import re
import os
import gradio as gr
from openai import OpenAI
from fuzzy_backend import single_step

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    occ, dT, t_future, action = single_step(indoor, outdoor, co2, light)

    action_text = {
        "HEAT_ON": "Heat now (room will be too cold soon).",
        "COOL_ON": "Cool now (room will be too warm soon).",
        "IDLE": "Comfortable range – no action needed.",
        "OFF": "Room probably empty – turn HVAC OFF to save energy.",
    }.get(action, "")

    base_reply = (
        f"**Inputs**\n"
        f"- Indoor T: `{indoor:.2f} °C`\n"
        f"- Outdoor T: `{outdoor:.2f} °C`\n"
        f"- CO₂: `{co2:.1f} ppm`\n"
        f"- Lighting: `{light:.1f}`\n\n"
        f"**Fuzzy occupancy**: `{occ:.2f}` (0–1)\n"
        f"ΔT next 15 min: `{dT:+.3f} °C`\n"
        f"Predicted T in 15 min: `{t_future:.2f} °C`\n\n"
        f"**HVAC suggestion**: **{action}**\n"
        f"{action_text}\n"
    )

    # optional: ask GPT to rewrite this in more natural language
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly building assistant explaining HVAC decisions simply."
                },
                {
                    "role": "user",
                    "content": (
                        f"Indoor={indoor:.2f}, Outdoor={outdoor:.2f}, CO2={co2:.1f}, "
                        f"Lighting={light:.1f}, Occupancy={occ:.2f}, "
                        f"DeltaT={dT:+.3f}, FutureT={t_future:.2f}, Action={action}."
                    ),
                },
            ],
        )
        explanation = completion.choices[0].message.content
        return base_reply + "\n---\n**GPT explanation:**\n" + explanation
    except Exception:
        # If OpenAI fails for any reason, still return fuzzy result
        return base_reply


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Smart HVAC Fuzzy Assistant",
    description=(
        "Type: `21 12 400 60` (indoor, outdoor, CO₂, lighting) and I will "
        "predict T in 15 minutes and suggest HEAT / COOL / IDLE / OFF, "
        "plus a natural-language explanation."
    ),
)

if __name__ == "__main__":
    demo.launch()