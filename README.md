# Smart HVAC Fuzzy Assistant

- Predict temperature 
- Detect Occupancy
- Decide HEAT/COOL/OFF 
- Explain using AI

This project implement an intelligence building assistant using:
- Fuzzy logic for occupancy detection.
- A Mamdani FIS that choose HEAT/COOL/IDLE/OFF
- An LLM (OpenAI) to generate human friendly explanations.
- A Gradio chatbot UI and deploy on Hugging Face Spaces.

This assistant can answer the question: 
"Should the HACV" heat, cool, idle or turn off in the next 15 minutes?"
In order to get the value, the user need to send the value of (indoor, outdoor, CO2 and lighting) For example: 21, 12, 400, 60. The model wil then:

1) Fuzzy Occupancy detection
Uses CO2 and Lightning to infer occupancy in [0-1].
   - Low CO2 and Dark room = Room is empty.
   - High CO2 and Bright room = Room is occupied.
   - Use 6 fuzzy rules with membership functions fot low/medium and high.

2) Fuzzy predict (15 minutes ahead)
Predict temperature change using the inputs:
   - Indoor temperature.
   - Outdoor temperature.
     - Fuzzy occupancy. 
Use a Mamdani fuzzy inference system trained from the SML2010 dataset.

3) HVAC decision layer
   - If room is occupied:
     - Too cold = HEAT_ON
     - Too warm = COOL_ON
       - Comfortable = IDLE
   - If empty = Turn off HVAC to save eneergy.

4) AI generated explanations (using OpenAI)
The assistant not only returns numbers and nicely explanations:
   - Why the temperature will change.
   - Why HEAT/COOL/OFF was chosen.
   - Why occupancy was inferred.
   - What is the best energy action should be.

5) Gradio chat interface:
User can ask : 

        - 21 12 400 60

Or natural questions like

        - Who are you?
        - What should HVAC do?
        - Explain the decision

### Project structure




### How to run the code?
We have two different:
1) The whole pipeline include visualization inside Jupyter Notebook called : Fuzzy_system.ipynb
2) fuzzy-room-temperature-assistant/

          │
          ├── app.py                 # Gradio + OpenAI chatbot
          ├── fuzzy_backend.py       # Full fuzzy logic engine
          ├── NEW-DATA-1.T15.txt     # SML2010 dataset slice
          ├── requirements.txt
          └── README.md

How to run AI assistant: 
1) Users sends 4 numbers: indoor, outdoor, CO2 and lighting

        - 21 12 400 60

2) Backend compute
- Fuzzy occupacy
- ΔT (predicted change)
- T_Future (temperature in 15 minutes)
- HVAC action (HEAT/ COOL/ OFF and IDLE)

3) LLM converts outputs into natural explanations
Example 


## Running locally:
1) Git Clone
2) Install deps

        -  pip install -r requirements.txt

3) Add your OpenAI key go to website to genrate your API key

        - export OPENAI_API_KEY=your_key_here

4) Run

        - python app.py

## Deployment process:
This project is deployed on Hugging Face Spaces using Gradio. You just push your code to a space and it will auto generate or simple run Notbook insed to see a whole pipeline. 

## Dataset
This model use the SML 2010 Smart Home Dataset, specially NEW-DATA-1.T15.txt

Which includes :
- Indoor temperature
- Outdoor temperature
- CO₂ sensor
- Lighting

Dataset Source:
https://archive.ics.uci.edu/ml/datasets/SML2010

Candanedo, Luis M., and Véronique Feldheim. “Accurate occupancy detection of an office building using near‐cost sensors.” Energy and Buildings 86 (2015): 273–282.