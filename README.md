#  Smart HVAC Fuzzy Assistant

An intelligent building assistant designed to make smart, explainable heating, ventilation, and air conditioning (HVAC) decisions.

# How to run the project locally

## 1) Clone the Project
```bash
git clone https://github.com/QuasimodoCodes/Fuzzy_Temp.git
cd Fuzzy_Temp
```
## 2) Create and  Activate Virtual Environment

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

## 3) Install Requirements
pip install -r requirements.txt
pip install jupyter

## 4) Run the Jupyter Notebook and open
Fuzzy_system.ipynb

This notebook includes data loading, preprocessing, fuzzy membership functions, and the full 15-minute temperature forecasting pipeline.

##  Live Demo

Try the interactive chatbot below:

**[Open the Fuzzy Room Temperature Assistant](https://huggingface.co/spaces/chonthichar/fuzzy-room-temperature-assistant)**

* **Predict Temperature** ($\Delta T$ in 15 mins)
* **Detect Occupancy** (Fuzzy level [0-1])
* **Decide HEAT/COOL/OFF** (Optimal action)
* **Explain using AI** (Transparent decision-making)

---

##  Project Implementation and Core Logic

This assistant uses a combination of technologies: **Fuzzy Logic** for occupancy and prediction, a **Mamdani FIS** for action selection, an **OpenAI LLM** for explanations, and **Gradio** for the UI.

The assistant answers the question: *"Should the HVAC system heat, cool, idle, or turn off in the next 15 minutes?"*

**Example Input:** The user provides the sensor data: `21, 12, 400, 60` (Indoor Temp, Outdoor Temp, CO₂, Lighting).

### 1. Fuzzy Occupancy Detection

Uses CO₂ and Lighting levels to infer occupancy in the range **[0-1]**.

* **Rule Example:** Low CO₂ and Dark room $\rightarrow$ Room is empty. High CO₂ and Bright room $\rightarrow$ Room is occupied.
* **Mechanism:** Uses **6 fuzzy rules** with membership functions for **low/medium** and **high** inputs.
### 2. Fuzzy Temperature Prediction (15 Minutes Ahead)

Predicts the temperature change ($\Delta T$) using a Mamdani FIS trained on the SML2010 dataset.

* **Inputs:** Indoor Temperature, Outdoor Temperature, and Fuzzy Occupancy (from Step 1).

### 3. HVAC Decision Layer

Determines the final HVAC action based on the predicted future state:

* **If room is occupied:** Too cold $\rightarrow$ **HEAT\_ON** | Too warm $\rightarrow$ **COOL\_ON** | Comfortable $\rightarrow$ **IDLE**
* **If room is empty:** Turn **OFF** HVAC to save energy.

### 4. AI Generated Explanations (using OpenAI)

The LLM generates comprehensive, natural explanations that include: Why the temperature is predicted to change, why **HEAT/COOL/OFF** was chosen, why the specific **occupancy** was inferred, and what the **best energy action** should be.

### 5. Gradio Chat Interface

The user can interact with the system by:

* **Sending Sensor Data:**
    ```
    - 21 12 400 60
    ```
* **Asking Natural Questions:**
    ```
    - Who are you?
    - What should HVAC do?
    - Explain the decision
    ```

---

##  Project Structure

The project includes the full pipeline and visualization components. **For examination and easy viewing of the logic and visualization, the primary component is the Jupyter Notebook:**

* **`Fuzzy_system.ipynb`**: Contains the **whole pipeline**, including logic, training, **visualization of fuzzy sets and rules**, and testing. **Examiners can run everything here without the full app setup.**
  * **Assistant Files:** Used for the deployed web application:

      ```
      fuzzy-room-temperature-assistant/
                
              ├── app.py 
              ├── fuzzy_backend.py 
              ├── Fuzzy_system.ipynb  
              ├── NEW-DATA-1.T15.txt 
              ├── requirements.txt
              └── README.md
      ```


---


##  Running Locally

The project can be run in two ways:

### Option 1: Easy Examination (Jupyter Notebook)

The easiest way for the examiner to review and run the entire system is through:

1.  **Open `Fuzzy_system.ipynb`** in a Jupyter or Colab environment.
2.  **Run the cells sequentially.** This allows direct inspection of the fuzzy logic implementation, data handling, and $\Delta T$ prediction visualization.

This notebook includes:

- Complete fuzzy logic implementation
- Occupancy detection
- 15-minute temperature forecasting
- All membership functions and fuzzy rules
- Visualizations
- HVAC decision logic
- Example inputs and outputs

**The examiner can run everything from this notebook alone — no need to run the full chatbot application.**

The chatbot (`app.py`) is an optional extension that integrates the fuzzy system into a conversational assistant.



### Option 2: Full Web Assistant (Gradio)

1.  **Git Clone:** Clone the project repository.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add Your OpenAI Key:** Generate your key and set the environment variable:
    ```bash
    export OPENAI_API_KEY=your_key_here
    ```
4.  **Run Application:**
    ```bash
    python app.py
    ```
---

##  Deployment Process

This project is deployed on **Hugging Face Spaces** using **Gradio**. You can deploy by pushing the code to a Space, or by running the Jupyter Notebook `Fuzzy_system.ipynb` inside the Space.

##  Dataset

The model uses a slice of the **SML 2010 Smart Home Dataset**, specifically `NEW-DATA-1.T15.txt`.

* **Data Included:** Indoor temperature, Outdoor temperature, CO₂ sensor, and Lighting.

* **Dataset Source:** [https://archive.ics.uci.edu/ml/datasets/SML2010](https://archive.ics.uci.edu/ml/datasets/SML2010)

* **Citation:** Candanedo, Luis M., and Véronique Feldheim. “Accurate occupancy detection of an office building using near‐cost sensors.” *Energy and Buildings* 86 (2015): 273–282.
