import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartFlow AI",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("traffic_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    return model, encoder


model, le = load_model()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    # First try root folder
    try:
        df = pd.read_csv("Traffic.csv")
    except:
        # If not found, try data folder
        df = pd.read_csv("data/Traffic.csv")

    return df


traffic_df = load_data()


# =========================================================
# DATA PREPROCESSING
# =========================================================

if "Time" in traffic_df.columns:

    traffic_df["Time"] = pd.to_datetime(
        traffic_df["Time"],
        errors="coerce"
    )

    traffic_df["Hour"] = traffic_df["Time"].dt.hour


# =========================================================
# GEMINI AI FUNCTION
# =========================================================

def get_gemini_recommendation(
    traffic,
    hour,
    car_count,
    bike_count,
    bus_count,
    truck_count
):

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to Streamlit Secrets."
        )


    prompt = f"""
You are an intelligent smart-city traffic management assistant.

Analyze the following traffic prediction:

Predicted Traffic: {traffic}
Hour: {hour}:00

Vehicle Counts:
Cars: {car_count}
Bikes: {bike_count}
Buses: {bus_count}
Trucks: {truck_count}

Provide a practical traffic-management recommendation.

Include:
1. Current traffic assessment
2. Main concern
3. Recommended traffic-management actions
4. Public transportation suggestion
5. Route or signal-management suggestion

Keep the answer professional, practical and concise.
Do not use unnecessary technical language.
"""


    # Current Gemini model
    model_name = "gemini-3.5-flash"


    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:generateContent"
    )


    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }


    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }


    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )


        if response.status_code == 200:

            result = response.json()

            recommendation = (
                result["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

            return recommendation


        else:

            return (
                f"Gemini could not generate the recommendation.\n\n"
                f"API Status: {response.status_code}\n\n"
                f"{response.text}"
            )


    except Exception as e:

        return (
            "Gemini AI is temporarily unavailable.\n\n"
            f"Error: {str(e)}"
        )


# =========================================================
# HEADER
# =========================================================

st.title("🚦 SmartFlow AI")

st.write(
    "Smart traffic prediction dashboard using "
    "Machine Learning and Generative AI."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🚦 SmartFlow AI")

    st.write(
        "An intelligent traffic-management dashboard "
        "combining Machine Learning and Generative AI."
    )

    st.divider()

    st.success("🤖 ML Model: Random Forest")

    st.success("🧠 AI Engine: Gemini")

    st.info(
        "📊 Historical traffic analysis enabled"
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Traffic Dataset Overview")


total_records = len(traffic_df)


if "CarCount" in traffic_df.columns:
    average_cars = traffic_df["CarCount"].mean()
else:
    average_cars = 0


if "BikeCount" in traffic_df.columns:
    average_bikes = traffic_df["BikeCount"].mean()
else:
    average_bikes = 0


if "BusCount" in traffic_df.columns:
    average_buses = traffic_df["BusCount"].mean()
else:
    average_buses = 0


if "TruckCount" in traffic_df.columns:
    average_trucks = traffic_df["TruckCount"].mean()
else:
    average_trucks = 0


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        "📄 Total Records",
        total_records
    )


with kpi2:

    st.metric(
        "🚗 Avg Cars",
        f"{average_cars:.1f}"
    )


with kpi3:

    st.metric(
        "🏍️ Avg Bikes",
        f"{average_bikes:.1f}"
    )


with kpi4:

    st.metric(
        "🚌 Avg Buses",
        f"{average_buses:.1f}"
    )


# =========================================================
# HISTORICAL TRAFFIC ANALYSIS
# =========================================================

st.header("📈 Historical Traffic Analysis")


if "Traffic Situation" in traffic_df.columns:

    traffic_counts = (
        traffic_df["Traffic Situation"]
        .value_counts()
    )

    st.bar_chart(
        traffic_counts
    )


# =========================================================
# TRAFFIC PREDICTION INPUT
# =========================================================

st.header("🚗 Traffic Prediction")


col1, col2 = st.columns(2)


with col1:

    car_count = st.number_input(
        "🚗 Car Count",
        min_value=0,
        value=100,
        step=1
    )


    bike_count = st.number_input(
        "🏍️ Bike Count",
        min_value=0,
        value=20,
        step=1
    )


with col2:

    bus_count = st.number_input(
        "🚌 Bus Count",
        min_value=0,
        value=10,
        step=1
    )


    truck_count = st.number_input(
        "🚚 Truck Count",
        min_value=0,
        value=5,
        step=1
    )


# =========================================================
# TIME INPUT
# =========================================================

hour = st.slider(
    "🕐 Select Hour",
    min_value=0,
    max_value=23,
    value=18
)


# Determine traffic period

if 7 <= hour <= 10:

    traffic_period = "Morning Peak Hour"

elif 16 <= hour <= 19:

    traffic_period = "Evening Peak Hour"

elif 11 <= hour <= 15:

    traffic_period = "Normal Hour"

else:

    traffic_period = "Off-Peak Hour"


st.info(
    f"🕐 Selected Time: **{hour}:00**  |  "
    f"Traffic Period: **{traffic_period}**"
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🚦 Predict Traffic",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input DataFrame

    new_data = pd.DataFrame({

        "CarCount": [car_count],

        "BikeCount": [bike_count],

        "BusCount": [bus_count],

        "TruckCount": [truck_count],

        "Hour": [hour]

    })


    # Machine Learning prediction

    prediction = model.predict(
        new_data
    )


    # Convert encoded value to original label

    traffic = le.inverse_transform(
        prediction
    )[0]


    # =====================================================
    # PREDICTION PROBABILITY
    # =====================================================

    probability = None

    if hasattr(model, "predict_proba"):

        probability = (
            model.predict_proba(new_data)[0]
        )

        confidence = probability.max() * 100

    else:

        confidence = 0


    # =====================================================
    # SAVE RESULTS IN SESSION
    # =====================================================

    st.session_state["traffic"] = traffic
    st.session_state["hour"] = hour
    st.session_state["traffic_period"] = traffic_period
    st.session_state["confidence"] = confidence
    st.session_state["car_count"] = car_count
    st.session_state["bike_count"] = bike_count
    st.session_state["bus_count"] = bus_count
    st.session_state["truck_count"] = truck_count
    st.session_state["probability"] = probability


# =========================================================
# DISPLAY RESULTS
# =========================================================

if "traffic" in st.session_state:

    traffic = st.session_state["traffic"]
    hour = st.session_state["hour"]
    traffic_period = st.session_state["traffic_period"]
    confidence = st.session_state["confidence"]

    car_count = st.session_state["car_count"]
    bike_count = st.session_state["bike_count"]
    bus_count = st.session_state["bus_count"]
    truck_count = st.session_state["truck_count"]

    probability = st.session_state["probability"]


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.divider()

    st.header("📋 Prediction Summary")


    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.metric(
            "Predicted Traffic",
            traffic
        )


    with summary2:

        st.metric(
            "Selected Hour",
            f"{hour}:00"
        )


    with summary3:

        st.metric(
            "Traffic Period",
            traffic_period
        )


    # =====================================================
    # TRAFFIC STATUS
    # =====================================================

    traffic_lower = str(traffic).lower()


    if traffic_lower == "low":

        st.success(
            "🟢 Low Traffic"
        )

    elif traffic_lower == "normal":

        st.info(
            "🟡 Normal Traffic"
        )

    elif traffic_lower == "high":

        st.warning(
            "🟠 High Traffic"
        )

    elif traffic_lower == "heavy":

        st.error(
            "🔴 Heavy Traffic"
        )

    else:

        st.info(
            f"Traffic Condition: {traffic}"
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.subheader("🎯 Model Confidence")

    st.progress(
        min(int(confidence), 100)
    )

    st.write(
        f"Prediction confidence: **{confidence:.2f}%**"
    )


    # =====================================================
    # PROBABILITY CHART
    # =====================================================

    if probability is not None:

        st.subheader(
            "📊 Prediction Probability"
        )


        probability_df = pd.DataFrame({

            "Traffic Situation":
                le.classes_,

            "Probability":
                probability * 100

        })


        st.bar_chart(

            probability_df.set_index(
                "Traffic Situation"
            )

        )


    # =====================================================
    # GEMINI AI RECOMMENDATION
    # =====================================================

    st.divider()

    st.header(
        "🤖 AI Traffic Recommendation"
    )


    with st.spinner(
        "Gemini AI is analyzing the traffic conditions..."
    ):

        recommendation = get_gemini_recommendation(

            traffic,

            hour,

            car_count,

            bike_count,

            bus_count,

            truck_count

        )


    st.info(
        recommendation
    )


    # =====================================================
    # BASIC SMART RECOMMENDATION
    # =====================================================

    st.subheader(
        "💡 Traffic Management Guidance"
    )


    if traffic_lower == "low":

        st.success(
            "Traffic is currently low. "
            "Normal traffic monitoring is sufficient."
        )


    elif traffic_lower == "normal":

        st.info(
            "Traffic is at a normal level. "
            "Continue monitoring, especially near peak hours."
        )


    elif traffic_lower == "high":

        st.warning(
            "Traffic is high. Consider optimizing "
            "traffic signal timing and encouraging "
            "public transportation."
        )


    elif traffic_lower == "heavy":

        st.error(
            "Heavy congestion detected. Consider "
            "route diversion, traffic control measures, "
            "and increased public transportation."
        )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.divider()

    st.header(
        "📥 Download Prediction Report"
    )


    report = f"""
SMARTFLOW AI - TRAFFIC PREDICTION REPORT
=========================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TRAFFIC PREDICTION
------------------
Predicted Traffic: {traffic}
Selected Hour: {hour}:00
Traffic Period: {traffic_period}
Model Confidence: {confidence:.2f}%

VEHICLE INFORMATION
-------------------
Cars: {car_count}
Bikes: {bike_count}
Buses: {bus_count}
Trucks: {truck_count}

MODEL
-----
Machine Learning Model: Random Forest
AI Engine: Gemini

AI TRAFFIC RECOMMENDATION
-------------------------
{recommendation}

=========================================
SmartFlow AI
Smart Traffic Prediction using
Machine Learning and Generative AI
"""


    st.download_button(

        label="📥 Download Prediction Report",

        data=report,

        file_name="SmartFlow_AI_Traffic_Report.txt",

        mime="text/plain",

        use_container_width=True

    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SmartFlow AI | Smart City Traffic Prediction "
    "using Machine Learning and Generative AI"
)