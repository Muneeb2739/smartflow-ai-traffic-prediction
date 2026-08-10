import streamlit as st
import pandas as pd
import joblib
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart City Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# LOAD GEMINI
# =========================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load("traffic_model.pkl")
le = joblib.load("label_encoder.pkl")


# =========================================================
# LOAD DATASET
# =========================================================

traffic_df = pd.read_csv("data/Traffic.csv")

traffic_df["Time"] = pd.to_datetime(
    traffic_df["Time"]
)

traffic_df["Hour"] = traffic_df["Time"].dt.hour


# =========================================================
# TITLE
# =========================================================

st.title("🚦 Smart City Traffic Prediction System")

st.write(
    "Predict traffic conditions using Machine Learning "
    "and generate smart recommendations using Gemini AI."
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.subheader("📊 Traffic Dataset Overview")

total_records = len(traffic_df)

average_cars = traffic_df["CarCount"].mean()

average_hour = traffic_df["Hour"].mean()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📄 Total Records",
        total_records
    )

with col2:
    st.metric(
        "🚗 Average Cars",
        round(average_cars, 1)
    )

with col3:
    st.metric(
        "🕐 Average Hour",
        round(average_hour, 1)
    )


# =========================================================
# TRAFFIC INPUT
# =========================================================

st.subheader("🚗 Traffic Information")

col1, col2, col3, col4 = st.columns(4)


with col1:
    car_count = st.number_input(
        "🚗 Car Count",
        min_value=0,
        value=100
    )


with col2:
    bike_count = st.number_input(
        "🏍️ Bike Count",
        min_value=0,
        value=20
    )


with col3:
    bus_count = st.number_input(
        "🚌 Bus Count",
        min_value=0,
        value=10
    )


with col4:
    truck_count = st.number_input(
        "🚚 Truck Count",
        min_value=0,
        value=5
    )


# =========================================================
# TIME INPUT
# =========================================================

st.subheader("🕐 Time Information")

hour = st.slider(
    "Select Hour",
    min_value=0,
    max_value=23,
    value=18
)

st.write(
    f"Selected time: **{hour}:00**"
)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🚦 Predict Traffic",
    use_container_width=True
):

    # -----------------------------------------------------
    # Create input data
    # -----------------------------------------------------

    new_data = pd.DataFrame({
        "CarCount": [car_count],
        "BikeCount": [bike_count],
        "BusCount": [bus_count],
        "TruckCount": [truck_count],
        "Hour": [hour]
    })


    # -----------------------------------------------------
    # Predict traffic
    # -----------------------------------------------------

    prediction = model.predict(new_data)


    # Convert encoded value to original label

    traffic = le.inverse_transform(
        prediction
    )[0]


    # -----------------------------------------------------
    # Prediction probability
    # -----------------------------------------------------

    probability = model.predict_proba(
        new_data
    )[0]


    # Model confidence

    confidence = probability.max() * 100


    # =====================================================
    # DISPLAY PREDICTION
    # =====================================================

    result1, result2 = st.columns(2)


    with result1:

        st.subheader("🚦 Traffic Prediction")


        if traffic.lower() == "low":

            st.success("🟢 LOW TRAFFIC")


        elif traffic.lower() == "normal":

            st.info("🟡 NORMAL TRAFFIC")


        elif traffic.lower() == "high":

            st.warning("🟠 HIGH TRAFFIC")


        elif traffic.lower() == "heavy":

            st.error("🔴 HEAVY TRAFFIC")


        else:

            st.write(traffic)


    with result2:

        st.subheader("🎯 Prediction Confidence")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


    # =====================================================
    # PROBABILITY CHART
    # =====================================================

    st.subheader("📊 Prediction Probability")


    probability_df = pd.DataFrame({
        "Traffic Situation": le.classes_,
        "Probability": probability * 100
    })


    st.bar_chart(
        probability_df.set_index(
            "Traffic Situation"
        )
    )


    # =====================================================
    # PEAK HOUR
    # =====================================================

    if 7 <= hour <= 9 or 17 <= hour <= 20:

        peak_hour = "Yes"

    else:

        peak_hour = "No"


    # =====================================================
    # GEMINI AI RECOMMENDATION
    # =====================================================

    st.subheader(
        "🤖 Smart Traffic Recommendation"
    )


    prompt = f"""
You are a smart city traffic management assistant.

Traffic information:

Car count: {car_count}
Bike count: {bike_count}
Bus count: {bus_count}
Truck count: {truck_count}
Hour: {hour}:00
Peak hour: {peak_hour}

Machine learning prediction:
Traffic situation: {traffic}
Prediction confidence: {confidence:.2f}%

Give a short practical traffic management recommendation.

Consider:
- Traffic signal optimization
- Public transportation
- Alternative routes
- Traffic congestion
- Peak-hour management

Keep the answer short, clear and suitable for a dashboard.
"""


    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        recommendation = response.text

        st.info(recommendation)


    except Exception as e:

        recommendation = (
            "Gemini recommendation is currently unavailable."
        )

        st.warning(recommendation)


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.subheader(
        "📥 Download Prediction Report"
    )


    report = pd.DataFrame({

        "Car Count": [car_count],

        "Bike Count": [bike_count],

        "Bus Count": [bus_count],

        "Truck Count": [truck_count],

        "Hour": [hour],

        "Peak Hour": [peak_hour],

        "Predicted Traffic": [traffic],

        "Confidence": [
            f"{confidence:.2f}%"
        ],

        "AI Recommendation": [
            recommendation
        ]

    })


    csv_report = report.to_csv(
        index=False
    )


    st.download_button(

        label="📥 Download Report",

        data=csv_report,

        file_name="traffic_prediction_report.csv",

        mime="text/csv"

    )


# =========================================================
# HISTORICAL TRAFFIC CHART
# =========================================================

st.divider()

st.subheader(
    "📈 Historical Traffic Situation"
)


traffic_counts = (
    traffic_df["Traffic Situation"]
    .value_counts()
)


st.bar_chart(
    traffic_counts
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚦 Smart City Traffic Prediction | "
    "Random Forest + Gemini AI"
)