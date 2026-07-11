"""
SmartStay — Streamlit Deployment App
Enter listing details -> get an instant fair-price estimate + the top reasons
driving that number. Run locally with: streamlit run app.py
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="SmartStay Price Estimator", page_icon="🏠", layout="centered")

MODEL_PATH = "smartstay_model.joblib"
CITY_CENTER = (40.7580, -73.9855)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🏠 SmartStay — Fair Price Estimator")
st.caption("Explainable rental price predictor · Gradient Boosting Regressor")

col1, col2 = st.columns(2)
with col1:
    borough = st.selectbox("Neighbourhood group", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
    min_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=2)
with col2:
    num_reviews = st.number_input("Number of reviews", min_value=0, max_value=1000, value=10)
    reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=30.0, value=1.0)
    host_listings = st.number_input("Host's total listings", min_value=1, max_value=200, value=1)
availability = st.slider("Availability (days/year)", 0, 365, 200)
distance_km = st.slider("Distance to city center (km)", 0.0, 25.0, 5.0)

if st.button("Predict fair price", type="primary"):
    row = pd.DataFrame([{
        "minimum_nights": min_nights,
        "number_of_reviews": num_reviews,
        "reviews_activity_score": reviews_per_month,
        "distance_to_center_km": distance_km,
        "host_scale": np.log1p(host_listings),
        "availability_ratio": availability / 365.0,
        "neighbourhood_group": borough,
        "room_type": room_type,
    }])
    log_pred = model.predict(row)[0]
    price = np.expm1(log_pred)
    st.metric("Estimated fair nightly price", f"${price:,.0f}")

    importances = model.named_steps["model"].feature_importances_
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["neighbourhood_group", "room_type"]))
    num_names = ["minimum_nights", "number_of_reviews", "reviews_activity_score",
                 "distance_to_center_km", "host_scale", "availability_ratio"]
    imp_df = pd.DataFrame({"feature": num_names + cat_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(3)
    st.write("**Top factors behind this estimate:**")
    for _, r in imp_df.iterrows():
        st.write(f"- {r['feature'].replace('_', ' ').title()}")
