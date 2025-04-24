import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("trained_pred_model.sav", "rb"))

model = load_model()

st.title("🛒 Big Mart Sales Predictor")
st.markdown("Predict the sales of retail products using a trained Random Forest model.")

# UI Elements
fat_content_map = {"Low Fat": 0, "Regular": 1}
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
location_type_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_type_map = {
    "Grocery Store": 0,
    "Supermarket Type1": 1,
    "Supermarket Type2": 2,
    "Supermarket Type3": 3
}
item_category_map = {"Food": 0, "Drinks": 1, "Non-Consumable": 2}

# Form for user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        item_fat = st.selectbox("Item Fat Content", list(fat_content_map.keys()))
        item_type = st.number_input("Item Type (encoded)", min_value=0, step=1, value=4)
        item_mrp = st.number_input("Item MRP", min_value=1.0, value=150.0)
        outlet_size = st.selectbox("Outlet Size", list(outlet_size_map.keys()))
        outlet_type = st.selectbox("Outlet Type", list(outlet_type_map.keys()))
        outlet = st.number_input("Outlet (encoded)", min_value=0, step=1, value=2)

    with col2:
        item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05)
        outlet_loc = st.selectbox("Outlet Location Type", list(location_type_map.keys()))
        outlet_year = st.number_input("Outlet Age (since 2013)", min_value=0, step=1, value=10)
        item = st.selectbox("Item Category", list(item_category_map.keys()))
        item_weight = st.number_input("Item Weight", min_value=1.0, value=10.0)

    submit = st.form_submit_button("Predict Sales")

# Predict and show result
if submit:
    input_array = np.array([[
        fat_content_map[item_fat],
        item_visibility,
        item_type,
        item_mrp,
        outlet_size_map[outlet_size],
        location_type_map[outlet_loc],
        outlet_type_map[outlet_type],
        outlet_year,
        outlet,
        item_category_map[item],
        item_weight
    ]])

    prediction = model.predict(input_array)
    st.success(f"💰 Predicted Sales: ₹{prediction[0]:,.2f}")
