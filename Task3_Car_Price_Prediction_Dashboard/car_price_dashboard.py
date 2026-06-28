# ==========================================
# CodeAlpha Internship - Task 3
# Car Price Prediction Dashboard
# Created by: Tejaswini Suryawanshi
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("### CodeAlpha Internship - Task 3")
st.write("Predict the selling price of a used car using Machine Learning.")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("car data.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
data["Car_Age"] = 2026 - data["Year"]

# Remove Car Name
data.drop("Car_Name", axis=1, inplace=True)

# -----------------------------
# Convert Categorical Columns
# -----------------------------
data = pd.get_dummies(
    data,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

# -----------------------------
# Features and Target
# -----------------------------
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Model Performance
# -----------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Price Analysis",
        "Prediction",
        "Correlation Heatmap"
    ]
)

# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.header("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows : {data.shape[0]}")
        st.write(f"Columns : {data.shape[1]}")

    with col2:
        st.subheader("Model Performance")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Mean Absolute Error", f"{mae:.3f}")

    st.markdown("---")

    st.subheader("First 5 Records")
    st.dataframe(data.head())

    st.markdown("---")

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isnull().sum().values
    })

    st.dataframe(info_df)

    st.markdown("---")

    st.subheader("Statistical Summary")
    st.dataframe(data.describe())

    st.markdown("---")

    st.subheader("Missing Values")

    missing = data.isnull().sum()

    st.dataframe(
        pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })
    )

# ==========================================
# PRICE ANALYSIS PAGE
# ==========================================

elif page == "Price Analysis":

    st.header("📈 Price Analysis")

    # -----------------------------
    # Selling Price Distribution
    # -----------------------------
    st.subheader("Selling Price Distribution")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(data["Selling_Price"], bins=20)
    ax.set_xlabel("Selling Price (Lakhs)")
    ax.set_ylabel("Number of Cars")
    ax.set_title("Distribution of Selling Price")
    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # Present Price vs Selling Price
    # -----------------------------
    st.subheader("Present Price vs Selling Price")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(data["Present_Price"], data["Selling_Price"])
    ax.set_xlabel("Present Price")
    ax.set_ylabel("Selling Price")
    ax.set_title("Present Price vs Selling Price")
    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # Fuel Type Analysis
    # -----------------------------
    st.subheader("Average Selling Price by Fuel Type")

    fuel_columns = [
        "Fuel_Type_Diesel",
        "Fuel_Type_Petrol"
    ]

    fuel_avg = []

    for col in fuel_columns:
        fuel_avg.append(
            data[data[col] == 1]["Selling_Price"].mean()
        )

    fuel_names = ["Diesel", "Petrol"]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(fuel_names, fuel_avg)
    ax.set_ylabel("Average Selling Price")
    ax.set_title("Fuel Type Analysis")
    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # Transmission Analysis
    # -----------------------------
    st.subheader("Average Selling Price by Transmission")

    auto_avg = data[data["Transmission_Manual"] == 0]["Selling_Price"].mean()
    manual_avg = data[data["Transmission_Manual"] == 1]["Selling_Price"].mean()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(
        ["Automatic", "Manual"],
        [auto_avg, manual_avg]
    )

    ax.set_ylabel("Average Selling Price")
    ax.set_title("Transmission Analysis")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # Owner Analysis
    # -----------------------------
    st.subheader("Average Selling Price by Owner")

    owner_avg = data.groupby("Owner")["Selling_Price"].mean()

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(owner_avg.index.astype(str), owner_avg.values)

    ax.set_xlabel("Owner")
    ax.set_ylabel("Average Selling Price")
    ax.set_title("Owner vs Selling Price")

    st.pyplot(fig)

# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Prediction":

    st.header("🚗 Car Price Prediction")

    st.write("Enter the details of the car below:")

    # -----------------------------
    # User Inputs
    # -----------------------------
    year = st.number_input(
        "Year of Purchase",
        min_value=2000,
        max_value=2026,
        value=2018
    )

    present_price = st.number_input(
        "Present Price (in Lakhs)",
        min_value=0.0,
        value=5.0
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=30000
    )

    owner = st.selectbox(
        "Number of Previous Owners",
        [0, 1, 2, 3]
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    seller = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("Predict Price"):

        car_age = 2026 - year

        fuel_diesel = 1 if fuel == "Diesel" else 0
        fuel_petrol = 1 if fuel == "Petrol" else 0

        seller_individual = 1 if seller == "Individual" else 0

        transmission_manual = 1 if transmission == "Manual" else 0

        input_data = pd.DataFrame({
            "Year": [year],
            "Present_Price": [present_price],
            "Driven_kms": [kms_driven],
            "Owner": [owner],
            "Car_Age": [car_age],
            "Fuel_Type_Diesel": [fuel_diesel],
            "Fuel_Type_Petrol": [fuel_petrol],
            "Selling_type_Individual": [seller_individual],
            "Transmission_Manual": [transmission_manual]
        })

        prediction = model.predict(input_data)

        st.success(
            f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs"
        )

# ==========================================
# CORRELATION HEATMAP PAGE
# ==========================================

elif page == "Correlation Heatmap":

    st.header("🔥 Correlation Heatmap")

    st.write("Correlation between numerical features in the dataset.")

    corr = data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap = ax.imshow(corr, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    plt.colorbar(heatmap)

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Correlation Matrix")
    st.dataframe(corr.round(2))

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:16px;'>
        © 2026 | Developed by <b>Tejaswini Suryawanshi</b> |
        CodeAlpha Data Science Internship
    </div>
    """,
    unsafe_allow_html=True
)