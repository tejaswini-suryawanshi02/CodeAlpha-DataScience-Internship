# ==========================================
# CodeAlpha Internship - Task 2
# Unemployment Analysis Dashboard
# Created by: Tejaswini Suryawanshi
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------
# Custom Styling
# ------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#F5F7FA;
}

[data-testid="stSidebar"]{
    background:#E8F0FE;
}

h1,h2,h3{
    color:#1F3A5F;
}

div[data-testid="metric-container"]{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("Unemployment.csv")

    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Estimated Unemployment Rate (%)":"Unemployment_Rate",
        "Estimated Employed":"Employed",
        "Estimated Labour Participation Rate (%)":"Labour_Rate"
    }, inplace=True)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df.dropna(inplace=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

    return df

df = load_data()

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.title("📊 Dashboard Menu")

page = st.sidebar.radio(
    "Select Option",
    [
        "Overview",
        "State Analysis",
        "Covid Analysis",
        "Heatmap",
        "Insights"
    ]
)

state = st.sidebar.selectbox(
    "Select State",
    ["All States"] + sorted(df["Region"].unique())
)

filtered_df = df.copy()

if state != "All States":
    filtered_df = df[df["Region"] == state]

# ------------------------------------------
# Overview
# ------------------------------------------

if page == "Overview":

    st.title("📊 Unemployment Analysis Dashboard")

    st.markdown("**Developed by: Tejaswini Suryawanshi**")

    st.write(
        "This dashboard provides an interactive analysis of unemployment trends across different states in India."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Average Rate",
        f"{df['Unemployment_Rate'].mean():.2f}%"
    )

    col3.metric(
        "Highest Rate",
        f"{df['Unemployment_Rate'].max():.2f}%"
    )

    col4.metric(
        "Lowest Rate",
        f"{df['Unemployment_Rate'].min():.2f}%"
    )

    col5.metric(
        "Total States",
        df["Region"].nunique()
    )

    st.info("The dataset contains unemployment statistics collected across different states of India.")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Overall Trend")

    trend = (
        df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Date",
        y="Unemployment_Rate",
        markers=True,
        title="Average Unemployment Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------
# State Analysis
# ------------------------------------------

elif page == "State Analysis":

    st.title("📈 State-wise Unemployment Analysis")

    trend = (
        filtered_df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Date",
        y="Unemployment_Rate",
        markers=True,
        title=f"Unemployment Trend - {state}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Average Unemployment Rate")

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    monthly = (
        filtered_df.groupby("Month")["Unemployment_Rate"]
        .mean()
        .reindex(month_order)
        .fillna(0)
        .reset_index()
    )

    fig2 = px.bar(
        monthly,
        x="Month",
        y="Unemployment_Rate",
        color="Unemployment_Rate",
        title="Month-wise Average Unemployment Rate"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# Covid Analysis
# ------------------------------------------

elif page == "Covid Analysis":

    st.title("🦠 Covid-19 Impact Analysis")

    covid_df = filtered_df.copy()

    covid_df["Covid_Period"] = np.where(
        covid_df["Date"] >= pd.Timestamp("2020-03-01"),
        "During Covid",
        "Before Covid"
    )

    covid = (
        covid_df.groupby("Covid_Period")["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        covid,
        x="Covid_Period",
        y="Unemployment_Rate",
        color="Covid_Period",
        title="Average Unemployment Before and During Covid"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("State Comparison")

    state_avg = (
        filtered_df.groupby("Region")["Unemployment_Rate"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        state_avg,
        x="Region",
        y="Unemployment_Rate",
        color="Unemployment_Rate",
        title="Top 10 States by Average Unemployment Rate"
    )

    st.plotly_chart(fig2, use_container_width=True)
# ------------------------------------------
# Heatmap
# ------------------------------------------

elif page == "Heatmap":

    st.title("🌡️ Correlation Heatmap")

    numeric_data = filtered_df[
        ["Unemployment_Rate", "Employed", "Labour_Rate"]
    ]

    corr = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(7,5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Employment vs Unemployment")

    fig2 = px.scatter(
        filtered_df,
        x="Employed",
        y="Unemployment_Rate",
        color="Region",
        size="Labour_Rate",
        hover_name="Region",
        title="Employment vs Unemployment"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# Insights
# ------------------------------------------

elif page == "Insights":

    st.title("📑 Key Insights")

    highest_state = (
        df.groupby("Region")["Unemployment_Rate"]
        .mean()
        .idxmax()
    )

    highest_rate = (
        df.groupby("Region")["Unemployment_Rate"]
        .mean()
        .max()
    )

    lowest_state = (
        df.groupby("Region")["Unemployment_Rate"]
        .mean()
        .idxmin()
    )

    st.success("Project Summary")

    st.write(f"""
### Key Findings

• Highest average unemployment was observed in **{highest_state}**
with approximately **{highest_rate:.2f}%** unemployment.

• States with higher labour participation generally show
different unemployment patterns.

• Covid-19 significantly affected unemployment levels
across several states.

• Data visualization helps in understanding employment trends
more effectively.

### Recommendations

1. Improve skill development programs.
2. Increase employment opportunities.
3. Support MSMEs and startups.
4. Focus on state-specific employment policies.
5. Encourage youth employment initiatives.
""")

    st.markdown("---")
    st.caption("© 2026 | Developed by Tejaswini Suryawanshi | CodeAlpha Data Science Internship")
    st.success("Overall, the dashboard helps understand unemployment trends and supports data-driven decision making.")