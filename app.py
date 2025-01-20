import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import cryptocompare
import pickle
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from nixtla import NixtlaClient
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

# HTML and CSS for the neon heading animation
neon_text_html = """
<div style="text-align: center; margin-top: 50px;">
    <h1 class="heading">
        BITCOIN PRIC<span class="flicker">E</span>&nbsp;PREDICTOR
    </h1>
</div>

<style>
    .heading {
        font-size: 3em;
        color: #39ff14;
        font-family: Arial, sans-serif;
        text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14, 
                     0 0 40px #0fa, 0 0 80px #0fa, 0 0 90px #0fa, 
                     0 0 100px #0fa, 0 0 150px #0fa;
        letter-spacing: 5px;
        display: flex;
        justify-content: center;
    }

    .flicker {
        display: inline-block;
        color: #39ff14;
        animation: flicker 1.5s infinite alternate, shake 0.5s infinite;
    }

    @keyframes flicker {
        0%, 18%, 22%, 25%, 53%, 57%, 100% {
            opacity: 1;
            text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14, 
                         0 0 40px #0fa, 0 0 80px #0fa, 0 0 90px #0fa, 
                         0 0 100px #0fa, 0 0 150px #0fa;
        }
        20%, 24%, 55% {
            opacity: 0.2;
            text-shadow: none;
        }
    }

    @keyframes shake {
        0% { transform: translate(1px, 1px) rotate(0deg); }
        10% { transform: translate(-1px, -2px) rotate(-1deg); }
        20% { transform: translate(-3px, 0px) rotate(1deg); }
        30% { transform: translate(3px, 2px) rotate(0deg); }
        40% { transform: translate(1px, -1px) rotate(1deg); }
        50% { transform: translate(-1px, 2px) rotate(-1deg); }
        60% { transform: translate(-3px, 1px) rotate(0deg); }
        70% { transform: translate(3px, 1px) rotate(-1deg); }
        80% { transform: translate(-1px, -1px) rotate(1deg); }
        90% { transform: translate(1px, 2px) rotate(0deg); }
        100% { transform: translate(1px, -2px) rotate(-1deg); }
    }
</style>
"""
st.markdown(neon_text_html, unsafe_allow_html=True)

# HTML and CSS for the text roller at the bottom right
text_roller_html = """
<div class="roller">
   <span class="class">System</span>.<span class="string">out</span>.<span class="method">madeby</span>(<span class="hover-text"></span>);
</div>

<style>
        .roller {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-family: monospace;
        font-size: 1.2em;
        white-space: nowrap;
        overflow: hidden;
    }

    /* Colors for different parts of the text */
    .roller .keyword {
        color: #f78c6c; /* orange-like color for keywords */
        text-shadow: 0 0 5px #f78c6c, 0 0 10px #f78c6c;
    }

    .roller .class {
        color: #82aaff; /* blue-like color for class names */
        text-shadow: 0 0 5px #82aaff, 0 0 10px #82aaff;
    }

    .roller .method {
        color: orange; /* purple color for method names */
        text-shadow: 0 0 5px orange, 0 0 10px orange;
    }

    .roller .string {
        color: #c792ea; /* green color for string literals */
        text-shadow: 0 0 5px #c792ea, 0 0 10px #c792ea;
    }

    .roller .hover-text {
        color: #ff1053; /* Change hover color here */
        transition: color 0.3s ease-in-out;
    }

    .hover-text::before {
        content: "Sneha Sharma";
    }
    .hover-text:hover::before {
        content: "Data Analyst";
        color: ; /* Change this to another color for hover effect */
    }
</style>
"""
st.markdown(text_roller_html, unsafe_allow_html=True)

# CSS for gradient background, padding, margin, and vertical divider
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #e66465, #9198e5);
        padding: 20px 50px;
    }
    .css-1r6slb0.e1fqkh3o8 {
        margin-top: 50px;
    }
    .css-18e3th9 {
        padding: 20px;
    }
    .vertical-line {
        border-left: 2px solid #CCCCCC;
        height: 100%;
        position: absolute;
        left: 50%;
        top: 0;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .red-button {
        background-color: red;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
lstm_model = load_model("D:\\Projects\\Bitcoin Price Prediction\\lstm_model.h5")
scaler = MinMaxScaler()

with open("D:\\Projects\\Bitcoin Price Prediction\\prophet_model.pkl", 'rb') as f:
    prophet_model = pickle.load(f)

with open("D:\\Projects\\Bitcoin Price Prediction\\arima_model.pkl", 'rb') as f:
    arima_model = joblib.load(f)

nixtla_client = NixtlaClient(api_key='nixtla-tok-xmAyKHqhIBdxxbi4cHkXXu2QxqAIhzfmykdQw4SkorJjqlO2ZjI5JuDQ0yYiCZplmhaDM0DPbnDbhpMs')

# Prediction functions
def make_prophet_predictions(selected_date):
    future_df = pd.DataFrame({'ds': [selected_date]})
    forecast = prophet_model.predict(future_df)
    return forecast['yhat'].values[0]

def make_arima_prediction(selected_date):
    selected_date_ts = pd.to_datetime(selected_date)
    n_periods = (selected_date_ts - df['Date'].max()).days
    if n_periods < 0:
        st.warning("Selected date is before the latest available data.")
        return None
    
    try:
        forecast = arima_model.predict(steps=n_periods)
        return forecast[-1]
    except Exception as e:
        st.write(f"Error during ARIMA prediction: {str(e)}")
        return None

def make_timegpt_forecast(df, h=10, level=[50, 80, 90]):
    try:
        df_timegpt = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        fcst = nixtla_client.forecast(df_timegpt, h=h, level=level, add_history=True)
        return fcst['TimeGPT'].iloc[0]
    except Exception as e:
        st.write(f"Error during TimeGPT prediction: {str(e)}")
        return None

def make_lstm_prediction():
    if df is not None and 'Close' in df.columns:
        data_scaled = scaler.fit_transform(df[['Close']].values)
        n_steps = 60
        X_test = []

        if len(data_scaled) >= n_steps:
            for i in range(n_steps, len(data_scaled)):
                X_test.append(data_scaled[i-n_steps:i, 0])

            X_test = np.array(X_test)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            lstm_predicted_price_scaled = lstm_model.predict(X_test)
            lstm_predicted_price = scaler.inverse_transform(lstm_predicted_price_scaled)

            return lstm_predicted_price[-1][0]
        else:
            st.warning(f"Not enough data to make a prediction. Need at least {n_steps} data points.")
            return None
    else:
        st.warning("Dataframe is empty or 'Close' column is missing.")
        return None

with st.container():

    # Fetch historical price data
    btc_data = cryptocompare.get_historical_price_day('BTC', currency='USD', toTs=datetime.now())
    if btc_data:
        df = pd.DataFrame(btc_data)

        # Ensure 'time' is converted to 'Date'
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volumeto': 'Volume'}, inplace=True)
            df['Adj Close'] = df['Close']
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

            # Add a slider for selecting date range
            min_date = df['Date'].min().date()  # Convert to date
            max_date = datetime.now().date()  # Set max date to today's date
            
            start_date, end_date = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Filter the DataFrame based on the selected date range
            filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

            # Visualize Bitcoin Price Over Time for the selected range
            fig = go.Figure(data=[go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Close'],
                mode='lines+markers',
                marker=dict(
                    color=filtered_df['Close'],
                    colorscale='Viridis',
                    size=8
                )
            )])

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                title='Bitcoin Price Evolution'
            )

            st.plotly_chart(fig)

            selected_date = st.date_input("Select a Date for Prediction", datetime.now().date())
            selected_model = st.selectbox("Select Prediction Model", ["Prophet", "ARIMA", "LSTM", "TimeGPT"])

            # Button for prediction
            if st.button("Predict Price", key="predict_button"):

                if selected_model == "Prophet":
                    prediction = make_prophet_predictions(selected_date)
                    if prediction is not None:
                        st.write(f"Predicted value using {selected_model}: {round(prediction, 2)}")

                elif selected_model == "ARIMA":
                    prediction = make_arima_prediction(selected_date)
                    if prediction is not None:
                        st.write(f"Predicted value using {selected_model}: {round(prediction, 2)}")

                elif selected_model == "LSTM":
                    prediction = make_lstm_prediction()
                    if prediction is not None:
                        st.write(f"Predicted value using {selected_model}: {round(prediction, 2)}")

                elif selected_model == "TimeGPT":
                    prediction = make_timegpt_forecast(df)
                    if prediction is not None:
                        st.write(f"Predicted value using {selected_model}: {round(prediction, 2)}")
        else:
            st.warning("Could not retrieve historical price data.")
    else:
        st.warning("Failed to retrieve data from CryptoCompare.")