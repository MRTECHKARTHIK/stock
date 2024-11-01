import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import time
import requests
import re
from sklearn.linear_model import LinearRegression
import numpy as np

# Set up the Streamlit app header
st.header('Indian Stock Dashboard')

selected_ticker = st.session_state.get('selected_ticker', None)

# Initialize selected ticker variable
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = None

# Watchlist section with columns
with st.expander("Watchlist", expanded=False):
    watchlist = [
        'INFY', 'ITC', 'ZOMATO', 'NMDC', 'MANINFRA', 'BPCL', 
        'GAIL', 'SAIL', 'IRFC', 'IREDA', 'RVNL', 'TATASTEEL', 
        'SUZLON', 'IOC', 'CANBK', 'ENGINERSIN', 
        'UNIONBANK', 'PNB', 'TATAMOTORS', 'VRLLOG', 'BEL', 
        'COALINDIA', 'BHEL', 'VEDL', 'BANKBARODA', 'SBIN', 
        'ADANIPOWER', 'AWL', 'HINDUNILVR', 'PHOENIXLTD', 'HAL', 
        'SIEMENS', 'TCS', 'RELIANCE', 'HDFCBANK', 'ADANIGREEN', 
        'ULTRACEMCO', 'ALSTONE'
    ]

    num_columns = 4
    columns = st.columns(num_columns)
    
    for index, company in enumerate(watchlist):
        with columns[index % num_columns]:  # Cycle through columns
            if st.button(company, key=f"{company}_{index}"):
                selected_ticker = company
                st.session_state['selected_ticker'] = company  # Store in session state

    if selected_ticker:
        st.write(f"Currently selected: {selected_ticker}")

# Sidebar inputs
ticker = st.sidebar.text_input('Symbol Code', selected_ticker or "")
exchange = st.sidebar.selectbox('Select Exchange', ['NSE', 'BSE'])

if exchange == 'NSE' and not ticker.endswith('.NS'):
    ticker += '.NS'
elif exchange == 'BSE' and not ticker.endswith('.BO'):
    ticker += '.BO'

# Option to enable live update
live_update = st.sidebar.checkbox("Enable Live Update (Refresh every 10 seconds)", value=True)

# Set "Today" as the default time range
time_range = st.sidebar.selectbox(
    'Select Time Range',
    ('Today', '5 Days', '1 Month', '1 Quarter', '1 Year', 'Lifetime'),
    index=0  # Set to 0 for "Today"
)

# Get time period and chart type
chart_type = st.sidebar.selectbox('Select Chart Type', ('Candlestick', 'Baseline', 'High-Low'))

# Function to get stock data
@st.cache_data
def get_stock_data(selected_ticker, interval='1m'):
    stock = yf.Ticker(selected_ticker)
    try:
        hist = stock.history(period='1d', interval=interval)
        if hist.empty:
            st.error(f"No data found for {selected_ticker}. Please check the ticker symbol.")
        return hist
    except Exception as e:
        st.error(f"An error occurred while fetching data for {selected_ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

@st.cache_data
def get_performance_metrics(selected_ticker):
    stock = yf.Ticker(selected_ticker)
    history = stock.history(period='max')

    if history.empty:
        st.error(f"No historical data found for {selected_ticker}. Please check the ticker symbol.")
        return None  # Return None if there is no data

    # Today's High and Low
    todays_high = history['High'].iloc[-1]
    todays_low = history['Low'].iloc[-1]


    # All-Time High and Low
    all_time_high = history['High'].max()
    all_time_low = history['Low'].min()

    # 52-week High and Low (Past Year)
    past_year = history.tail(252)  # 252 trading days in a year
    week_52_high = past_year['High'].max() if not past_year.empty else 'N/A'
    week_52_low = past_year['Low'].min() if not past_year.empty else 'N/A'

    return todays_high, todays_low, all_time_high, all_time_low, week_52_high, week_52_low

@st.cache_data
def get_balance_sheet(ticker):
    stock = yf.Ticker(ticker)
    return stock.balance_sheet

@st.cache_data
def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

# Function to fetch company fundamentals including bid and ask prices
def get_company_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = {
        'Market Cap': stock.info.get('marketCap', 'N/A'),
        'P/E Ratio': stock.info.get('forwardPE', 'N/A'),
        'P/B Ratio': stock.info.get('priceToBook', 'N/A'),
        'Debt to Equity': stock.info.get('debtToEquity', 'N/A'),
        'ROE': stock.info.get('returnOnEquity', 'N/A'),
        'EPS': stock.info.get('trailingEps', 'N/A'),
        'Dividend Yield': stock.info.get('dividendYield', 'N/A'),
        'Book Value': stock.info.get('bookValue', 'N/A'),
        'Face Value': stock.info.get('faceValue', 'N/A'),
        'Bid Price': stock.info.get('bid', 'N/A'),
        'Ask Price': stock.info.get('ask', 'N/A'),
        'Bid Size': stock.info.get('bidSize', 'N/A'),
        'Ask Size': stock.info.get('askSize', 'N/A')
    }
    return fundamentals

# Function to fetch the latest news
def get_latest_news():
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'Indian stock market',
        'apiKey': '356a2bd6dbdd4b0abcd1fe9472ba59a6',  # Replace with your News API key
        'pageSize': 5,  # Number of news articles to fetch
        'sortBy': 'publishedAt'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error("Failed to fetch news articles.")
        return []

# Function to fetch additional company information
def get_company_about(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'Owner': info.get('owner', 'N/A'),  # Assuming 'owner' might be present, otherwise set to 'N/A'
        'Founded Year': info.get('yearFounded', 'N/A'),
        'Revenue': info.get('totalRevenue', 'N/A')  # This is typically in the format of USD
    }

try:
    # Initialize refresh time
    if 'last_run' not in st.session_state:
        st.session_state['last_run'] = time.time()

    current_time = time.time()

    # Refresh if live update is enabled and it's time to refresh
    if live_update and (current_time - st.session_state['last_run'] >= 10):
        st.session_state['last_run'] = current_time

    # Append the correct suffix based on the selected exchange
    if exchange == 'NSE':
        full_ticker = ticker + '.NS' if not ticker.endswith('.NS') else ticker
    else:  # For BSE
        full_ticker = ticker + '.BO' if not ticker.endswith('.BO') else ticker
    
    stock_data = get_stock_data(full_ticker)

    if not stock_data.empty:
        # Get performance metrics
        performance_metrics = get_performance_metrics(full_ticker)

        if performance_metrics is None:
            st.warning("Performance metrics could not be retrieved due to lack of historical data.")
        else:
            todays_high, todays_low, all_time_high, all_time_low, week_52_high, week_52_low = performance_metrics

            # Set up the plot
            if chart_type == 'Candlestick':
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                )])
                fig.update_layout(
                    title=f'{full_ticker} Stock Price (Live) - Candlestick Chart',
                    xaxis_title='Time',
                    yaxis_title='Price (INR)',
                    template='plotly_white'
                )

            elif chart_type == 'Baseline':
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Closing Price',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title=f'{full_ticker} Stock Price (Live) - Baseline Chart',
                    xaxis_title='Time',
                    yaxis_title='Price (INR)',
                    template='plotly_white'
                )

            elif chart_type == 'High-Low':
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['High'],
                    mode='lines',
                    name='High',
                    line=dict(color='green')
                ))
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Low'],
                    mode='lines',
                    name='Low',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title=f'{full_ticker} Stock Price (Live) - High-Low Chart',
                    xaxis_title='Time',
                    yaxis_title='Price (INR)',
                    template='plotly_white'
                )

            st.plotly_chart(fig)

            st.subheader("Performance Metrics")
            combined_metrics = pd.DataFrame({
                "Today's High": [todays_high],
                "Today's Low": [todays_low],
                "All-Time High": [all_time_high],
                "All-Time Low": [all_time_low],
                "52-Week High": [week_52_high],
                "52-Week Low": [week_52_low]
            })

            st.table(combined_metrics)

        st.subheader('Fundamentals')

# Function to add spaces between camel case words and capitalize first letters
        def format_text(text):
            if isinstance(text, str):
                # Add space before capital letters and capitalize
                formatted_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
                formatted_text = formatted_text.capitalize()
                return formatted_text
            return text

# Fetch fundamentals information
        fundamentals = get_company_fundamentals(full_ticker)

# Format the fundamentals with spaced keys and values
        formatted_fundamentals = {format_text(k): format_text(v) for k, v in fundamentals.items()}

# Convert dictionary to DataFrame and sort
        fundamentals_df = pd.DataFrame.from_dict(formatted_fundamentals, orient='index', columns=['Details']).sort_index()
        fundamentals_df = fundamentals_df.astype(str)
# Display the formatted DataFrame
        st.table(fundamentals_df)
        # Display balance sheet
        st.subheader('Balance Sheet')
        balance_sheet = get_balance_sheet(full_ticker)
        if balance_sheet is not None:
            st.write(balance_sheet)
        else:
            st.warning("Balance sheet data is unavailable.")

        # Display company info
        st.subheader("Company Information")

# Function to add spaces and format strings for better readability
        def format_text(text, add_space=False):
            if isinstance(text, str):
        # Replace underscores with spaces
                text = text.replace('_', ' ')
        
        # Add space before capital letters and capitalize
                formatted_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
                formatted_text = formatted_text.capitalize()

        # Add extra spaces for specific phrases
                if add_space:
                    formatted_text = formatted_text.replace('52week change', '52 week change ')
                    formatted_text = formatted_text.replace('Address1', 'Address 1')
                    formatted_text = formatted_text.replace('Address2', 'Address 2')
                    formatted_text = formatted_text.replace('Average daily volume10day', 'Average daily volume 10 day ')
                    formatted_text = formatted_text.replace('Sand p52 week change', 'Sand p 52 week change')
                    formatted_text = formatted_text.replace('Uuid', 'Uid')
                    formatted_text = formatted_text.replace('Average volume10days', 'Average volume 10 days ')

                return formatted_text
            return text

# Fetch company information
        company_info = get_company_info(full_ticker)

# Filter out unwanted keys
        filtered_info = {k: v for k, v in company_info.items()}

# Apply formatting to keys and values in the filtered company info dictionary
        formatted_info = {format_text(k, add_space=True): format_text(v) for k, v in filtered_info.items()}

# Convert the formatted dictionary to a DataFrame and sort by key
        company_info_df = pd.DataFrame.from_dict(formatted_info, orient='index', columns=['Details']).sort_index()
        company_info_df = company_info_df.astype(str)


# Display the formatted DataFrame in a table
        st.table(company_info_df)

        st.subheader("Latest News on Indian Stock Market")
        news_articles = get_latest_news()
        for article in news_articles:
            st.markdown(f"**{article['title']}**")
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")
            st.write("Published at: " + article['publishedAt'])
            st.write("---")

# Prediction Section for Stock Price
        st.subheader("Stock Price Prediction")

# Ensure ticker includes exchange suffix
        def generate_prediction_graph(ticker, exchange):
    # Append suffix based on the selected exchange
            if exchange == 'NSE':
                full_ticker = ticker + '.NS' if not ticker.endswith('.NS') else ticker
            else:  # BSE
                full_ticker = ticker + '.BO' if not ticker.endswith('.BO') else ticker
    
            data = get_stock_data(full_ticker)  # Fetch data with the correct ticker
            if data.empty:
                st.warning("No stock data available for prediction.")
                return

    # Use the closing price for prediction
            data['Date'] = data.index
            data['Day'] = np.arange(len(data))  # Add sequential day numbers
            X = data[['Day']]
            y = data['Close']

    # Fit a simple linear regression model
            model = LinearRegression()
            model.fit(X.values, y)            
            future_days =  30 # Predict for the next 30 days
            future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
            future_predictions = model.predict(future_X)

    # Plot the actual prices
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Price'))

    # Plot the predictions
            future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_days + 1, freq='D')[1:]
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted Price', line=dict(color='orange', dash='dash')))

            fig.update_layout(
                title=f'{full_ticker} Stock Price Prediction (Next 30 Days)',
                xaxis_title='Date',
                yaxis_title='Price (INR)',
                template='plotly_white'
            )

            st.plotly_chart(fig)

# Call the prediction function if a ticker is selected
        if selected_ticker:
            generate_prediction_graph(ticker, exchange)
        else:
            st.write("Please select a ticker to view the prediction graph.")

        def get_stock_data_for_recommendation(ticker):
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")  # Get the last 6 months of data
            return hist

# Function to calculate recommendation based on technical indicators
        def get_recommendation(ticker):
            data = get_stock_data_for_recommendation(ticker)
            if data.empty:
                return "No data available"
    
    # Calculate moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Short-term moving average (20-day)
            data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Long-term moving average (50-day)

    # Calculate RSI (Relative Strength Index)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

    # Ensure values for the last available data point
            if pd.isna(data['SMA_20'].iloc[-1]) or pd.isna(data['SMA_50'].iloc[-1]) or pd.isna(data['RSI'].iloc[-1]):
                return "Insufficient data for recommendation"
    
            current_price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]

    # Strong Buy
            if current_price > sma_50 and rsi < 30:
                return "Strong Buy"
    # Buy
            elif current_price > sma_20 and rsi < 50:
                return "Buy"
    # Hold
            elif sma_20 < current_price < sma_50:
                return "Hold"
    # Sell
            elif current_price < sma_20 and rsi > 70:
                return "Sell"
    # Strong Sell
            elif current_price < sma_50 and rsi > 70:
                return "Strong Sell"
            else:
                return "Hold"


# Add this function to display a speedometer-type gauge for stock recommendation
# Add this function to display a speedometer-type gauge for stock recommendation
        def display_recommendation_gauge(recommendation):
    # Map recommendation to a value for gauge
            recommendation_map = {
                "Strong Sell": 0,
                "Sell": 25,
                "Hold": 50,
                "Buy": 75,
                "Strong Buy": 100
            }

    # Get the value for the current recommendation
            recommendation_value = recommendation_map.get(recommendation, 50)  # Default to "Hold" if unknown

    # Create the gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendation_value,
                title={'text': "Stock Recommendation"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "red"},
                        {'range': [20, 40], 'color': "orange"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': recommendation_value
                    }
                }
            ))

    # Display the gauge chart in Streamlit
            st.plotly_chart(fig)

    # Display the recommendation text below the chart with larger, bold, and centered formatting
            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>{recommendation}</div>", unsafe_allow_html=True)

# After generating the recommendation, call this function to show the gauge
        st.subheader("Stock Recommendation Gauge for Selected Company")

        if selected_ticker:
    # Append the correct suffix based on the selected exchange
            full_ticker = f"{selected_ticker}.NS" if exchange == "NSE" else f"{selected_ticker}.BO"
            recommendation = get_recommendation(full_ticker)

    # Display the gauge based on the recommendation
            display_recommendation_gauge(recommendation)
        else:
            st.write("No company selected.")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
