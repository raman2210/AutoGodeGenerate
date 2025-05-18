import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta

# Set page configuration
st.set_page_config(
    page_title="Indian Index Fund Ranking Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .results-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<div class="main-header">Indian Index Fund Ranking Tool</div>', unsafe_allow_html=True)
st.markdown("""
This application calculates and ranks Indian Index funds based on Rate of Change (ROC) metrics.
Select your funds, date range, and ROC parameters to analyze fund performance.
""")

# Load pre-configured list of Indian index funds from CSV
@st.cache_data
def load_fund_list():
    # In a real application, this would load from a CSV file
    # For this demo, we'll create a sample dataframe
    funds_df = pd.DataFrame({
        'Fund Name': [
            'Nifty 50 Index Fund',
            'Sensex Index Fund',
            'Nifty Next 50 Index Fund',
            'Nifty Midcap 150 Index Fund',
            'Nifty Smallcap 250 Index Fund',
            'Gold ETF Fund',
            'Nifty Bank Index Fund',
            'Nifty IT Index Fund',
            'Nifty FMCG Index Fund',
            'Nifty Pharma Index Fund'
        ],
        'Ticker': [
            'NIFTYBEES.NS',
            'SETFNIF50.NS',
            'SETFNN50.NS',
            'JUNIORBEES.NS',
            'SETFNIFBK.NS',
            'GOLDBEES.NS',
            'BANKBEES.NS',
            'SETFNIFIT.NS',
            'CONSUMERBEES.NS',
            'PHARMBEES.NS'
        ]
    })
    return funds_df

# Load ROC parameters from CSV
@st.cache_data
def load_roc_params():
    # In a real application, this would load from a CSV file
    # For this demo, we'll create sample dataframes
    roc_days_df = pd.DataFrame({
        'Days': [5, 8, 13, 21, 34]
    })
    
    multiply_factors_df = pd.DataFrame({
        'Factors': [34, 21, 13, 8, 5]
    })
    
    return roc_days_df, multiply_factors_df

# Calculate ROC for a given number of days
def calculate_roc(prices, days):
    roc = prices.pct_change(days) * 100
    return roc

# Download data from yfinance
def download_fund_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    # If only one ticker is selected, yfinance may return a Series, so convert to DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    return data

# Calculate ROC metrics and ranking
def calculate_metrics(fund_data, roc_days, multiply_factors):
    results = {}
    
    for fund in fund_data.columns:
        # Calculate ROC for each day value
        roc_values = pd.DataFrame(index=fund_data.index)
        
        for i, days in enumerate(roc_days):
            roc_column = f'ROC_{days}'
            factor = multiply_factors[i]
            
            # Calculate ROC
            roc = calculate_roc(fund_data[fund], days)
            
            # Multiply by factor
            weighted_roc = roc * factor
            
            # Add to results
            roc_values[roc_column] = weighted_roc
        
        # Calculate sum of weighted ROCs
        roc_values['Sum_ROC'] = roc_values.sum(axis=1)
        
        # Store results
        results[fund] = roc_values
    
    # Combine all results into one dataframe
    combined_results = pd.DataFrame()
    
    for fund, data in results.items():
        combined_results[fund] = data['Sum_ROC']
    
    # Calculate daily ranks (descending order - higher ROC gets better rank)
    ranks = combined_results.rank(axis=1, ascending=False)
    
    return combined_results, ranks

# Load the data
funds_df = load_fund_list()
roc_days_df, multiply_factors_df = load_roc_params()

# Sidebar for input parameters
st.sidebar.markdown('## Input Parameters')

# Date selection
today = datetime.today()
min_date = today - relativedelta(years=10)
max_date = today

# Define default end date as today and start date as 3 years ago
default_end_date = today
default_start_date = today - relativedelta(years=3)

# Date range input
start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end_date,
    min_value=min_date,
    max_value=max_date
)

# Ensure dates have at least 3 years gap
if (end_date - start_date).days < 3*365:
    st.sidebar.error("Please select dates with at least 3 years gap.")
    valid_dates = False
else:
    valid_dates = True

# Fund selection
st.sidebar.markdown('### Fund Selection')
num_funds = st.sidebar.slider('Number of funds', 2, len(funds_df), 5)

selected_funds = []
selected_fund_names = []

for i in range(num_funds):
    remaining_funds = [fund for fund in funds_df['Fund Name'] if fund not in selected_fund_names]
    
    if remaining_funds:
        selected_fund = st.sidebar.selectbox(
            f'Fund {i+1}',
            remaining_funds,
            key=f'fund_{i}'
        )
        selected_funds.append(funds_df[funds_df['Fund Name'] == selected_fund]['Ticker'].values[0])
        selected_fund_names.append(selected_fund)

# Base fund selection (Gold ETF is recommended)
base_fund = st.sidebar.selectbox(
    'Select Base Fund (Gold ETF recommended)',
    selected_fund_names
)
base_fund_ticker = funds_df[funds_df['Fund Name'] == base_fund]['Ticker'].values[0]

# ROC parameters
st.sidebar.markdown('### ROC Parameters')

# Display ROC days selection
roc_days = []
multiply_factors = []

for i in range(5):
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        day = st.selectbox(
            f'ROC Days {i+1}',
            roc_days_df['Days'],
            index=i,
            key=f'roc_day_{i}'
        )
        roc_days.append(day)
    
    with col2:
        factor = st.selectbox(
            f'Multiply Factor {i+1}',
            multiply_factors_df['Factors'],
            index=i,
            key=f'factor_{i}'
        )
        multiply_factors.append(factor)

# Button to calculate
calculate_button = st.sidebar.button('Calculate Rankings')

# Main content
# Check if minimum requirements are met
valid_input = (
    valid_dates and
    len(selected_funds) >= 2 and
    len(set(selected_funds)) == len(selected_funds)  # Check for duplicates
)

# If button is clicked and inputs are valid, calculate and show results
if calculate_button and valid_input:
    with st.spinner('Downloading fund data and calculating metrics...'):
        # Convert dates to string format for yfinance
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Download data
        fund_data = download_fund_data(selected_funds, start_date_str, end_date_str)
        
        # Map ticker symbols back to fund names for display
        ticker_to_name = {row['Ticker']: row['Fund Name'] for _, row in funds_df.iterrows()}
        display_columns = {ticker: ticker_to_name.get(ticker, ticker) for ticker in fund_data.columns}
        fund_data = fund_data.rename(columns=display_columns)
        
        # Calculate metrics
        sum_roc, ranks = calculate_metrics(fund_data, roc_days, multiply_factors)
        
        # Display results
        st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Rank History", "ROC Summary", "Raw Data"])
        
        with tab1:
            st.markdown("### Fund Rankings Over Time")
            st.markdown("Lower numbers indicate better performance (1 = highest ROC sum)")
            
            # Create a date selector to view rankings for a specific date
            available_dates = ranks.index
            selected_date = st.selectbox(
                "Select date to view rankings:",
                available_dates,
                index=len(available_dates)-1  # Default to most recent date
            )
            
            # Display rankings for the selected date
            if selected_date in ranks.index:
                daily_rank = ranks.loc[selected_date].sort_values()
                daily_rank_df = pd.DataFrame({
                    'Fund': daily_rank.index,
                    'Rank': daily_rank.values
                })
                
                # Highlight base fund
                is_base_fund = [ticker_to_name.get(base_fund_ticker, base_fund_ticker) in fund for fund in daily_rank_df['Fund']]
                
                # Create a horizontal bar chart
                fig = px.bar(
                    daily_rank_df,
                    x='Rank',
                    y='Fund',
                    orientation='h',
                    title=f'Fund Rankings for {selected_date.strftime("%Y-%m-%d")}',
                    color=is_base_fund,
                    color_discrete_map={True: '#FFD700', False: '#636EFA'},
                    labels={'color': 'Base Fund'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Display rank history chart
            st.markdown("### Ranking History")
            
            fig = go.Figure()
            for fund in ranks.columns:
                fig.add_trace(go.Scatter(
                    x=ranks.index,
                    y=ranks[fund],
                    mode='lines',
                    name=fund,
                    line=dict(width=2, dash='solid' if fund == ticker_to_name.get(base_fund_ticker, base_fund_ticker) else None)
                ))
            
            fig.update_layout(
                title="Fund Ranking History (Lower is Better)",
                xaxis_title="Date",
                yaxis_title="Rank",
                legend_title="Funds",
                yaxis=dict(autorange="reversed")  # Lower values (better ranks) shown at top
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### ROC Sum History")
            
            # Plot ROC sum history
            fig = go.Figure()
            for fund in sum_roc.columns:
                fig.add_trace(go.Scatter(
                    x=sum_roc.index,
                    y=sum_roc[fund],
                    mode='lines',
                    name=fund,
                    line=dict(width=2, dash='solid' if fund == ticker_to_name.get(base_fund_ticker, base_fund_ticker) else None)
                ))
            
            fig.update_layout(
                title="Sum of Weighted ROC Values",
                xaxis_title="Date",
                yaxis_title="Sum of Weighted ROC",
                legend_title="Funds"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current ROC values
            st.markdown("### Current ROC Values")
            latest_date = sum_roc.index[-1]
            latest_roc = sum_roc.loc[latest_date].sort_values(ascending=False)
            latest_roc_df = pd.DataFrame({
                'Fund': latest_roc.index,
                'ROC Sum': latest_roc.values
            })
            
            # Create bar chart for current ROC values
            fig = px.bar(
                latest_roc_df,
                x='Fund',
                y='ROC Sum',
                title=f'Sum of Weighted ROC Values on {latest_date.strftime("%Y-%m-%d")}',
                color='Fund',
                labels={'color': 'Fund'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Data Explorer")
            
            # Fund price history
            st.subheader("Fund Price History")
            fig = px.line(fund_data, title="Fund Price History")
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data tabs
            data_tab1, data_tab2 = st.tabs(["Price Data", "Rank Data"])
            
            with data_tab1:
                st.dataframe(fund_data)
                
            with data_tab2:
                st.dataframe(ranks)
            
            # Download options
            st.download_button(
                label="Download Price Data as CSV",
                data=fund_data.to_csv().encode('utf-8'),
                file_name='fund_price_data.csv',
                mime='text/csv',
            )
            
            st.download_button(
                label="Download Rank Data as CSV",
                data=ranks.to_csv().encode('utf-8'),
                file_name='fund_rank_data.csv',
                mime='text/csv',
            )

elif calculate_button and not valid_input:
    st.error("Please correct the input parameters and try again:")
    if not valid_dates:
        st.error("- Dates must have at least 3 years gap")
    if len(selected_funds) < 2:
        st.error("- Select at least 2 funds")
    if len(set(selected_funds)) != len(selected_funds):
        st.error("- Remove duplicate fund selections")
else:
    st.info("Configure your parameters in the sidebar and click 'Calculate Rankings' to begin the analysis.")
    
    # Display app instructions
    st.markdown("""
    ## How to Use This Tool
    
    1. **Select Date Range**: Choose start and end dates with at least 3 years of data
    2. **Select Funds**: Choose the number of index funds and select each fund from the dropdown
    3. **Define Base Fund**: Select one fund as your base fund (Gold ETF recommended)
    4. **Configure ROC Parameters**: Set the days for ROC calculation and their multiply factors
    5. **Calculate**: Click the 'Calculate Rankings' button to run the analysis
    
    ## Understanding the Results
    
    - **Rank History**: Shows how each fund's rank changes over time (lower is better)
    - **ROC Summary**: Displays the weighted ROC sum values over time
    - **Raw Data**: Access the underlying price and rank data for further analysis
    
    ## Methodology
    
    This tool calculates Rate of Change (ROC) for each selected fund over different time periods:
    
    1. ROC is calculated for 5 different day intervals (default: 5, 8, 13, 21, 34 days)
    2. Each ROC is multiplied by its corresponding factor (default: 34, 21, 13, 8, 5)
    3. The weighted ROCs are summed for each day
    4. Funds are ranked based on their sum of weighted ROCs (higher values get better ranks)
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Indian Index Fund Ranking Tool")
