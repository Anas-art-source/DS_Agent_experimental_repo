import pandas as pd
from xgboost import XGBRegressor
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.dates as mdates
import streamlit as st
import matplotlib.pyplot as plt


def xgb_forecasting_store_level(datapath, date_start, date_end, is_store_level):
    """
    Time series forecasting using XGBoost model at store level breakdown.

    Args:
    datapath (str): Path to the CSV file containing the data.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.
    is_store_level (bool): Boolean indicating whether to perform store-level forecasting.

    Returns:
    dict: Dictionary containing metrics and sales predictions for each store.
    """
    metrics = {}
    # Read data
    data = pd.read_csv(datapath, usecols=['Store', 'Date', 'Sales'])

    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Store', 'Date'], inplace=True)

    # Feature engineering
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday

    numerical_features = ['Year', 'Month', 'Day', 'Weekday']
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
        ])

    # Initialize dictionary to store results for each store
    store_predictions = {}

    # Group by Store and apply forecasting model
    for store, store_data in data.groupby('Store'):
        X = store_data[numerical_features]
        y = store_data['Sales']

        # Define model pipeline
        xgb_regressor = XGBRegressor()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb_regressor)
        ])

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        predictions = agent_inference(pipeline, date_start, date_end, is_store_level)

        # Compute metrics
        if is_store_level:
            # If performing store-level forecasting, align y_test with predictions
            y_test = y_test.loc[X_test.index]
        # metrics['mse'] = mean_squared_error(y_test, predictions)
        metrics['average_forecast'] = np.average(predictions)
        metrics['total_forecast'] = np.sum(predictions)

        # Store predictions for this store
        store_predictions[store] = predictions

    return {'store_predictions': store_predictions}

def agent_inference(pipeline, date_start, date_end, is_store_level):
    """
    Make predictions using the trained pipeline for the given date range.

    Args:
    pipeline: Trained pipeline containing the forecasting model.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.
    is_store_level (bool): Boolean indicating whether to perform store-level forecasting.

    Returns:
    numpy.ndarray: Predicted sales values.
    """
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    date_dataframe = pd.DataFrame({'Date': date_range})

    # Preprocess date dataframe
    date_dataframe['Date'] = pd.to_datetime(date_dataframe['Date'])
    date_dataframe.sort_values('Date', inplace=True)
    date_dataframe['Year'] = date_dataframe['Date'].dt.year
    date_dataframe['Month'] = date_dataframe['Date'].dt.month
    date_dataframe['Day'] = date_dataframe['Date'].dt.day
    date_dataframe['Weekday'] = date_dataframe['Date'].dt.weekday

    # Make predictions
    predictions = pipeline.predict(date_dataframe.drop(columns=['Date']))
    plt.style.use('dark_background')
    plt.plot(df['Date'], predictions, color='cyan')  # Cyan stands out on a dark background
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.tick_params(colors='white', which='both')  # Change the colors of the tick marks to white
    plt.tight_layout()
    plot = st.pyplot(plt,clear_figure=False)
    return predictions,plot

    return predictions
if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python <your_file_name.py>  <data_path> <date_start> <date_end>")
        sys.exit(1)
    
    # Extract the file name and data path from command line arguments
    data_path = sys.argv[1]
    date_start = sys.argv[2]
    date_end = sys.argv[3]
    is_store_level = sys.argv[4]

    print("here ", data_path, date_start, date_end, is_store_level)

    # Execute the forecasting function
    result = xgb_forecasting_store_level(data_path, date_start, date_end, is_store_level)
    print(result)