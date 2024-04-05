class CodeMemory:
    def __init__(self):
        self.code_files = []
    
    def add(self, date, file_path, code_str):
        self.code_files.append({"Date": date, "File_path": file_path, "Code": code_str})
    
    def remove(self, file_path):
        self.code_files = [code_file for code_file in self.code_files if code_file["File_path"] != file_path]
    
    def list_code_files(self):
        output = ""
        for i, code_file in enumerate(self.code_files, start=1):
            output += f"---{i}st file---\n"
            output += f"Date: {code_file['Date']}\n"
            output += f"File_path: {code_file['File_path']}\n"
            output += f"Code:\n{code_file['Code']}\n\n"
        return output.strip()

# Test the class
memory = CodeMemory()


# Test the class
memory = CodeMemory()
memory.add("16-12-2023", "/home/khudi/Desktop/autogen_ds/code/xgb_forecast_daily_level.py", """
    import pandas as pd
from xgboost import XGBRegressor
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def xgb_forecasting_daily_level(datapath, date_start, date_end):
    \"""
    Time series forecasting using XGBoost model.

    Args:
    datapath (str): Path to the CSV file containing the data.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing the predicted values.
    \"""

    metrics = \{\}
    # Read data
    data = pd.read_csv(datapath,   usecols=['Date', "Sales"])
    data = data[['Sales', "Date"]]


    # Preprocess data (if required)
    # Example: Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data["Sales"] = data["Sales"].astype(float)

    data.sort_values('Date', inplace=True)
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
    X = data[numerical_features]

    xgb_regressor = XGBRegressor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_regressor)])

    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics['mse'] = mean_squared_error(y_test,predictions)


    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    date_dataframe = pd.DataFrame({'Date': date_range})
    results = agent_inference(pipeline,date_dataframe)
    metrics['results'] = results
    metrics['average_forcast'] = np.average(results)
    metrics['total_forecast'] = np.sum(results)
    print(metrics)
    return metrics

def agent_inference(pipeline,date_dataframe):
    df = date_dataframe.copy()
    print(df.head())
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    X = df.copy()
    predictions = pipeline.predict(X)
    return predictions




if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python <your_file_name.py>  <data_path> <date_start> <date_end>")
        sys.exit(1)
    
    # Extract the file name and data path from command line arguments
    data_path = sys.argv[1]
    date_start = sys.argv[2]
    date_end = sys.argv[3]

    print("here ", data_path, date_start, date_end)

    # Execute the forecasting function
    predicted_values = xgb_forecasting_daily_level(data_path, date_start, date_end)
    print(predicted_values)
""")


memory.add("16-12-2023", "/home/khudi/Desktop/autogen_ds/code/xgb_forecast_daily_level_store_level.py", """
iimport pandas as pd
from xgboost import XGBRegressor
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def xgb_forecasting_store_level(datapath, date_start, date_end, is_store_level):
    \"""
    Time series forecasting using XGBoost model at store level breakdown.

    Args:
    datapath (str): Path to the CSV file containing the data.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.
    is_store_level (bool): Boolean indicating whether to perform store-level forecasting.

    Returns:
    dict: Dictionary containing metrics and sales predictions for each store.
    \"""
    metrics = \{\}
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
    store_predictions = \{\}

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
    \"""
    Make predictions using the trained pipeline for the given date range.

    Args:
    pipeline: Trained pipeline containing the forecasting model.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.
    is_store_level (bool): Boolean indicating whether to perform store-level forecasting.

    Returns:
    numpy.ndarray: Predicted sales values.
    \"""
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


""")

def list_code_files():
    """
    Lists all code files stored in the memory.

    This function accesses the global `memory` instance of the `CodeMemory` class and
    invokes its `list_code_files` method to retrieve a string representation of all code files
    currently stored. Each code file's details are formatted and returned in a single string,
    with each file's information (date, file path, and code) presented in a structured format.

    Returns:
        str: A formatted string containing the details of all code files in the memory,
             including their dates, file paths, and code snippets.
    """
    return memory.list_code_files()

