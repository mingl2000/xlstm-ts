# Ticker (check in Yahoo Finance) and custom stock name
TICKER = '^GSPC' # S&P 500 index
STOCK = 'S&P 500'

# Date range (YYYY-MM-DD) and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d' # daily frequency

FILE_NAME = 'sp500_daily' # custom file name

# Train, validation, test split
TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'


import sys
import os

# Get the path of the current working directory
current_dir = os.getcwd()

# Construct the path to the 'src' directory in the current folder
src_path = os.path.join(current_dir, 'src')

# Add the 'src' directory to the Python path
if src_path not in sys.path:
    sys.path.append(src_path)

from ml.utils.imports import *
import datetime
import pandas as pd
# Read the CSV file and set the "Date" column as the index
file_path = os.path.join('data', 'datasets', FILE_NAME + '.csv')
print(f"Loading data from {file_path}")
df = pd.read_csv(file_path, header=0, index_col='Date')
# Convert the index to a datetime object
#df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z"))
df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], "%Y-%m-%d"))



from ml.data.preprocessing import wavelet_denoising, plot_wavelet_denoising
# Apply denoising
df['Close_denoised'] = wavelet_denoising(df['Close'])
df['Noise'] = df['Close'] - df['Close_denoised']

plot_wavelet_denoising(df, STOCK)
from ml.data.preprocessing import process_dates
df = process_dates(df)


from ml.models.darts.preprocessing import convert_to_ts_daily
series, series_denoised = convert_to_ts_daily(df)
series_denoised

from ml.models.darts.preprocessing import split_train_val_test_darts
TRAIN_END_DATE = datetime.datetime.strptime(TRAIN_END_DATE, '%Y-%m-%d')
VAL_END_DATE = datetime.datetime.strptime(VAL_END_DATE, '%Y-%m-%d')

train, val, test = split_train_val_test_darts(series, TRAIN_END_DATE, VAL_END_DATE)
train_denoised, val_denoised, test_denoised = split_train_val_test_darts(series_denoised, TRAIN_END_DATE, VAL_END_DATE)

train_denoised.plot(label="train");
val_denoised.plot(label="val");
test_denoised.plot(label="test");

from ml.models.darts.preprocessing import normalise_split_data_darts

train, val, test, scaler_darts = normalise_split_data_darts(train, val, test)
train_denoised, val_denoised, test_denoised, scaler_darts_denoised = normalise_split_data_darts(train_denoised, val_denoised, test_denoised)
from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm
close_scaled, scaler = normalise_data_xlstm(df['Close'].values)
close_scaled_denoised, scaler_denoised = normalise_data_xlstm(df['Close_denoised'].values)


from ml.models.xlstm_ts.preprocessing import create_sequences
X, y, dates = create_sequences(close_scaled, df.index)
X_denoised, y_denoised, _ = create_sequences(close_scaled_denoised, df.index)

from ml.models.xlstm_ts.preprocessing import split_train_val_test_xlstm
train_X, train_y, train_dates, val_X, val_y, val_dates, test_X, test_y, test_dates = split_train_val_test_xlstm(X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler, STOCK)
train_X_denoised, train_y_denoised, _, val_X_denoised, val_y_denoised, _, test_X_denoised, test_y_denoised, _ = split_train_val_test_xlstm(X_denoised, y_denoised, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_denoised, STOCK)


from ml.models.shared.directional_prediction import *
from ml.models.shared.metrics import *
from ml.models.shared.visualisation import *
from ml.models.darts.darts_models import *
from ml.models.darts.training import *
from ml.models.xlstm_ts.xlstm_ts_model import *
from ml.models.xlstm_ts.logic import *

model_name = 'xLSTM-TS'
results_df, metrics = run_xlstm_ts(train_X, train_y, val_X, val_y, test_X, test_y, scaler, STOCK, 'Original', test_dates)
results_denoised_df, metrics_denoised = run_xlstm_ts(train_X_denoised, train_y_denoised, val_X_denoised, val_y_denoised, test_X_denoised, test_y, scaler_denoised, STOCK, 'Denoised', test_dates, train_y, val_y, test_y)
