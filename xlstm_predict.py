# Utility script for training a denoised xLSTM-TS model on all available data
# and producing a next-day directional forecast.

import datetime
import os
import sys
import torch
import pandas as pd

# Ensure src/ is on the Python path
CURRENT_DIR = os.getcwd()
SRC_PATH = os.path.join(CURRENT_DIR, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from ml.constants import SEQ_LENGTH_XLSTM
from ml.data.preprocessing import process_dates, wavelet_denoising
from ml.models.xlstm_ts.preprocessing import (
    create_sequences,
    inverse_normalise_data_xlstm,
    normalise_data_xlstm,
)
from ml.models.xlstm_ts.training import evaluate_model, train_model
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model

# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------

TICKER = '^GSPC'
STOCK = 'SS 000001'
START_DATE = '2000-01-01'
END_DATE = '2025-10-07'
FREQ = '1d'
FILE_NAME = '000001SS_daily'
DATA_PATH = os.path.join('data', 'datasets', f'{FILE_NAME}.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('xLSTM-TS requires a CUDA-enabled GPU for training and inference.')

# -------------------------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------------------------

def load_price_data():
    """Load, filter, and index the historical price data."""
    print(f'Loading data from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, header=0, index_col=0, skiprows=[1, 2])
    df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))

    start_dt = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(END_DATE, '%Y-%m-%d')
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df = process_dates(df)

    if df.empty:
        raise ValueError('No data left after applying the requested date range.')

    return df


def apply_wavelet_denoising(df):
    """Add denoised close values to the dataframe."""
    df['Close_denoised'] = wavelet_denoising(df['Close'])
    df['Noise'] = df['Close'] - df['Close_denoised']
    return df


def prepare_training_tensors(df):
    """Create CUDA tensors ready for training plus the scaler and scaled array."""
    close_scaled, scaler = normalise_data_xlstm(df['Close_denoised'].values)
    seq_x, seq_y, _ = create_sequences(close_scaled, df.index)

    if seq_x.shape[0] < 2:
        raise ValueError('Not enough samples to build training sequences.')

    train_x = seq_x.to(DEVICE)
    train_y = seq_y.to(DEVICE)
    val_x = train_x.clone().detach()
    val_y = train_y.clone().detach()

    return train_x, train_y, val_x, val_y, close_scaled, scaler


# -------------------------------------------------------------------------------------------
# Prediction logic
# -------------------------------------------------------------------------------------------

def predict_next_day_direction(xlstm_stack, input_projection, output_projection, close_scaled, scaler, last_close):
    """
    Predict the next close value and map it to a direction relative to the latest close.
    Returns (predicted_close, direction_label).
    """
    if len(close_scaled) < SEQ_LENGTH_XLSTM:
        raise ValueError('Need at least SEQ_LENGTH_XLSTM observations to forecast the next close.')

    latest_sequence = close_scaled[-SEQ_LENGTH_XLSTM:]
    latest_tensor = torch.from_numpy(latest_sequence).float().unsqueeze(0).to(DEVICE)

    next_close_scaled = evaluate_model(xlstm_stack, input_projection, output_projection, latest_tensor)
    next_close = inverse_normalise_data_xlstm(next_close_scaled.squeeze(), scaler).item()
    direction = 'Up' if next_close > last_close else 'Down'

    return next_close, direction


def main():
    df = load_price_data()
    df = apply_wavelet_denoising(df)

    train_x, train_y, val_x, val_y, close_scaled, scaler = prepare_training_tensors(df)

    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)
    xlstm_stack, input_projection, output_projection = train_model(
        xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y
    )

    last_close = df['Close'].iloc[-1]
    predicted_close, direction = predict_next_day_direction(
        xlstm_stack, input_projection, output_projection, close_scaled, scaler, last_close
    )

    next_session = df.index[-1] + datetime.timedelta(days=1)
    print('-------------------------------------------------------')
    print(f'Latest available close ({df.index[-1].date()}): {last_close:.2f}')
    print(f'Predicted close for {next_session.date()}: {predicted_close:.2f}')
    print(f'Expected direction for next session: {direction}')


if __name__ == '__main__':
    main()
