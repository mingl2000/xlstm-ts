# Utility script for training a denoised xLSTM-TS model on all available data
# and producing a next-day directional forecast (without probability output).

import datetime
import os
import sys
import random
import numpy as np
import pandas as pd
# Ensure deterministic CuBLAS when deterministic algorithms are enabled
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

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
FREQ = '1d'
FILE_NAME = '000001SS_daily'
DATA_PATH = os.path.join('data', 'datasets', f'{FILE_NAME}.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('xLSTM-TS requires a CUDA-enabled GPU for training and inference.')

# Global seed for reproducibility
SEED = 42

def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior where possible
    # xLSTM uses CUDA ops like cumsum that lack deterministic kernels.
    # Prefer warn-only to keep runs comparable without hard failures.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch without warn_only: fall back to best-effort determinism
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

# -------------------------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------------------------

def load_price_data():
    """Load, filter, and index the historical price data."""
    print(f'Loading data from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, header=0, index_col=0, skiprows=[1, 2])
    df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))

    start_dt = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df = df[df.index >= start_dt]
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
# Prediction logic (without probability)
# -------------------------------------------------------------------------------------------

def predict_next_day_direction(
    xlstm_stack,
    input_projection,
    output_projection,
    close_scaled,
    scaler,
    last_close,
):
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
        xlstm_stack,
        input_projection,
        output_projection,
        close_scaled,
        scaler,
        last_close,
    )

    next_session = df.index[-1] + datetime.timedelta(days=1)
    print('-------------------------------------------------------')
    print(f'Latest available close ({df.index[-1].date()}): {last_close:.2f}')
    print(f'Predicted close for {next_session.date()}: {predicted_close:.2f}')
    print(f'Expected direction for next session: {direction}')


if __name__ == '__main__':
    main()
