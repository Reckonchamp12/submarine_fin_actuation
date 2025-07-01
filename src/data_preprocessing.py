# Feature Engineering for Submarine Fin Control (data not included)

import pandas as pd

def process_features(df):
    df = df.dropna()
    df['angle_diff'] = df['fin_angle'] - df['pitch']
    df['velocity_depth_ratio'] = df['flow_velocity'] / (df['depth'] + 1e-3)
    return df
