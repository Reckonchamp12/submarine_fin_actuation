from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)
    rf_mse = mean_squared_error(y_test, rf.predict(X_test))

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    xgb_mse = mean_squared_error(y_test, xgb.predict(X_test))

    return {'RF_MSE': rf_mse, 'XGB_MSE': xgb_mse}
