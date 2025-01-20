import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime 

class OptunaLGBMRegressor:
    def __init__(self, n_estimators: int, learning_rate: float = 0.01, metric: str = 'rmse', cat_columns: str = 'auto', seed: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "objective": "regression",
            "verbosity": -1,
            "metric": metric,
            "learning_rate": learning_rate,
            "boosting_type": 'gbdt',
            "random_state": seed
        }
        self.cat_columns = cat_columns
        self.model = None
        self.features = None
        self.is_fitted_ = False
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        end_date = datetime.now().strftime('%Y-%m-%d')
        btc_data = cryptocompare.get_historical_price_day('BTC', currency='USD', toTs=datetime.now())
        #print(btc_data)
        """ Preprocessing for CryptoCompare data """
        df = pd.DataFrame(btc_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['timestamp'] = pd.to_datetime(df['Date'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Drop columns that aren't useful for the model
        df = df.drop(['timestamp', 'time'], axis=1)
        df = df.dropna()
        
        return df
    
    def _to_datasets(self, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray) -> (lgb.Dataset, lgb.Dataset):
        self.features = list(x_train.columns)
        dtrain = lgb.Dataset(x_train[self.features], label=y_train, categorical_feature=self.cat_columns)
        dval = lgb.Dataset(x_val[self.features], label=y_val, categorical_feature=self.cat_columns)
        return dtrain, dval
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        X_train = self._preprocess(X_train)
        X_val = self._preprocess(X_val)
        dtrain, dval = self._to_datasets(X_train, y_train, X_val, y_val)
        self.model = lgb.train(self.params, dtrain, valid_sets=[dtrain, dval])
        self.is_fitted_ = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted_, 'Model is not fitted!'
        X_test = self._preprocess(X_test)
        return self.model.predict(X_test[self.features], num_iteration=self.model.best_iteration)
