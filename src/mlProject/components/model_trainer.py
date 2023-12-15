import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from mlProject import logger
from mlProject.entity.config_entity import *


class ModelTrainer:
    # def __init__(self, config: ModelTrainerConfig):
    def __init__(self, config):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        model_name = self.config.model_name

        # add if using different pretrain model
        if model_name == "ElasticNet":
            lr = ElasticNet(
                alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42
            )
            lr.fit(train_x, train_y)

            joblib.dump(
                lr, os.path.join(self.config.root_dir, self.config.model_output)
            )

        elif model_name == "RandomForestRegression":
            regressor = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
            )
            regressor.fit(train_x, train_y)

            joblib.dump(
                regressor, os.path.join(self.config.root_dir, self.config.model_output)
            )

        elif model_name == "KNNRegression":
            knn_regressor = KNeighborsRegressor(n_neighbors=self.config.n_neighbors)
            knn_regressor.fit(train_x, train_y)

            joblib.dump(
                knn_regressor,
                os.path.join(self.config.root_dir, self.config.model_output),
            )

        elif model_name == "XGboostRegression":
            xgb_regressor = XGBRegressor(
                objective=self.config.objective,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                colsample_bylevel=self.config.colsample_bylevel,
                gamma=self.config.gamma,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                scale_pos_weight=self.config.scale_pos_weight,
            )
            xgb_regressor.fit(train_x, train_y)

            joblib.dump(
                xgb_regressor,
                os.path.join(self.config.root_dir, self.config.model_output),
            )
