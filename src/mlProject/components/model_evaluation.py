import json
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlProject import logger
from mlProject.constants import *
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import read_yaml


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, config_filepath=CONFIG_FILE_PATH):
        self.config = config
        self.model = read_yaml(config_filepath)

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        model_name = self.model.model_trainer.model_name

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            pred = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, pred)

            # Saving metrics as local json file
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            metric_file_name_json = Path(self.config.metric_file_name)

            # Save the scores dictionary as JSON
            with open(metric_file_name_json, "w") as json_file:
                json.dump(scores, json_file)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                # mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name=f"{model_name}"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
