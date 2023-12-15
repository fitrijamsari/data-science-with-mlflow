import os

from dotenv import load_dotenv

from mlProject import logger
from mlProject.constants import *
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerElasticNetConfig,
    ModelTrainerKNNRegressionConfig,
    ModelTrainerRandomForestRegressionConfig,
    ModelTrainerXGboostRegressionConfig,
)
from mlProject.utils.common import create_directories, read_yaml

load_dotenv()

# Access environment variables
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            test_split_percentage=config.test_split_percentage,
        )

        return data_transformation_config

    # def get_model_trainer_config(self) -> ModelTrainerConfig:
    #     config = self.config.model_trainer
    #     params = self.params.ElasticNet
    #     schema = self.schema.TARGET_COLUMN

    #     create_directories([config.root_dir])

    #     model_trainer_config = ModelTrainerConfig(
    #         root_dir=config.root_dir,
    #         train_data_path=config.train_data_path,
    #         test_data_path=config.test_data_path,
    #         model_output=config.model_output,
    #         alpha=params.alpha,
    #         l1_ratio=params.l1_ratio,
    #         target_column=schema.name,
    #     )

    #     return model_trainer_config

    def get_model_trainer_config(self):
        config = self.config.model_trainer
        model_name = config.model_name

        def get_model_params(model_name: str):
            if model_name in self.params:
                return self.params[model_name]
            else:
                logger.error("Model name not exist in params.yaml")
                return None  # Handle the case where model_name is not found in params

        params = get_model_params(model_name)
        # params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        if model_name == "ElasticNet":
            model_trainer_config = ModelTrainerElasticNetConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                model_output=config.model_output,
                alpha=params.alpha,
                l1_ratio=params.l1_ratio,
                target_column=schema.name,
            )
        elif model_name == "RandomForestRegression":
            model_trainer_config = ModelTrainerRandomForestRegressionConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                model_output=config.model_output,
                n_estimators=params.n_estimators,
                random_state=params.random_state,
                target_column=schema.name,
            )

        elif model_name == "KNNRegression":
            model_trainer_config = ModelTrainerKNNRegressionConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                model_output=config.model_output,
                n_neighbors=params.n_neighbors,
                target_column=schema.name,
            )

        elif model_name == "XGboostRegression":
            model_trainer_config = ModelTrainerXGboostRegressionConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                model_output=config.model_output,
                objective=params.objective,
                learning_rate=params.learning_rate,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_child_weight=params.min_child_weight,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                colsample_bylevel=params.colsample_bylevel,
                gamma=params.gamma,
                reg_alpha=params.reg_alpha,
                reg_lambda=params.reg_lambda,
                scale_pos_weight=params.scale_pos_weight,
                target_column=schema.name,
            )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        model_name = self.config.model_trainer.model_name

        def get_model_params(model_name: str):
            if model_name in self.params:
                return self.params[model_name]
            else:
                logger.error("Model name not exist in params.yaml")

        params = get_model_params(model_name)
        # params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            all_params=params,
            target_column=schema.name,
            mlflow_uri=mlflow_uri,
        )

        return model_evaluation_config
