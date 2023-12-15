from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    test_split_percentage: float


# @dataclass(frozen=True)
# class ModelTrainerConfig:
#     root_dir: Path
#     train_data_path: Path
#     test_data_path: Path
#     model_output: str
#     alpha: float
#     l1_ratio: float
#     target_column: str


@dataclass(frozen=True)
class ModelTrainerElasticNetConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_output: str
    alpha: float
    l1_ratio: float
    target_column: str


@dataclass(frozen=True)
class ModelTrainerRandomForestRegressionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_output: str
    n_estimators: int
    random_state: int
    target_column: str


@dataclass(frozen=True)
class ModelTrainerKNNRegressionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_output: str
    n_neighbors: int
    target_column: str


@dataclass(frozen=True)
class ModelTrainerXGboostRegressionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_output: str
    objective: str
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    gamma: int
    reg_alpha: int
    reg_lambda: int
    scale_pos_weight: int
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    all_params: dict
    target_column: str
    mlflow_uri: str
