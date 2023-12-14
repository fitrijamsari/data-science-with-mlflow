import os

import pandas as pd
from sklearn.model_selection import train_test_split

from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # Note: We can add different data transformation techniques such as Scaler, PCA and all
    # We can perform all kinds of EDA in ML cycle here before passing this data to the model

    # Adding only train_test_spliting because this data is already cleaned up

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        test_size = self.config.test_split_percentage

        # Split the data into training and test sets. (0.7, 0.3) split.
        train, test = train_test_split(data, test_size=test_size, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("SUCCESS: Splited data into training and test sets")
        logger.info(f"Training Dataset: {train.shape}")
        logger.info(f"Test Dataset: {test.shape}")
