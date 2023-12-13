from pathlib import Path

from mlProject import logger
from mlProject.components.model_trainer import ModelTrainer
from mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training Stage"


class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        # Check if train.csv and test.csv exist in the 03_data_validation folder
        dataset_folder = (
            ConfigurationManager().get_data_transformation_config().root_dir
        )
        train_file = Path(dataset_folder + "/train.csv")
        test_file = Path(dataset_folder + "/test.csv")

        if train_file.exists() and test_file.exists():
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        else:
            logger.error("Files not found: train.csv and/or test.csv missing")


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
