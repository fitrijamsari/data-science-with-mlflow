from mlProject import logger
from mlProject.components.data_validation import DataValidation
from mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "DATA VALIDATION STAGE"


class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()


if __name__ == "__main__":
    try:
        logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
