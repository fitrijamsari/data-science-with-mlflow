from mlProject import logger
from mlProject.components.data_ingestion import DataIngestion
from mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "DATA INGESTION STAGE"


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
