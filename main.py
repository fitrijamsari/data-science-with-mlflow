from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "DATA INGESTION STAGE"
try:
    logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "DATA VALIDATION STAGE"
try:
    logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "DATA TRANSFORMATION STAGE"
try:
    logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "MODEL TRAINING STAGE"
try:
    logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "MODEL EVALUATION STAGE"
try:
    logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
