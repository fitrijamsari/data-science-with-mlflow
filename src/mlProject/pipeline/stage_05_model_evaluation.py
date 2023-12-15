from pathlib import Path

from mlProject import logger
from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL EVALUATION STAGE"


class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        # Check if model.joblib exist in the 04_model_trainer folder
        model_folder = ConfigurationManager().get_model_trainer_config().root_dir
        model_output = ConfigurationManager().get_model_trainer_config().model_output
        model_path = Path(model_folder + "/" + model_output)

        if model_path.exists():
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_config.log_into_mlflow()
        else:
            logger.error("Model not found: model.joblib is missing")


if __name__ == "__main__":
    try:
        logger.info(f">>>> STARTED: {STAGE_NAME} <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> COMPLETED: {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
