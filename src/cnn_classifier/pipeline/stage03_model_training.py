from cnn_classifier.config.configuration import ModelTrainingConfigurationManager
from cnn_classifier.components.model_training import ModelTraining
from src import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass
    

    def main(self):
        model_prep_config=ModelTrainingConfigurationManager()
        model=ModelTraining(config=model_prep_config)
        model.train_valid_generator()
        model.train()


STAGE_NAME="BASE MODEL TRAINING STAGE"

if __name__=="main":
    try:
       logger.info(f"{STAGE_NAME} STARTED ")
       obj=ModelTrainingPipeline()
       obj.main()
       logger.info(f"{STAGE_NAME} COMPLETED ")
    except Exception as e:
        raise e
