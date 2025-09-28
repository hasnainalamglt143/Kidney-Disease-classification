from cnn_classifier.config.configuration import BaseModelPrepConfigManager
from cnn_classifier.components.base_model_preparation import PrepareBaseModel
from src import logger

class BaseModelPrepPipeline:
    def __init__(self):
        pass
    

    def main(self):
        model_prep_config=BaseModelPrepConfigManager()
        cnf=model_prep_config.get_model_preparation_config()
        model=PrepareBaseModel(config=cnf)
        model.get_base_model()
        model.update_base_model()


STAGE_NAME="BASE MODEL PREPARATION STAGE"

if __name__=="main":
    try:
       logger.info(f"💥 {STAGE_NAME} STARTED 💥 ")
       obj=BaseModelPrepPipeline()
       obj.main()
       logger.info(f"💥 {STAGE_NAME} COMPLETED 💥 ")
    except Exception as e:
        raise e
