from cnn_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeine
from cnn_classifier.pipeline.stage01_model_prep import BaseModelPrepPipeline
from cnn_classifier.pipeline.stage03_model_training import ModelTrainingPipeline
from src import logger


STAGE_NAME="data ingestion stage"

try:
    logger.info(f"💥{STAGE_NAME} has started💥")
    obj=DataIngestionPipeine()
    obj.main()
    logger.info(f"{STAGE_NAME} has completed")


except Exception as e:
        
        raise e



STAGE_NAME2="BASE MODEL PREPARATION STAGE"
try:
    logger.info(f"💥 {STAGE_NAME2} STARTED 💥 ")
    obj=BaseModelPrepPipeline()
    obj.main()
    logger.info(f"💥 {STAGE_NAME2} COMPLETED 💥 ")
    
except Exception as e:
          raise e



STAGE_NAME="BASE MODEL TRAINING STAGE"
try:
       logger.info(f"💥 {STAGE_NAME} STARTED 💥 ")
       obj=ModelTrainingPipeline()
       obj.main()
       logger.info(f"💥 {STAGE_NAME} COMPLETED 💥 ")
except Exception as e:
        raise e