from cnn_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeine
from cnn_classifier.pipeline.stage01_model_prep import BaseModelPrepPipeline
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

