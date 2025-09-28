from cnn_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeine
from src import logger


STAGE_NAME="data ingestion stage"

try:
    logger.info(f"💥{STAGE_NAME} has started💥")
    obj=DataIngestionPipeine()
    obj.main()
    logger.info(f"{STAGE_NAME} has completed")


except Exception as e:
        raise e


