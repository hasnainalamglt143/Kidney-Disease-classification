from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion   import DataIngestion
from src import logger

STAGE_NAME="<------💥DATA INGESTION STAGE----->"

class DataIngestionPipeine:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        cnf=DataIngestion(config=data_ingestion_config)
        cnf.download_file()
        cnf.extract_zip_file()



if __name__=="main":
    try:
      logger.info(f"{STAGE_NAME} has started")
      obj=DataIngestionPipeine()
      obj.main()
      logger.info(f"{STAGE_NAME} has completed")


    except Exception as e:
        raise e


