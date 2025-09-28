from cnn_classifier.constants import  *
from cnn_classifier.utils.common import read_yaml,create_directories
from cnn_classifier.entity.config_entity import DataIngestionConfig,BaseModelPrepConfigEntity

class ConfigurationManager():
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH):
        self.params=read_yaml(params_filepath)
        self.config=read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config=DataIngestionConfig(root_dir=config.root_dir,src_url=config.SRC_URL,local_data_file=config.local_data_file,unzip_dir=config.unzip_dir)
        return data_ingestion_config



class BaseModelPrepConfigManager():
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH):
        self.params=read_yaml(params_filepath)
        self.config=read_yaml(config_filepath)

        # create_directories([self.config.artifacts_root])
    
    def get_model_preparation_config(self)->BaseModelPrepConfigEntity:
        config=self.config.prepare_base_model
        create_directories([config.root_dir])

        model_preparation_config=BaseModelPrepConfigEntity(root_dir=config.root_dir,base_model_path=Path(config.base_model_path),updated_base_model_path=Path(config.updated_base_model_path),params_include_top=self.params.INCLUDE_TOP,params_classes=self.params.CLASSES,   params_batch_size= self.params.BATCH_SIZE,
         params_img_shape=self.params.IMAGE_SHAPE ,params_learning_rate= self.params.LEARNING_RATE, params_weights=self.params.WEIGHTS,params_epochs=self.params.EPOCHS)

        return model_preparation_config
