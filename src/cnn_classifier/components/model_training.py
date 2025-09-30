from pathlib import Path
import tensorflow as tf
from cnn_classifier.config.configuration import ModelTrainingConfigurationManager

class ModelTraining:
    def __init__(self,config:ModelTrainingConfigurationManager):
        self.config=config.get_model_config()
        

    
    def get_base_model(self):
        model=tf.keras.models.load_model(self.config.base_model_path)
        return model


    def train_valid_generator(self):
         self.model=self.get_base_model()
         datagenerator_kwargs = dict( rescale = 1./255, validation_split=0.20 ) 
         dataflow_kwargs = dict( target_size=self.config.params_img_shape[:-1], batch_size=self.config.params_batch_size, interpolation="bilinear" ) 
         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
              **datagenerator_kwargs )
         
         self.valid_generator = valid_datagenerator.flow_from_directory( 
             directory=self.config.dataset_path, 
             subset="validation", 
             shuffle=False, **dataflow_kwargs ) 
         
         if self.config.params_is_augmented: 
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
                rotation_range=40,
                  horizontal_flip=True,
                    width_shift_range=0.2, 
                    height_shift_range=0.2, 
                    shear_range=0.2, 
                    zoom_range=0.2, 
                    **datagenerator_kwargs ) 
         else: 
            train_datagenerator = valid_datagenerator 
         self.train_generator = train_datagenerator.flow_from_directory( 
                directory=self.config.dataset_path,shuffle=True,
                    **dataflow_kwargs )
             
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            # Method 1: Standard save
            model.save(path)
        except Exception as e:
            print(f"Standard save failed: {e}")
            print("Trying alternative save method...")
            raise e

            
            
        
    def train(self):
            self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size 
            self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size      
            self.model.fit( self.train_generator, epochs=self.config.params_epochs, validation_data=self.valid_generator ) 
            self.save_model( path=self.config.trained_model_path, model=self.model )



        