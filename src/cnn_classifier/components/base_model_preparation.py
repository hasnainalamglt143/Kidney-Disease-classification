import os
from pathlib import Path
# from urllib.request import requests
from cnn_classifier.utils.common import save_bin
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from cnn_classifier.config.configuration import BaseModelPrepConfigManager


class PrepareBaseModel:
    def __init__(self,config:BaseModelPrepConfigManager):
        self.config=config

    def get_base_model(self):
        model=VGG16(
            input_shape=self.config.params_img_shape,include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        self.save_model(path=self.config.base_model_path,model=model)
        self.model=model

    @staticmethod
    def _prepare_full_model(model:Sequential, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
                
        full_model =Sequential([
                model,                         # your pre-trained VGG16 backbone
                Flatten(),                     # flatten the output
                Dense(classes, activation="softmax")  # final prediction layer
            ])
        full_model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
        full_model.summary()
        full_model.summary()
        return full_model
    


    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path:Path,model:Sequential):
                 model.save(path)