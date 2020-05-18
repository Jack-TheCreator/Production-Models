from abc import ABC
from redis import Redis
import pickle
import CustomExceptions
import keras
from keras.models import  model_from_json
from keras.models import load_model
import tensorflow as tf
import orjson
from mongoHandler import MongoHandler
import pymongo

class ModelHandlerBase(ABC):
    def save_model(self, key:str, model, optimizer, version:int, features:dict):
        raise NotImplementedError

    def dictify(self, model, optimizer, features:dict) -> dict:
        raise NotImplementedError

    def load_latest_model(self, key:str):
        raise  NotImplementedError

    def dedictify(self, modelDict:dict):
        raise NotImplementedError

class ModelHandler(ModelHandlerBase):
    def __init__(self, redisConnection):
        self.redis = redisConnection
        self.mongo = MongoHandler()

    def save_model(self, key:str, model, optimizer, version:int, features:dict=None):
        modelDict = self.dictify(model, optimizer, features)
        pickled = pickle.dumps(modelDict)
        scored = {pickled: version}
        try:
            self.redis.zadd(key, scored)
        except:
            raise CustomExceptions.saveError("Error Saving Model to Redis")
        try:
            self.mongo.save_model(key, pickled, version)
            return True
        except:
            raise CustomExceptions.saveError("Error Saving Model to Mongo")



    def dictify(self, model, optimizer, features) -> dict:
        modelDict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'features':features
        }

        return modelDict

    def dedictify(self, modelDict:dict):
        features = modelDict['features']
        model = modelDict['state_dict']
        optimizer = modelDict['optimizer']


        return model, optimizer, features


    def load_latest_model(self, key:str):
        pickled = self.redis.zrange(key, -1, -1)
        modelDict = pickle.loads(pickled[0])
        return(self.dedictify(modelDict))


