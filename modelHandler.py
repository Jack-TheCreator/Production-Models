from abc import ABC
from redis import Redis
import pickle
import CustomExceptions
import keras
from keras.models import  model_from_json
from keras.models import load_model
import tensorflow as tf
import orjson

class ModelHandlerBase(ABC):
    def save_model(self, key:str, model, optimizer, version:int, model_type:str):
        raise NotImplementedError

    def dictify(self, model, optimizer, model_type:str) -> dict:
        raise NotImplementedError

    def load_latest_model(self, key:str):
        raise  NotImplementedError

    def dedictify(self, modelDict:dict):
        raise NotImplementedError

class ModelHandler(ModelHandlerBase):
    def __init__(self, redisConnection):
        self.redis = redisConnection

    def save_model(self, key:str, model, optimizer, epoch, version:int, model_type:str=''):
        modelDict = self.dictify(model, optimizer, epoch, model_type)
        pickled = pickle.dumps(modelDict)
        scored = {pickled: version}
        try:
            self.redis.zadd(key, scored)
            return True
        except:
            raise CustomExceptions.saveError("Error Saving Model")

    def dictify(self, model, optimizer, epoch, model_type:str) -> dict:
        modelDict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_type':model_type
        }

        return modelDict

    def dedictify(self, modelDict:dict):
        epoch = modelDict['epoch']
        model = modelDict['state_dict']
        optimizer = modelDict['optimizer']


        return epoch, model, optimizer


    def load_latest_model(self, key:str):
        pickled = self.redis.zrange(key, -1, -1)
        modelDict = pickle.loads(pickled[0])
        return(self.dedictify(modelDict))


