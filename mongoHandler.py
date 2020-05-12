from abc import ABC
import pickle
import CustomExceptions
import keras
from keras.models import  model_from_json
from keras.models import load_model
import tensorflow as tf
import pymongo
from pymongo import MongoClient

class MongoHandlerBase(ABC):

    def save_model(self, key:str, model, optimizer, epoch, version:int, model_type:str=''):
        raise NotImplementedError


class MongoHandler(MongoHandlerBase):
    def __init__(self):
        self.client = MongoClient("mongodb+srv://jack:lyons@cluster0-vl81d.mongodb.net/test?retryWrites=true&w=majority")
        self.db = self.client["test"]
        self.col = self.db["model"]

    def save_model(self, key:str, modelDict, version):
        self.col.insert_one({"_id":version, "modelDict": modelDict, "key":key})
