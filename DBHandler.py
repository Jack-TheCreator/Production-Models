from abc import ABC
from redis import Redis
from typing import List
import pandas as pd
import numpy as np
import orjson
import CustomExceptions
import datetime
import pymongo

class DBHandlerBase(ABC):
    def convert_dataframe_to_storable(data):
        raise NotImplementedError

    def save(key:str, storable:dict) -> bool:
        raise NotImplementedError

    def load(self, key:str):
        raise NotImplementedError

    #Helper Function
    def dictify(self, json) -> dict:
        raise NotImplementedError

    def convert_dict_to_dataframe(self, data:dict):
        raise NotImplementedError

    def backup_to_mongo(self, key:str):
        raise NotImplementedError


class DBHandler(DBHandlerBase):
    def __init__(self, redisConnection):
        self.redis = redisConnection


    def convert_dataframe_to_storable(self, data):
        data.index = data.index.astype(int)
        data = data.to_dict("index")
        storeable = {}
        for key, val in data.items():
            val = orjson.dumps(val)
            storeable.update({val:key})

        return storeable


    def save(self, key:str, data) -> bool:
        try:
            storable = self.convert_dataframe_to_storable(data)
            self.redis.zadd(key, storable)
            return True
        except:
            raise CustomExceptions.saveError("failure to save, check storable format")


    def load(self, key:str):
        json = self.redis.zrange(key, 0, -1, withscores=True)
        if(json==[]):
            raise CustomExceptions.loadError("no values loaded, check key value")
        else:
            loadDict = self.dictify(json)
            try:
                return(self.convert_dict_to_dataframe(loadDict))
            except:
                raise CustomExceptions.loadError("failure to convert to pd.Dataframe")


    def dictify(self, json) -> dict:
        loadDict = {}
        for item in json:
            try:
                dictItem = orjson.loads(item[0])
            except:
                raise CustomExceptions.loadError('json to dict load failure')

            loadDict.update({item[1]: dictItem})

        return(loadDict)

    def convert_dict_to_dataframe(self, data:dict):
        data = pd.DataFrame.from_dict(data, orient='index')
        data.index = pd.to_datetime(data.index, unit='ns')
        return(data)



