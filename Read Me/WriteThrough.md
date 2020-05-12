# Inserting trained models into a cache databases (Redis) using pymongo as a backup

## Why you need a backup?

Inserting trained models into a cache database allows for quick loading of these trained models to be put into production. However storing models in cache database is that if this server crashes for whatever reason, you lose these trained models. Therefore, these models must also be stored on a disk to ensure in the event of a crash, you will not lose these trained models, yet you still get the advantage of fast retrevial time.

## Write through 

Write through is when you store the data in both the cache and disk simultaneously.


```python
def save_model(self, key:str, model, optimizer, epoch, version:int, model_type:str=''):
        modelDict = self.dictify(model, optimizer, epoch, model_type)
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
```

While this does have an increase in latency when saving models, it ensures models are always backedup, and has no impact on the speed at which you can load a model.
