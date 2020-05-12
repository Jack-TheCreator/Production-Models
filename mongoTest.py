import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb+srv://jack:lyons@cluster0-vl81d.mongodb.net/test?retryWrites=true&w=majority")
db = client["test"]

col = db['model']

col.insert_one({})