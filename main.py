import yfinance as yf
from DBHandler import DBHandler
import pandas as pd
import numpy as np
import json
from redis import Redis
import CustomExceptions

redis = Redis()

def mainTest():
    tickerSymbol = 'MSFT'
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2020-1-25')

    handler = DBHandler(redis)

    print(handler.save('test', tickerDf))
    handler.backup_to_mongo('test')


mainTest()