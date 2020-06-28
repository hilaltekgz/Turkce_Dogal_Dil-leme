import os
import numpy as np
import pandas as pd
from tweepy import Stream, API
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import sqlite3
import json
from unidecode import unidecode
from textblob import TextBlob
import re
from threading import Thread, Lock, Event
from tweetprocessing import tweetprocessing
from keys import KeyConf
from subprocess import call

#
def thread_second():
    call(["python", "dashTweet.py"])

def main():

    ckey = KeyConf.ckey
    csecret = KeyConf.csecret
    atoken = KeyConf.atoken
    asecret = KeyConf.asecret
    dbName = KeyConf.dbName
    tableName = KeyConf.tableName
    keyWords= KeyConf.keyWords

    tweetsProcessing = tweetprocessing(ckey=ckey, csecret=csecret, atoken=atoken,
                                        asecret=asecret, dbName=dbName, tableName=tableName, keyWords=keyWords)


    tweetsProcessing.createTwitterDB()
    tweetsProcessing.run()
    print("Twitter bağlantısı başarılı!")

    processThread = Thread(target=thread_second)
    processThread.daemon = True
    processThread.start()
    print("DashBoard Oluşturuldu")



if __name__ == '__main__':
    main()
