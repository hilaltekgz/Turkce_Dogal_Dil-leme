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
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
stopwords=stopwords.words("turkish")
from threading import Thread, Lock, Event
import time
from datetime import datetime
from keys import KeyConf
from tweetsSideCounter import TweetsSideCounter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report,recall_score
import time
import pickle
import csv
import joblib
import jpype as jp
from sklearn.model_selection import train_test_split
#from processed import processed
class listener(StreamListener):
    ZEMBEREK_PATH = r"C:/Program Files/Java/zemberek_jar/zemberek-full.jar"
    jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
    def __init__(self, TweetsProcessing, dbName, tableName):
        self.TweetsProcessing = TweetsProcessing
        self.tableName = tableName
        self.conn = sqlite3.connect(dbName,  check_same_thread=False)
        self.c = self.conn.cursor()

    def on_data(self, data):

            tweet = ""
            data=json.loads(data)
            data=[data]
            print("data",data)

            tweet = unidecode(data[0]["text"])
            tweet = tweet.replace("\'", "").replace("'", "''").replace('"', "''").replace("&", "&&").strip()
            tweet = re.sub(r'\s*https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
            tweet_vect=[tweet]
            print("tweet_vect",tweet_vect)

            if len(tweet)>0 and len(tweet_vect)>0:

                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(penalty='l2',
                                        fit_intercept=True,
                                        solver='lbfgs',
                                        max_iter=500)
                df = pd.read_csv(r"C:\Users\hlltk\OneDrive\Masaüstü\sosyalmedya_datasets\gerekligereksiz1.csv")
                df.dropna(axis="columns", how="any")
                df = df[pd.notnull(df['text'])]
                gerekli = df["text"].values.tolist()
                y = df["sentiment"].values.tolist()

                time_ms = data[0]['timestamp_ms']
                turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"
                # ‪data/data.tsv ----  ------  C:/Users/hlltk/OneDrive/Masaüstü/sosyalmedya_datasets/Turkce-Anlam-Analizi-master/T1urkce-Anlam-Analizi-master/data/data.tsv
                data = pd.read_csv(r'C:\Users\hlltk\OneDrive\Masaüstü\train_three.csv',
                                sep='|', encoding='UTF-8')
                data.dropna(axis=0, how='all')
                data['text'] = data['text'].apply(lambda x: str(x).lower())
                data['text'] = data['text'].apply((lambda x: re.sub('[^' + turkish_characters + '\s]', '', x)))
                Xx = data['text']

                text = data['text']
                Y = data['sentiment'].values.tolist()

                TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
                TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')
                Paths = jp.JClass('java.nio.file.Paths')

                morphology = TurkishMorphology.createWithDefaults()
                spell_checker = TurkishSpellChecker(morphology)
                morphology = TurkishMorphology.createWithDefaults()
                for i, word in enumerate(text):
                    if spell_checker.suggestForWord(jp.JString(word)):
                        if not spell_checker.check(jp.JString(word)):
                            text[i] = str(spell_checker.suggestForWord(jp.JString(word))[0])
                cumle = []
                cumleler = []
                for i in text:
                    sentence = i
                    analysis = morphology.analyzeSentence(sentence)
                    results = morphology.disambiguate(sentence, analysis).bestAnalysis()

                    for i, result in enumerate(results):

                        if "UNK" in result.getLemmas():
                            pass
                        else:
                            # cumle.append(' '.join(result.getLemmas()))
                            kelime = result.getLemmas()
                            kelime = kelime[0]
                            cumle.append(kelime)

                    cumleler.append(' '.join(word for word in cumle))
                    cumle = []

                from sklearn.feature_extraction.text import TfidfVectorizer
                vect = TfidfVectorizer(ngram_range=(1, 3),
                                       stop_words=stopwords,
                                       max_df=0.8,
                                       min_df=3).fit(cumleler)

                X = vect.transform(cumleler)
                tweet_vect = vect.transform(tweet_vect)
                gerekli = vect.transform(gerekli)
                print("gerekli",gerekli)
                x_train, x_test, y_train, y_test = train_test_split(gerekli, y, test_size=0.20, random_state=2)
                lr.fit(x_train, y_train)
                gerek = lr.predict(tweet_vect)
                print("gerekli - Tweet",gerek,str(tweet))
                # print("gereklitürü",type(gerek))
                # print('LSAING . . .')
                # lsa = TruncatedSVD(n_components=600, n_iter=8)  # Perform the Feature Reduction
                # lsa.fit(X)
                # tweet_vect = lsa.fit_transform(tweet_vect)
                # matrix_T = lsa.fit_transform(X)
                if gerek==1:
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=2)  # random_state=1 sonuçlrın her zaman aynı gelmesi için kullanılır.

                    from imblearn.over_sampling import ADASYN
                    adasyn = ADASYN(ratio='minority', n_neighbors=4)  # minority': sadece azınlık sınıfını yeniden örnekle;
                    X_adasyn, y_adasyn = adasyn.fit_sample(X_train, Y_train)

                    lr.fit(X_adasyn, y_adasyn)
                    polarity = lr.predict(tweet_vect)
                    pol = lr.predict_proba(tweet_vect)
                    print("polarity ", type(polarity))
                    self.TweetsProcessing.tweetsCounter.update(polarity=polarity)
                    print(str(datetime.fromtimestamp(float(time_ms)/1000)),polarity,pol,str(tweet))
                    print("------")

                    self.c.execute("""INSERT INTO """ + self.tableName + """ (UnixTime, Tweet, Polarity) VALUES (%s, "%s", %f)"""
                                % (time_ms, tweet, polarity))
                    self.conn.commit() #sutunların içeriğini tabloya aktarır.
                    time.sleep(0.2)
                    time.sleep(0.1)
                else:
                    pass
        # except Exception as e:
        #     print(str(type(e)) + ": " + str(e))
        #
        #     return(True)

    def on_error(self, status):
        print (status)


class tweetprocessing(Thread):
    def __init__(self, ckey, csecret, atoken, asecret, dbName, tableName, keyWords=[]):
        #Thread.__init__(self, group=None, target=None, name='TweetsProcessing')
        self.ckey = ckey
        self.csecret = csecret
        self.atoken = atoken
        self.asecret = asecret
        self.dbName = dbName
        self.dbName = dbName
        self.tableName = tableName
        self.keyWords = keyWords
        self.tweetsCounter = TweetsSideCounter()



        # def checkKeyWordsUpdates(self):
    #     while True:
    #         reload(Config)
    #         from Config import RunConfig
    #         self.keyWords = RunConfig.keyWords
    #         print ("KeyWords: " + str(self.keyWords))
    #         time.sleep(2)

    def createTwitterDB(self):
        try:
            conn = sqlite3.connect(self.dbName)
            c = conn.cursor()#satır satır işlem yapmak
            c.execute("CREATE TABLE IF NOT EXISTS %s(UnixTime REAL, Tweet VARCHAR(300), Polarity REAL)" % (self.tableName))
            conn.commit()
            c.execute("CREATE INDEX fast_unix ON %s(UnixTime)" % (self.tableName))
            c.execute("CREATE INDEX fast_tweet ON %s(Tweet)" % (self.tableName))
            c.execute("CREATE INDEX fast_sentiment ON %s(Polarity)" % (self.tableName))
            c.execute("CREATE INDEX fast_unix_sentiment ON %s(UnixTime, Polarity)" % (self.tableName))
            conn.commit()
            conn.close()
            return(True)
        except Exception as e:

            print(str(type(e)) + ": " + str(e))
            return(False)


    def run(self):

             try:
                 auth = OAuthHandler(self.ckey, self.csecret)
                 auth.set_access_token(self.atoken, self.asecret)

                 api = API(auth)
                 print("Authorization: " + str(api.me().name))

                 twitterStream = Stream(auth, listener(self, dbName=self.dbName, tableName=self.tableName))
                 twitterStream.filter(track=self.keyWords, is_async =True)
                 time.sleep(0.5)
             except Exception as e:
                 print(str(type(e)) + ": " + str(e))
                 time.sleep(5)

