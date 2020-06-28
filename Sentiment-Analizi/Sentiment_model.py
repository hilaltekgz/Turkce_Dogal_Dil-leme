import tweepy
import jpype
import re
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
stopwords=stopwords.words("turkish")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report,recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

def startJVM():
    # JVM başlat
    # Aşağıdaki adresleri java sürümünüze ve jar dosyasının bulunduğu klasöre göre değiştirin
    ZEMBEREK_PATH = "C:/Program Files/Java/zemberek_jar/zemberek-tum-2.0.jar"
    jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
    print(jpype.getDefaultJVMPath())
    # Türkiye Türkçesine göre çözümlemek için gerekli sınıfı hazırla
    Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    print("tr",Tr)
    # tr nesnesini oluştur
    tr = Tr()
    # Zemberek sınıfını yükle
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")

    # zemberek nesnesini oluştur
    zemberek = Zemberek(tr)
    return zemberek

zemberek = startJVM()
# Gönderilen metinler Zemberek kütüphanesi fonksiyonları ile düzenlenir.
# Örnek: calismiyor -> çalışmıyor, dugun -> düğün veya söylersn -> söylersin
# Zemberek kütüphanesinin "oner" ve "asciidenTurkceye" fonksiyonları kullanılır.
def yorumDuzelt(yorum, zemberek):
    yorumEdit = ""
    for kelime in yorum.split():
        if ("et" == kelime.strip()) or ("https" in kelime.strip()):
            continue
        if zemberek.kelimeDenetle(kelime) == 0:
            turkce = []
            oneriler = []
            turkce.extend(zemberek.asciidenTurkceye(kelime))
            if len(turkce) == 0:
                oneriler.extend(zemberek.oner(kelime))
                if len(oneriler) > 0:
                    yorumEdit = yorumEdit + " " + oneriler[0]
                    #print("yorumEdit", yorumEdit)
                else:
                    yorumEdit = yorumEdit + " " + kelime
            else:
                yorumEdit = yorumEdit + " " + turkce[0]
        else:
            yorumEdit = yorumEdit + " " + kelime
    return re.sub(u'[^'+turkish_characters+'\s]','',yorumEdit.strip().lower())

# Bu kısım sadece örnek verisetine göre Tokenizer yapısının oluşturulması için kullanılır.
# Ağın elle verilen örneklerin anlam analizini yapabilmesi için bunları tam sayı dizilerine dönüştürmesi gerekir.
# Bu kısım tam sayı dizisine dönüştürme için gerekli olan Tokenizer yapısının örnek veriseti ile oluşturulmasını sağlar.
turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"
#‪data/data.tsv ----     ------  C:/Users/hlltk/OneDrive/Masaüstü/sosyalmedya_datasets/Turkce-Anlam-Analizi-master/T1urkce-Anlam-Analizi-master/data/data.tsv
data = pd.read_csv(r'C:\Users\hlltk\OneDrive\Masaüstü\İşyeriEgitimi\SosyalMedyaAnalizi\PreProcessed.csv',sep='|',encoding='UTF-8')
data['text'] = data['text'].apply(lambda x: str(x).lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^'+turkish_characters+'\s]','',x)))
Xx=data['text']
print('Xx uzunluk',Xx)
Y=data['sentiment'].values

sayi_n=0
sayi_p=0
sayi_u=0



for i in data['sentiment']:
    if i=='negative':
        sayi_n =sayi_n+1
    elif i=='neutral':
        sayi_u=sayi_u+1
    else:
        sayi_p=sayi_p+1
print('negatif sayisi',sayi_n)
print('nötr sayisi',sayi_u)
print('positive sayisi',sayi_p)


#tokenizer = Tokenizer(split=' ',num_words=25000)
#X=tokenizer.fit_on_texts(data['Review'].values)
#X = pad_sequences(X,maxlen=400)
# duzelt=[]
# for i in data['Review']:
#     duzelt=yorumDuzelt(i,zemberek)
# # print("düzelt",duzelt)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=(1,3),
                       stop_words=stopwords,
                       max_df=0.8,
                       min_df=2).fit(Xx)
#X= vect.transform(Xx)
# print("xx",type(Xx))


X_train, Xx_test, Y_train, Y_test = train_test_split(Xx,Y, test_size = 0.2, random_state = 2)#random_state=1 sonuçlrın her zaman aynı gelmesi için kullanılır.
import keras
print("X-test",Xx_test)


X_test=vect.transform(Xx_test)
X_train=vect.transform(X_train)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty="l2",
                      fit_intercept=True,
                      solver="liblinear",
                      max_iter=500)
lr.fit(X_train,Y_train)
print(type(lr))
pred = lr.predict(X_test)
cmlr = confusion_matrix(pred,Y_test)
print('logistic Regresyon')
print(cmlr)
print ("Final Accuracy: %s"
       % accuracy_score(Y_test, lr.predict(X_test)))
print (precision_score(Y_test, lr.predict(X_test), average='weighted')) # Tahminlerinizin yüzde kaçı doğru

report=classification_report(Y_test,lr.predict(X_test))

print("Report",report)


save_model=joblib.dump(lr, 'logistic_model.pkl')





