import pandas as pd
from string import digits, punctuation
import jpype as jp
from nltk.corpus import stopwords

stopwords = stopwords.words("turkish")
# %%
# Verileri pandas ile okuyup gerekli sütunları listeler olarak kaydediyor:
df = pd.read_csv(r'C:\Users\hlltk\OneDrive\Masaüstü\train_three.csv', sep='|', encoding='UTF-8')
text = df["text"].values.tolist()
sentiment = df["sentiment"].values.tolist()


# %%
# Tekrar eden birbirinin kopyası verileri silmek için tanımlanan fonksiyon:

def cleanDuplicate(text, sentiment):
    lines_seen = set()  # holds lines already seen
    delete_indexes = []
    for i in range(len(text)):
        line = text[i]
        if line not in lines_seen:  # not a duplicate
            lines_seen.add(line)
        else:  # a duplicate so delete items from texts and sentiments:
            delete_indexes.append(i)
    delete_indexes.sort(reverse=True)
    for item in delete_indexes:
        del text[item]
        del sentiment[item]

    return text
    return sentiment


# %%
# Fonksiyon kullanılıyor
cleanDuplicate(text, sentiment)
# %%

# Java virtual machine çalıştırılıyor

# Zemberek.jar'ın kurulu olduğu path
ZEMBEREK_PATH = r"C:\Program Files\Java\Zemberek-Python\Zemberek-Python-Examples-master\bin\zemberek-full.jar"
# Libjvm.so dosyasının bulunduğu path
jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

# Zemberek içerisinden kullanılacak Java sınıfları jpype yardımıyla çekiliyor
TurkishMorph = jp.JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')
morph = TurkishMorph.createWithDefaults()
spell = TurkishSpellChecker(morph)

# %%
# Spellchecker örnek
print(spell.suggestForWord(""))


# %%
# Spellcheck yapılıyor ve her kelime için ilk öneri yeni kelime olarak alınıyor.
def spellChecker(liste):
    clear = str.maketrans(' ', ' ', punctuation)
    suggestions = []
    last = []
    last2 = []
    for i in range(len(liste)):
        words = str(liste[i])
        words = words.translate(clear)
        words = words.split()
        for word in words:
            if spell.suggestForWord(word):
                suggestions.append(list(spell.suggestForWord(word)))

            else:
                pass
            for i in range(len(suggestions)):
                suggestion = suggestions[i]
                words[i] = suggestion[0]
                suggestions = []
                last.append(words[i])
        last2.append(last)
        last = []
    return last2


# %%


def listToCSV(text, sentiment):
    y = []
    x = []
    clear = str.maketrans(' ', ' ', punctuation)

    for i in range(len(sentiment)):
        temp = str(sentiment[i])
        temp = temp.translate(clear)
        temp = temp
        y.append(temp)

    for j in range(len(text)):
        temp = str(text[j])
        temp.lower()
        temp = temp.translate(clear)
        x.append(temp)
    x = pd.DataFrame(x, columns=["text"])
    y = pd.DataFrame(y, columns=["sentiment"])
    df = pd.concat([x, y], axis=1)
    df.to_csv("PreProcessed_test.csv", index=None, sep='|')

    print("file saved as csv")


# %%
listToCSV(text, sentiment)


