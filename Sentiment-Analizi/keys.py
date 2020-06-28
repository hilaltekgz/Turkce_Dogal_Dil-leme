class KeyConf:
    ckey="GxgcsIxg5y7ri2NDS3Ziun3cI"
    csecret="0a7nP0NM5sdqhVo6vyDWU0XpXeeNCJBf4yYu6IxqZETjGjfkiZ"
    atoken="1099803395093864449-wUl9Rl3fhloMqgcbkehFZmxPCgLIog"
    asecret="Ic9fJUzU7Kso1RFD98Griv65N0vxCMMdutl0MpYAideYP"
    dbNameInit = 'twitter111.db'
    tableName = 'Tweets11'
    positiveNegativeThreshold = 0.001 #eşik değeri
    keyWords= ["vakifbank", "Vakıfbank", "vakıfbank", "Vakifbank"]
    dbName = dbNameInit.split('.')[0]+"_".join(keyWords)+".db"