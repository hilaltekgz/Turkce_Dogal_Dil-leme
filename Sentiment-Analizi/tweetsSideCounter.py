from keys import KeyConf

class TweetsSideCounter(object):
    def __init__(self):
        self.positiveCount = 0
        self.negativeCount = 0
        self.neutralCount = 0
        self.positiveNegativeThreshold = KeyConf.positiveNegativeThreshold

    def update(self, polarity):
        if polarity == 1:
            self.positiveCount = self.positiveCount + 1
        else:
            self.negativeCount = self.negativeCount + 1
        print("negative Count:",self.negativeCount)
        print("positive Count:",self.positiveCount)