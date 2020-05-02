import numpy as np

class BagOfWords:

    def __init__(self):
        self.words = [] #dictionarul
        self.vocabulary_lenght = 0

    # vom da un id unic fiecarui cuvant (indexul)
    def build_vocabulary(self, data):
        for line in data:
            for word in line.split(' '):
                if word not in self.words:
                    self.words.append(word)
        self.vocabulary_lenght = len(self.words)
        self.words = np.array(self.words)

    # vom calcula frecventa pentru fiecare cuvant folosind id-ul sau
    def get_features(self, data):
        num_samples = len(data)
        features = np.zeros((num_samples, self.vocabulary_lenght))
        for line_idx, line in enumerate(data):
            for word in line.split(' '):
                if word in self.words:
                    features[line_idx, np.where(self.words == word)[0][0]] += 1
        return features

    def printVocabulary(self):
        print(self.words)