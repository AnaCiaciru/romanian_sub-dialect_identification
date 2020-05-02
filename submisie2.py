from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np

class BagOfWords:

    def __init__(self):
        self.words = [] #dictionarul
        self.vocabulary_lenght = 0


    def build_vocabulary(self, data):
        for line in data:
            for word in line.split(' '):
                if word not in self.words:
                    self.words.append(word)
        self.vocabulary_lenght = len(self.words)
        self.words = np.array(self.words)

    # vom calcula frecventa pentru fiecare cuvant
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

def read_data(file):
    f = open(file, "r")
    idx = []
    text = []
    for x in f:
        y = x.split('\t')
        idx.append(y[0])
        text.append(y[1])
    text = np.asarray(text)
    f.close()
    return text, idx

# read data
train_samples, id_train = read_data('data/train_samples.txt')
train_labels, id_train = read_data('data/train_labels.txt')

validation_samples, id_validation = read_data('data/validation_samples.txt')
validation_labels, id_validation = read_data('data/validation_labels.txt')

test_samples, id_test = read_data('data/test_samples.txt')

# definim modelul
bow_model = BagOfWords()
bow_model.build_vocabulary(train_samples)

# cream features-urile
train_features = bow_model.get_features(train_samples)
validation_features = bow_model.get_features(validation_samples)
test_features = bow_model.get_features(test_samples)

# impartim in subintervale
def values_to_bins(x, bins):
    x = np.digitize(x, bins)
    return x - 1

num_bins = 20
stop = int(max(train_features.max(), validation_features.max()))
bins = np.linspace(0, stop, num=num_bins)

x_train = values_to_bins(train_features, bins)
x_val = values_to_bins(validation_features, bins)
x_test = values_to_bins(test_features, bins)

# andrenam modelul
clf = MultinomialNB()
clf.fit(x_train, train_labels)

predicted = clf.predict(x_val)
print('Accuracy score: ', accuracy_score(validation_labels, predicted))

# convertim la int
predicted = predicted.astype(np.int)
validation_labels = validation_labels.astype(np.int)

print('F1 score: ', f1_score(validation_labels, predicted))
print('Confusion matrix:\n', confusion_matrix(validation_labels, predicted))

predicted = clf.predict(x_test)

def write_submission(nume_fisier, ids, preds):
    with open(nume_fisier, 'w') as fout:
        fout.write("id,label\n")
        for x, pred in zip(ids, preds):
            fout.write(x +',' + str(int(pred)) + '\n')


write_submission("file.csv", id_test, predicted)