import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def read_data(file):
    f = open(file, "r")
    idx = []
    text = []
    for x in f:
        y = x.split('\t')
        idx.append(y[0])
        text.append(y[1][:-1])
    text = np.asarray(text)
    f.close()
    return text, idx


# load data
test_samples, id_test = read_data('data/test_samples.txt')
train_samples, id_train = read_data('data/train_samples.txt')
train_labels, y = read_data('data/train_labels.txt')
validation_samples, id_validation = read_data('data/validation_samples.txt')
validation_labels, x = read_data('data/validation_labels.txt')

# definim modelul
count_vect = CountVectorizer()

# construim dictionarul si intoarcem features
x_train = count_vect.fit_transform(train_samples)

# intoarcem features
x_val = count_vect.transform(validation_samples)
x_test = count_vect.transform(test_samples)

# definim clasificatorul
clf = MultinomialNB()

# antrenam datele
clf.fit(x_train, train_labels)

# prezicem ptr validare
predicted = clf.predict(x_val)

print('Accuracy score: ', accuracy_score(validation_labels, predicted))

# convertim la int
predicted = predicted.astype(np.int)
validation_labels = validation_labels.astype(np.int)


def confusion_matrics(y_true, y_pred):
    num_classes = max(y_true.max(), y_pred.max()) + 1
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        conf_matrix[int(y_true[i]), int(y_pred[i])] += 1
    return conf_matrix

print('F1 score', f1_score(validation_labels, predicted))
print('Confusion matrix:\n', confusion_matrics(validation_labels, predicted))

# prezicem ptr test
predicted = clf.predict(x_test)

def write_submission(nume_fisier, ids, preds):
    with open(nume_fisier, 'w') as fout:
        fout.write("id,label\n")
        for x, pred in zip(ids, preds):
            fout.write(x +',' + str(int(pred)) + '\n')


write_submission("file.csv", id_test, predicted)
