## DECİSİON TREE Eğitim Doğruluğu: 0.9957446808510638

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

dataset = pd.read_csv("dataset.data", sep=",")
dataset.head()

X = dataset.iloc[:,0:5]
y = dataset.iloc[:,5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

model = DecisionTreeClassifier().fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Eğitim Doğruluğu:",accuracy_score(y_train,y_pred_train))

# Öğrenme Eğrisi
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))

train_mean = np.mean(train_scores, axis=1)
validation_mean = np.mean(test_scores, axis=1)

plt.style.use('seaborn')
plt.plot(train_sizes, train_mean, label = 'Eğitim Hatası')
plt.plot(train_sizes, validation_mean, label = 'Onaylama Hatası')
plt.ylabel('Doğruluk')
plt.xlabel('Eğitim seti boyutu')
plt.title('Öğrenme Eğrisi')
plt.legend()

# ROC Eğrisi

# Çıktıyı 2 li hale getirme
y = label_binarize(y, classes=[2,3,4,5,6])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(model)

y_score = classifier.fit(X_train, y_train).predict(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-ortalama (bölge = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC eğrisi sınıfı {0} (bölge = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Oranı')
plt.ylabel('TP Oranı')
plt.legend(loc="lower right")
plt.show()
