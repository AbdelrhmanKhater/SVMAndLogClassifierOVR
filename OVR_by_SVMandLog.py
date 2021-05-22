import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

def loadDataset():
    dataset = datasets.load_iris()
    features = dataset.data
    labels = dataset.target
    return dataset, features, labels

def SVM_model(X, y):
    model = svm.SVC(decision_function_shape='ovo', probability=True)
    model.fit(X, y)
    return model

def Log_model(X, y):
    model = LogisticRegression(random_state=0, solver='liblinear')
    model.fit(X, y)
    return model

def prepareDataset(X, y, cls_remove =None):
    cls = [ 0, 1, 2]
    if cls_remove is not None:
        features = []
        labels = []
        for index, (feat, label) in enumerate (zip(X, y)):
            if y[index] != cls_remove:
                features.append(feat)
                labels.append(label)
        cls.pop(cls_remove)
    else:
        features = X
        labels = y
    #print(np.array(features)[:, 2:4])
    return np.array(features)[:, 2:4], np.array(labels), cls

def plotData(features, labels, cls, class_names ):
    colors = [ 'r', 'b', 'g']
    markers = [ 'o', '*', '+']
    for class_index in range(len(cls)):
        #print(features[labels == cls[class_index], 0], features[labels == cls[class_index], 1])
        plt.scatter(features[labels == cls[class_index], 0], features[labels ==
    cls[class_index], 1], c=colors[class_index], marker=markers[class_index],
    label=class_names[cls[class_index]])
    Title = "Iris Dataset with {n} classes".format(n=int(len(cls)))
    plt.title(Title)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.legend()

def plotRegions (model):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[ 0], xlim[1], 50)
    yy = np.linspace(ylim[ 0], ylim[1], 50)
    XX, YY = np.meshgrid(xx, yy)
    z = np.vstack([XX.ravel(), YY.ravel()]).T
    ZZ = model.decision_function(z) .reshape(XX.shape)
    ax.contourf(XX, YY, ZZ, colors=[ 'c','y'], levels= 0, alpha=0.2)
    dataset, features, labels = loadDataset()

def getAccuracy (model, x, y):
    return model.score(x,y)* 100

def getClassNames (target_names):
    cls_all = {}
    for i, label in enumerate (target_names):
        cls_all[i] = label
    return cls_all

def Binarize(l, cls):
    labels = []
    for item in l:
        if item == cls:
            labels.append(1)
        else:
            labels.append(-1)
    return labels

#Prepare data
dataset, features, labels = loadDataset()
class_names = getClassNames(dataset.target_names)
f, l, c = prepareDataset(features, labels)
svm_models = []
log_models = []
yb = []
#Binarize
for i in range(3):
    yb.append(Binarize(l, i))

#Create Models
for i in range(3):
    svm_models.append(SVM_model(f, yb[i]))
    log_models.append(Log_model(f, yb[i]))


#Accuracy and Confusion matrix
for i in range(3):
    y_true, y_pred = yb[i], svm_models[i].predict(f)
    print('\nClassification SVM Report ' + str(i + 1) + ':\n')
    print(classification_report(y_true, y_pred))
    print('\nConfusion SVM Matrix ' + str(i + 1) + ':\n')
    print(confusion_matrix(yb[i], y_pred))
    print('Accuracy of the SVM model ' + str(i + 1) + ': {:.2f}'.format(getAccuracy(svm_models[i], f, yb[i])))
    y_true, y_pred = yb[i], log_models[i].predict(f)
    print('\nClassification Log Report ' + str(i + 1) + ':\n')
    print(classification_report(y_true, y_pred))
    print('\nConfusion Log Matrix ' + str(i + 1) + ':\n')
    print(confusion_matrix(yb[i], y_pred))
    print('Accuracy of the Log model ' + str(i + 1) + ': {:.2f}'.format(getAccuracy(log_models[i], f, yb[i])))

class_names= {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

#Plot regions
#SVM
for i in range(3):
    plotData(f, np.array(l), [0, 1, 2], class_names)
    plotRegions(svm_models[i])
    plt.show()

#Log
for i in range(3):
    plotData(f, np.array(l), [0, 1, 2], class_names)
    plotRegions(log_models[i])
    plt.show()

#SVM Prob
SVM_prob = []
for i in range(3):
   SVM_prob.append(svm_models[i].predict_proba(f)[:, 1].tolist())
SVM_prob = np.array(SVM_prob).T

#Log prob
Log_prob = []
for i in range(3):
    Log_prob.append(log_models[i].predict_proba(f)[:, 1].tolist())
Log_prob = np.array(Log_prob).T

#Argmax
svm_argmax = np.argmax(SVM_prob, axis = 1)
log_argmax = np.argmax(Log_prob, axis = 1)
print("SVM arg Accuracy : ", accuracy_score(l, svm_argmax))
print("Log arg Accuracy : ", accuracy_score(l, log_argmax))

#plot correct and wrong
wc = []
for i in range(len(svm_argmax)):
    if (l[i] == svm_argmax[i]):
        wc.append(1)
    else:
        wc.append(0)

class_names= {0: 'Wrong', 1: 'Correct'}
plotData(f, np.array(wc), [0, 1], class_names)
plt.show()

wc = []
for i in range(len(log_argmax)):
    if (l[i] == log_argmax[i]):
        wc.append(1)
    else:
        wc.append(0)

class_names= {0: 'Wrong', 1: 'Correct'}
plotData(f, np.array(wc), [0, 1], class_names)
plt.show()

#Decision Tree
SVM_DTC = DecisionTreeClassifier(random_state=0)
SVM_DTC.fit(SVM_prob, l)
ysvm_dtc = SVM_DTC.predict(SVM_prob)
Log_DTC = DecisionTreeClassifier(random_state=0)
Log_DTC.fit(Log_prob, l)
ylog_dtc = Log_DTC.predict(Log_prob)

#Accuracy and confusion
print('SVM DTC ACC : ', getAccuracy(SVM_DTC, SVM_prob, l))
print('Log DTC ACC : ', getAccuracy(Log_DTC, Log_prob, l))

wc = []
for i in range(len(ysvm_dtc)):
    if (l[i] == ysvm_dtc[i]):
        wc.append(1)
    else:
        wc.append(0)

class_names = {0: 'Wrong', 1: 'Correct'}
plotData(f, np.array(wc), [0, 1], class_names)
plt.show()

wc = []
for i in range(len(ylog_dtc)):
    if (l[i] == ylog_dtc[i]):
        wc.append(1)
    else:
        wc.append(0)

class_names= {0: 'Wrong', 1: 'Correct'}
plotData(f, np.array(wc), [0, 1], class_names)
plt.show()





