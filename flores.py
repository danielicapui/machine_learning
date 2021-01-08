import sys
import scipy
import numpy as np 
import matplotlib as mpl
import pandas
import sklearn as skl

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Data:
    def __init__(self,url,names,dataset):
        self.url=url
        self.names=names
        self.dataset=dataset 
    def criar_data(self):  
        #carraga os dados
        self.url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        #or url="iris.csv"
        self.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataset=pandas.read_csv(url,names=names)
        #shape para ter ideias de quantas instâncias,linhas e atibutos,colunas o arquivo possui
    def infor(self):
        print(self.dataset.shape)
    def ler_linhas(self):
        print(self.dataset.head(self.numero))
    def descricao(self):
        print(self.dataset.describe())
    def mostrar_classe(self):
        # class distribution
        print(self.dataset.groupby('class').size())
    def grafico(self):
        # box and whisker plots grafico
        self.dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
        plt.show()
    def histograma(self):
        self.dataset.hist()
        plt.show()
    def grafico_multi(self):
        #scatter plot matrix
        scatter_matrix(self.dataset)
        plt.show()
    def validar(self):
        # split-out vlaidation dataset
        array=self.dataset.values
        self.x=array[:,0:4]
        self.y=array[:,4]
        self.validation_size=0.20
        self.seed=7
        self.scoring ='accuracy'
        self.x_train,self.x_validation,self.y_train,self.y_validation=model_selection.train_test_split(self.x, self.y, test_size=self.validation_size, random_state=self.seed)
    
    def modelos(self):
        models=[]
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('analise', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
           kfold = model_selection.KFold(n_splits=10, random_state=self.seed,shuffle=True)
           cv_results = model_selection.cross_val_score(model, self.x_train, self.y_train, cv=kfold, scoring=self.scoring)
           results.append(cv_results)
           names.append(name)
           msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
           print(msg)
        return results,names
    def comparar(self,results,names):
        # Compare Algorithms
        figura = plt.figure()
        figura.suptitle('Comparação de algoritmos')
        ax = figura.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
    def analisar_resultado(self):
        #fazer previsões baseadas no resultado
        analise = KNeighborsClassifier()
        analise.fit(self.x_train,self.y_train)
        prever = analise.predict(self.x_validation)
        print(accuracy_score(self.y_validation, prever))
        print(confusion_matrix(self.y_validation, prever))
        print(classification_report(self.y_validation, prever))
    
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#or url="iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pandas.read_csv(url,names=names)      
data=Data(url,names,dataset)
data.validar()
# results,names=data.modelos()
#data.comparar(results,names)
data.analisar_resultado()
print("yupi is on")
