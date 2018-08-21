#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:21:53 2018

@author: mindsound
"""

from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import itertools, re
import seaborn as sns
import pandas as pd

class Shapelet():
    def __init__(self, shapeletId, start, length, gain, splitD, rightId):
        self.shapeletId = shapeletId
        self.rightId = rightId
        self.start = start
        self.length = length
        self.gain = gain
        self.splitD = splitD
       
    def addTS(self, TS):
        self.tsN = [float(x) for x in TS.split()]
    
    def denormalize(self, mean, std):
        self.ts = [x*std+mean for x in self.tsN]
    
    
class ConfusionMatrix():
    def __init__(self, matrixFile, listLabel):
        # Extraction de la matrice de confusion
        try:
            with open(matrixFile, "r") as file:
                self._matrix_file = file.readlines()
        except:
            print("Can't open matrix file. ")
        
        self._matrix_file = [x.strip() for x in self._matrix_file]
        #print(self._matrix_file)
        self._labels = listLabel
        self.matrix = []
        
        for i in range(1,len(self._labels)+1):
            #print(self._matrix_file[i])
            line = self._matrix_file[i].split()
            line = [int(x) for x in line]
            self.matrix.append(line)
            
        self.matrix = np.array(self.matrix)
        
        
        mC = re.search("Missed Count ([0-9]+)",self._matrix_file[len(self._labels)+1])
        Acc = re.search("Accuracy ([0-1]?\.?[0-9]+)",self._matrix_file[len(self._labels)+2])
        self.missedCount = int(mC.group(1))
        self.accuracy = float(Acc.group(1))
        
    @classmethod
    def plot_confusion_matrix(cls, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    @classmethod
    def print_confusion_matrix(cls, confusion_matrix, class_names, figsize = (10,7), fontsize=14):
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig
    
    def display(self):
        #ConfusionMatrix.print_confusion_matrix(self.matrix, self._labels)
        plt.figure(figsize=(5,5))
        ConfusionMatrix.plot_confusion_matrix(self.matrix, classes=self._labels, title="Nombre de TS : {} | Accuracy : {}".format(np.sum(self.matrix), self.accuracy))
        plt.show()

    def save(self, path):
        plt.figure(figsize=(5,5))
        ConfusionMatrix.plot_confusion_matrix(self.matrix, classes=self._labels, title="Accuracy : {}".format(np.sum(self.matrix), self.accuracy))
        plt.savefig(path)
        

   
class DecisionTree():
    def __init__(self, treeFile, shapeletFile, groupDict, labels, thEntropy, originalClass, shapeletsDict):
        self._shapelets = {}
        self.shapelets = {}
        self.dot = Digraph(comment="Arbre de décision de shapelet")
        self.dot.attr('node', shape='box')
        
        # Traitement du fichier d'output de main (la recherche de shapelet)
        try:
            with open(shapeletFile, "r") as file:
                self._shapelet_file = file.readlines()
        except:
            print("Can't open shapelet file. ")
        
        self._shapelet_file = [x.strip() for x in self._shapelet_file]
        
        i=3
        while i < len(self._shapelet_file):
            idLine = re.search('Shpelet ID : ([0-9]+) , Start Position : ([0-9]+) , Shapelet Length : ([0-9]+)', self._shapelet_file[i])
            splitLine = re.search('Split informationGain : ([0-9]\.[0-9]+) , split position ([0-9]+) , split distance ([0-9]\.[0-9]+)', self._shapelet_file[i+2])
            
            self._shapelets["{} {}".format(idLine.group(3),round(float(splitLine.group(3)),4))] = Shapelet(idLine.group(1), idLine.group(2), idLine.group(3), splitLine.group(1), splitLine.group(3), shapeletsDict[idLine.group(1)])
            #print("{} {}".format(idLine.group(3),round(float(splitLine.group(3)),4)))
            i+=14
        
        # Traitement du fichier de sortie de main
        try:
            with open(treeFile, "r") as file:
                self._tree_file = file.readlines()
        except:
            print("Can't open tree file. ")
        
        self._tree_file = [x.strip() for x in self._tree_file]
        
        
        pNodes = []
        pNodes.append(self._tree_file[0])

        # Trouver la bonne clé malgré les problèmes d'arrondies
        try:
            shp = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4))]
        except KeyError:
            try:
                shp = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)-0.0001)]
            except KeyError:
                shp = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)+0.0001)]

        labelNode = "idNoeud : {}\nIdShapelet : {}\nStart : {}\nTaille : {}\nDistance de séparation \u2264 {}".format(self._tree_file[0], shp.rightId, shp.start, shp.length, shp.splitD)
        self.dot.node(self._tree_file[0], labelNode)

        # Trouver la bonne clé malgré les problèmes d'arrondies
        try:
            self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4))].addTS(self._tree_file[3])
            self.shapelets[int(self._tree_file[0])] = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4))]
        except KeyError:
            try:
                self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)-0.0001)].addTS(self._tree_file[3])
                self.shapelets[int(self._tree_file[0])] = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)-0.0001)]
            except KeyError:
                self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)+0.0001)].addTS(self._tree_file[3]) 
                self.shapelets[int(self._tree_file[0])] = self._shapelets["{} {}".format(self._tree_file[1], round(float(self._tree_file[2]),4)+0.0001)]  
        
        n = 4
        while n < len(self._tree_file):
            if self._tree_file[n+1] == "0":
                # C'est une feuille
                cNode = self._tree_file[n]
                
                if float(thEntropy) == 0: 
                    classId = "Classe "+labels[int(self._tree_file[n+2])]
                else:
                    classId = ""
                    for label in labels.values():
                        classId += "Classe {} : {} ".format(label, sum(originalClass.iloc[groupDict[int(cNode)]] == label))
                        
                self.dot.node(cNode, classId)
                n+=3
            else:
                # C'est un noeud interne
                cNode  = self._tree_file[n]

                # Trouver la bonne clé malgré les problèmes d'arrondies
                try:
                    self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4))].addTS(self._tree_file[n+3])
                    self.shapelets[int(cNode)] = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4))]
                    shp = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4))]
                except KeyError:
                    try:
                        self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)-0.0001)].addTS(self._tree_file[n+3])
                        self.shapelets[int(cNode)] = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)-0.0001)]
                        shp = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)-0.0001)]
                    except KeyError:
                        self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)+0.0001)].addTS(self._tree_file[n+3])
                        self.shapelets[int(cNode)] = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)+0.0001)]
                        shp = self._shapelets["{} {}".format(self._tree_file[n+1], round(float(self._tree_file[n+2]),4)+0.0001)]

                #labelNode = "idNoeud : {}\nIdShapelet : {}\nStart : {}\nTaille : {}\nDistance de séparation \u2264 {}".format(cNode, groupDict[int(cNode)][int(shp.shapeletId)], shp.start, shp.length, shp.splitD)
                labelNode = "idNoeud : {}\nIdShapelet : {}\nStart : {}\nTaille : {}\nDistance de séparation \u2264 {}".format(cNode, shp.rightId, shp.start, shp.length, shp.splitD)
                self.dot.node(cNode, labelNode)
                pNodes.append(cNode)
                
                n+=4
            
            i = len(pNodes)-1
            added = False
            while not added:
                if i < 0:
                    raise IndexError
                    
                if int(pNodes[i])*2 == int(cNode):
                    lb = "Vrai ({})".format(len(groupDict[int(cNode)]))
                    self.dot.edge(pNodes[i], cNode, label=lb)
                    added = True
                elif int(pNodes[i])*2+1 == int(cNode):
                    lb = "Faux ({})".format(len(groupDict[int(cNode)]))
                    self.dot.edge(pNodes[i], cNode, label=lb)
                    added = True
                else:
                    i-=1
    def getTree(self):
        return self.dot
    
    def saveTree(self, fileName):
        self.dot.render("figures/{}".format(fileName))
    
    def getShapelets(self):
        return self.shapelets
 

#labels = ["WW","MWD","SWD"]
#path = "/home/mindsound/Nextcloud/Documents/PROJETS/LIRMM_ECD/data/WR-paper-2017/temp/"
#myTree = DecisionTree(path+"data_train_tree", path+"output", labels)
#dot = myTree.getTree()
#res["dot"]
#res = ConfusionMatrix("temp/confusionMatrix", labels)
#res.display()