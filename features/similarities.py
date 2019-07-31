##########################################################################
# Copyright (c) 2015 Institut de Recherche Technologique SystemX
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General
# Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
#
# Contributors:
#     Emma Effa - emma.effa-bella@irt-systemx.fr
##############################################################################

# -*- coding: utf-8 -*-
import numpy as np
import csv
import re
import pandas
from features import extractDict
from utils import Readingfile
import time


def countAllTerms(artefact1, artefact2):
    """count all the terms in the pairs of artefacts"""
    total= len(artefact1)+len(artefact2)
    if (total == 0):
        total= 1
    return total


def countIdentities(artefact1, artefact2):
    """count all the identical terms between the pairs of artefacts"""
    sumIdentities=0
   
    for word in artefact1:
            if (word in artefact2):
                sumIdentities = sumIdentities +1
    
    return sumIdentities 
    
def countSynonym(artefact1, artefact2, dict):
    """count all the synonym terms between the pairs of artefacts based on the buils dictionary"""
    sumSynonym=0
    listeSym=[]
 
    for word in artefact1:
        
        
        listeSym=listsynonyms(word, dict)
        
        if(len(listeSym)>0):
            for word2 in artefact2:
                
                if (word2 in listeSym):
                    sumSynonym = sumSynonym + 1
                     
    
    return sumSynonym

def listsynonyms(word, dict):
    """find all the synonym of a word based on the dictionnary"""
    chaine=""
    liste=[]
    listeSym=[]
    
    print(word)
    #recupère l'indice de word
    p=dict[dict['Word']==word].index.values.astype(int)
    
    if(len(p)!=0):
        x=dict[dict['Word']==word].index.item()
        #recuperer liste synonymes
        chaine=dict['Synonym'][x]
        if len(chaine)>2:
            ch1=re.sub("]",'', chaine)
            
            ch2=re.sub("'",'', ch1)
            liste=ch2.split(",")      
            for w in liste:
                ch3=w.strip() 
                listeSym.append(ch3)
          
    return listeSym

def synomym(dict, word):
    """find all the synonym of a word based on the dictionnary"""
    liste=[]
          
    for i in range(0,len(dict)):
        if(word==dict['term'][i]):
            liste.append(dict['synonym'][i])    
    
    return liste


def synonymFile(dict,textNoId, modelNoId, nameF):
    """create the term synonym  file for all the word in the dictionnary """
    listeSym=[]
    allWordListe=[]
    allSymList=[]
    allLineWordListe=[]
        
    for art in textNoId:
        allLineWordListe.append(art)
    for art in modelNoId:
        allLineWordListe.append(art)
    
    print(len(allLineWordListe))
    
    
    dictionary = set()
    for line in allLineWordListe:
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)
    print("here")
    
    for word in dictionary:
        allWordListe.append(word)
    
    print(len(allWordListe))
    
    i=0
    for word in allWordListe:
        listeSym=synomym(dict, word)
        allSymList.append(listeSym)
        i=i+1
        print(i)
    
    print("here2")
    
    for i in range(0, len(allWordListe)):
        print(allWordListe[i])
        print(allSymList[i])
        
    
    #creation of the csv file
    with open(nameF, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Word","Synonym"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len(allWordListe)):
            writer.writerow({'Word': str("{0}".format(allWordListe[i])), 'Synonym': str("{0}".format(allSymList[i]))})  
        
    
    print("synonyms file build")    
    

def synonymPhraseFile(dict,textNoId, modelNoId, nameF):
    """create the pharse synonym  file for all the word in the dictionnary """
    listeSym=[]
    allWordListe=[]
    allSymList=[]
    allLineWordListe=[]
     
      
    for art in textNoId:
        artefact1=chunkPharse(art)
        allLineWordListe.append(artefact1)
    for art in modelNoId:
        artefact2=chunkPharse(art)  
        allLineWordListe.append(artefact2)
    
    print(len(allLineWordListe))
    
    
    dictionary = set()
    for line in allLineWordListe:
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)
    
    print("here")
    
    for word in dictionary:
        allWordListe.append(word)
    
    print(len(allWordListe))
    
    i=0
    for word in allWordListe:
        listeSym=synomym(dict, word)
        allSymList.append(listeSym)
        i=i+1
        print(i)
        
    
    print("here2")
    
    for i in range(0, len(allWordListe)):
        print(allWordListe[i])
        print(allSymList[i])
        
    
    #creation of the csv file
    with open(nameF, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Word","Synonym"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len(allWordListe)):
            writer.writerow({'Word': str("{0}".format(allWordListe[i])), 'Synonym': str("{0}".format(allSymList[i]))})  
        
    
    print("synonyms file build")    
    

def TFIDFWordDoc(doc):
    """compute the frequency of all words in the vocabulary and keep only word with high frequency"""
    liste=[]
    wordstring=" "
    
    for p in doc:
        for q in p:
            wordstring=wordstring + q+ ' '
            
    wordfreq = []
    words=[]
    wordImp=[]
    
    dictionary = set()
    for line in doc:
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)
    
    for word in dictionary:
        words.append(word)
    
    wordlist = wordstring.split()
    
    totalAllWords=len(wordlist)
    totalWords=len(words)
    
    for w in words:
        wordfreq.append(wordlist.count(w))
    
    for i in range(0,len(words)):
        wordImp.append((wordfreq[i] / totalAllWords )* totalWords)
    
    for i in range(0,len(words)):
        if(wordImp[i]>=3):
            liste.append(words[i])
    
    return liste

def TFIDFWord(textNoId, modelNoId):
    """create the liste of frequent word"""
    liste=[]
    reqListe= TFIDFWordDoc(textNoId)
    for art in reqListe:
        liste.append(art)
    modListe= TFIDFWordDoc(modelNoId)
    for art in modListe:
        liste.append(art)
    
    return liste
    

def systemTerms(artefact, liste):
    newArtefact=[]
    """compute tfidf of each term to identifiy its importance in the doc"""
       
    
    for word in artefact:
        #remove words with one letter, number
        if (len(word)>1):
            if (word not in liste):
                newArtefact.append(word)
            
    
    return newArtefact

def chunkPharse(texts):
    """divide a sentence into chunk and preprocessed data"""
    chunk=[]
    
    nlp=extractDict.prepareForNLP(texts)
    
    for p in nlp:
        sent=extractDict.chunk(p)
           
    for psent in sent:
            psent=re.sub("'",'', psent)
            qsent=Readingfile.camelCaseSplitter(psent)
            chun=""
            for u in qsent:
                u=re.sub('\W+','',u)   
                u=re.sub('_','',u)  
                u=u.strip()             
                chun=chun + ' '+ u.lower()
            chun=re.sub(' u  ','',chun)
            chunk.append(chun.strip())
        
       
    return chunk
    


def simDiffTerm(sourceTexts, source_id, len_source, targetTexts, target_id, len_target,simfilename,difffilename,dictTermFilename):
    """Read source and target artefacts and compute similarity for each pair of artefacts.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    source_id: a list of source artefacts ids;
    len_source: number of source artefacts;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    target_id: a list of target artefacts ids;
    len_target: number of target artefacts;
    simfilename: file where the f1 similarity scores results will be saved;
    difffilename: file where the f4 similarity scores results will be saved;
    dictTermFilename: file where the synonym term have been saved.
  Returns:
    None.
    """
    
    vectSim=np.zeros((len_source,len_target))
    vectDiff=np.zeros((len_source,len_target))
    dict=pandas.read_csv(dictTermFilename, names = ['Word','Synonym'], header = 0)    
   
    
    i=0
    for artefact1 in sourceTexts:
        j=0
        for artefact2 in targetTexts:
            total=countAllTerms(artefact1, artefact2)
            sumIdenticalTerm=countIdentities(artefact1, artefact2)
            sumSynonym=countSynonym(artefact1, artefact2, dict)
            sumCommonTerm = (sumSynonym + sumIdenticalTerm)
            vectSim[i,j]= sumCommonTerm / total
            sumDiffTerm=(len(artefact1)-sumCommonTerm)+(len(artefact2)-sumCommonTerm)
            vectDiff[i,j]=sumDiffTerm / total
            toto=str(i)+" "+str(j)
            print(toto)
            j=j+1
        i=i+1
        
    #creation of the csv file
    with open(difffilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectDiff[i,j]))})  
       
    
    print("dissimilarity vector build")
    
    
    with open(simfilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectSim[i,j]))})  
  
    
    print("similarity vector build")
    

def TFIDFsimDiffTerm(sourceTexts, source_id, len_source, targetTexts, target_id, len_target,simfilename,difffilename,dictTermFilename):
    """Read source and target artefacts and compute similarity for each pair of artefacts.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    source_id: a list of source artefacts ids;
    len_source: number of source artefacts;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    target_id: a list of target artefacts ids;
    len_target: number of target artefacts;
    simfilename: file where the f1 similarity scores results will be saved;
    difffilename: file where the f4 similarity scores results will be saved;
    dictTermFilename: file where the synonym term have been saved.
  Returns:
    None.
    """
    
    vectSim=np.zeros((len_source,len_target))
    vectDiff=np.zeros((len_source,len_target))
    #='outputs/EVA/v2/req-mod/synonymsterms.csv'
    dict=pandas.read_csv(dictTermFilename, names = ['Word','Synonym'], header = 0)    
    liste=TFIDFWord(sourceTexts, targetTexts)
   
    
    i=0
    for list1 in sourceTexts:
        j=0
        for list2 in targetTexts:
            artefact1=systemTerms(list1, liste)
            artefact2=systemTerms(list2, liste)
            total=countAllTerms(artefact1, artefact2)
            sumIdenticalTerm=countIdentities(artefact1, artefact2)
            sumSynonym=countSynonym(artefact1, artefact2, dict)
            sumCommonTerm = (sumSynonym + sumIdenticalTerm)
            vectSim[i,j]= sumCommonTerm / total
            sumDiffTerm=(len(artefact1)-sumCommonTerm)+(len(artefact2)-sumCommonTerm)
            vectDiff[i,j]=sumDiffTerm / total
            toto=str(i)+" "+str(j)
            print(toto)
            j=j+1
        i=i+1
    #creation of the csv file
    with open(difffilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectDiff[i,j]))})  
       
    print("dissimilarity vector build")
    
    #creation of the csv file
    with open(simfilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectSim[i,j]))})  
       
    print("common term vector build")
     


def simDiffPharse(sourceArtefacts, source_id, len_source, targetArtefacts, target_id, len_target,simfilename,difffilename,dictPharseFilename):
    """Read source and target artefacts and compute similarity for each pair of artefacts.
  Args:
    sourceArtefacts: a list of source artefacts ;
    source_id: a list of source artefacts ids;
    len_source: number of source artefacts;
    targetArtefacts: a list of target artefacts ;
    target_id: a list of target artefacts ids;
    len_target: number of target artefacts;
    simfilename: file where the f3 similarity scores results will be saved;
    difffilename: file where the f6 similarity scores results will be saved;
    dictPharseFilename: file where the synonym pharses have been saved.
  Returns:
    None.
    """
    
    vectSim=np.zeros((len_source,len_target))
    vectDiff=np.zeros((len_source,len_target)) 
    #='outputs/EVA/v2/req-mod/synonymsPhrases.csv'
    dict=pandas.read_csv(dictPharseFilename, names = ['Word','Synonym'], header = 0)
    
    
    i=0
    for art1 in sourceArtefacts:
        j=0
        for art2 in targetArtefacts:
            start = time. time()
            artefact1=chunkPharse(art1)
            artefact2=chunkPharse(art2)
            total=countAllTerms(artefact1, artefact2)
            sumIdenticalTerm=countIdentities(artefact1, artefact2)
            sumSynonym=countSynonym(artefact1, artefact2, dict)
            sumCommonTerm = (sumSynonym + sumIdenticalTerm)
            vectSim[i,j]= sumCommonTerm / total
            sumDiffTerm=(len(artefact1)-sumCommonTerm)+(len(artefact2)-sumCommonTerm)
            vectDiff[i,j]=sumDiffTerm / total
            end = time. time()
            print(end - start)
            toto=str(i)+" "+str(j)
            print(toto)
            j=j+1
        i=i+1
    
    #creation of the csv file
    with open(difffilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectDiff[i,j]))})  
       
    print("dissimilarity vector build")
    
    
    with open(simfilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(vectSim[i,j]))})  
  
    print("similarity vector build")
    


