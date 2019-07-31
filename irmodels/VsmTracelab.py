"""
/// this is a python version of the VSM.cs file
/// Responsible for computing VSM similarities

/// SEMERU Component Library - TraceLab Component Plugin
/// Copyright � 2012-2013 SEMERU
/// 
/// This file is part of the SEMERU Component Library.
/// 
/// The SEMERU Component Library is free software: you can redistribute it and/or
/// modify it under the terms of the GNU General Public License as published by the
/// Free Software Foundation, either version 3 of the License, or (at your option)
/// any later version.
/// 
/// The SEMERU Component Library is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
/// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
/// more details.
/// 
/// You should have received a copy of the GNU General Public License along with the
/// SEMERU Component Library.  If not, see <http://www.gnu.org/licenses/>.
"""


# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import numpy as np
import pandas
import csv
from nltk.stem.porter import PorterStemmer

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()



def TLSimilarityMatrixCompute(sourceTexts, source_id, len_source, TargetTexts, target_id, len_target,filename):
    """
    /// <summary>
    /// Computes similarities between documents by transforming 
    /// in term-by-document matrices via the Vector Space Model
    /// using a tf-idf weighting scheme and cosine similarity and
    /// </summary>
    /// <param name="source">Source artifacts</param>
    /// <param name="target">Target artifacts</param>
    /// <returns>Similarity matrix</returns>
    """
    #print(texts)
    #print(modeltexts)
    src_stem=[]
    tg_stem=[]
    # stem tokens
    for t in sourceTexts:
        #print(t)
        stemmed_texts = [p_stemmer.stem(i) for i in t]
        #print(stemmed_texts)
        src_stem.append(stemmed_texts)
   
    for t in TargetTexts:
        #print(t)
        stemmed_texts = [p_stemmer.stem(i) for i in t]
        #print(stemmed_texts)
        tg_stem.append(stemmed_texts)
    
    #print(stemmed_texts)
    #print(stemmed_modeltexts)
    
    #build the source term Dictionary
    dictSource = buildDictArtefact(src_stem)
    numTermsSrc = len(dictSource) 
    #create the TermDocumentMatrix
    source = TermDocumentMatrix(sourceTexts, len_source, dictSource)
    TermDocumentMatrixIDs = ComputeIdentities(source, len_source, numTermsSrc )
    
    print("Source TermDocumentMatrix created")
    
    #build the target term Dictionary
    dictTarget = buildDictArtefact(tg_stem)
    numTermsTg = len(dictTarget) 
    #create the TermDocumentMatrix
    target = TermDocumentMatrix(TargetTexts, len_target, dictTarget)
    TermDocumentMatrixTF = ComputeTF(target, len_target, numTermsTg, TargetTexts)
    DFtarget = ComputeDF(target, len_target, numTermsTg)
    IDF = ComputeIDF(DFtarget, len_target)
    TermDocumentMatrixTFIDF = ComputeTFIDF(TermDocumentMatrixTF, IDF, len_target, numTermsTg)
    
    print("Target TermDocumentMatrix created")
    
    #build the common term Dictionary
    #print(texts)
    #print(modeltexts)
    dictionary=buildDict(sourceTexts, TargetTexts)
    numTerms=len(dictionary) 
    sims = ComputeSimilarities(TermDocumentMatrixIDs, TermDocumentMatrixTFIDF, len_source, len_target, numTerms, dictionary, dictSource ,dictTarget) 
    
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(sims[i,j]))})  
       
    
    print("similarity matrix build")
        
    
    


def buildDict(texts, modeltexts):
    """
    /// dictionary: a set to contain all terms (i.e., words) in the document corpus.
    /// Reads in each input document splits into a list of terms and adds new terms
    /// to the global dictionary
    """
    dictionary = set()
    for line in texts:
        #print (line)
        unique_terms = set(line)
        #print (unique_terms)
        dictionary = dictionary.union(unique_terms)
        #print(dictionary)
    for line in modeltexts:
    #    print (line)
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)   
    return dictionary    




def buildDictArtefact(artefacts):
    """
    /// dictionary: a set to contain all terms (i.e., words) in the document corpus.
    /// Reads in each input document splits into a list of terms and adds new terms
    /// to the global dictionary
    """
    dictionary = set()
    for line in artefacts:
        #print (line)
        unique_terms = set(line)
        #print (unique_terms)
        dictionary = dictionary.union(unique_terms)
     
    return dictionary




def TermDocumentMatrix(texts, len_req, dictionary):
    """
    /// <summary>
    /// Raw term-by-document matrix
    /// Rows: documents
    /// Columns: terms
    /// </summary>
    """
    numTerms=len(dictionary)
    #print(numTerms)
    #print(len_req)
    #print(dictionary)
    matrix= np.zeros((len_req,numTerms))
    #print(matrix)
    # the value is the frequency of the term in the document     
    for i in range(0, len_req):
        for j in range(0, len(texts[i])):
            #print(texts[i][j])
            k=0
            for term in dictionary:
                if (term == texts[i][j]):
                    matrix[i][k]=matrix[i][k]+1
                k=k+1 
     
    #print (matrix)        
    return matrix 




def ComputeIdentities(matrix, NumDocs, NumTerms ):    
    """    
    /// <summary>
    /// Computes boolean (0|1) terms in documents.
    /// </summary>
    /// <param name="matrix">Term-by-document matrix</param>
    /// <returns>Term-by-document matrix with 1s for terms that are in the document and 0s for terms that are not.</returns>
    """
    for i in range(0, NumDocs):
        for j in range(0, NumTerms):
            #print(matrix[i,j])
            if (matrix[i,j] > 0.0):
                matrix[i,j] = 1.0 
            else: 
                matrix[i,j] = 0.0
             
    #print (matrix)            
    return matrix
        

def ComputeTF(matrix, NumDocs, NumTerms, modeltexts ):
    """
    /// <summary>
    /// Computes the term frequencies of each document.
    /// </summary>
    /// <param name="matrix">Term-by-document matrix</param>
    /// <returns>tf-weighted term-by-document matrix</returns>
    """
    for i in range(0, NumDocs):
        max = len(modeltexts[i])
        #print(max)
        for j in range(0, NumTerms):
            matrix[i,j] = matrix[i,j] / max
    #print(matrix)
    return matrix
    

def ComputeDF(matrix, NumDocs, NumTerms ):
    """
    /// <summary>
    /// Computes the document frequencies of each term
    /// </summary>
    /// <param name="matrix">Term-by-document matrix</param>
    /// <returns>df-weighted term distribution</returns>
    """
    df = np.zeros((NumTerms))
    for j in range(0, NumTerms):
        for i in range(0, NumDocs):
            if (matrix[i,j] > 0.0):
                #print(matrix[i,j])
                df[j] = df[j] + 1.0 
            #else: 
                #df[j]= df[j]+ 0.0
    #print(df)        
    return df
    


def ComputeIDF(df, numDocs ):
    """
    /// <summary>
    /// Computes the inverse document frequencies of a document frequencies vector
    /// </summary>
    /// <param name="df">Document frequencies vector</param>
    /// <returns>Inverse document frequencies vector</returns>
    """
    idf = np.zeros((len(df)))
    for i in range(0, len(df)):
        if (df[i] > 0.0):
            idf[i] = math.log(numDocs / df[i])
    
    #print(idf)    
    return idf
        

def ComputeTFIDF(tf, idf, NumDocs, NumTerms):
    
    """
    /// <summary>
    /// Computes tf-idf weights
    /// </summary>
    /// <param name="tf">Term-frequency weighted matrix</param>
    /// <param name="idf">Inverse document frequencies vector</param>
    /// <returns></returns>
    """
    for i in range(0,NumDocs):
        for j in range(0, NumTerms):
            tf[i,j] = tf[i,j] * idf[j]
            
    #print(tf)        
    return tf


def Equalize(matrix1, matrix2, len_req, len_mod, NumTerms, dictionary, dictSource ,dictTarget):
    """
    /// <summary>
    /// Recreates each matrix with documents containing missing terms.
    /// List[0] : matrix 1
    /// List[1] : matrix 2
    /// </summary>
    /// <param name="matrix1">First term-by-document matrix</param>
    /// <param name="matrix2">Second term-by-document matrix</param>
    /// <returns>Copies of original matrices with missing terms from each</returns>
    """
    # create new term distributions for each document
    # matrix 1
    #  fill in missing terms
    #print(dictionary)
    #print(dictSource)
    #print(matrix1)
    #print(dictTarget)
    #print(matrix2)
    source=np.zeros((len_req, NumTerms))
    for i in range(0,len_req):
        j=0
        for term in dictionary:
            #print(term)
            # fill in original values
            k=0
            for word in dictSource:
                #print(word)
                if (term == word):
                    source[i, j] = matrix1[i, k]
                k=k+1
            j=j+1  
    # matrix 2
    #  fill in missing terms
    target=np.zeros((len_mod, NumTerms))
    for i in range(0, len_mod):
        j=0
        for term in dictionary:
            #print(term)
            # fill in original values
            k=0
            for word in dictTarget:
                #print(word)
                if (term == word):
                    target[i, j] = matrix2[i, k]
                k=k+1
            j=j+1  
    # return
    #print(source)
    #print(target)
    
    return source,target
    

    


def ComputeSimilarities(ids, tfidf, len_req, len_mod, NumTerms, dictionary, dictSource ,dictTarget):
    """
    /// <summary>
    /// Computes cosine similarities between a set of boolean document vectors and a tfidf weighted corpus
    /// </summary>
    /// <param name="ids">Boolean document vectors</param>
    /// <param name="tfidf">tf-idf weighted document vectors</param>
    /// <returns>Similarity matrix</returns>
    """
    matrice0, matrice1 = Equalize(ids, tfidf, len_req, len_mod, NumTerms, dictionary, dictSource, dictTarget)
    sims = np.zeros((len_req,len_mod))
    for i in range(0, len_req):
        for j in range(0, len_mod):
            product  = 0.0
            asquared = 0.0
            bsquared = 0.0
            for k in range (0, NumTerms):
                a = matrice0[i, k]
                b = matrice1[j, k]
                product  = product + (a * b)
                asquared = asquared + math.pow(a, 2)
                bsquared = bsquared + math.pow(b, 2)
                cross = math.sqrt(asquared) * math.sqrt(bsquared)
            if (cross == 0.0):
                sims[i,j] = 0.0
            else:
                sims[i,j] = product / cross
                
            
    #print(sims)        
    return sims
 
   
    