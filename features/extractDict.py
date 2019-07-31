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
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from utils import Readingfile
from SIFmaster.src import data_io, params, sim_algo, SIF_embedding 


def cleanArt(artefact):
    """clean data by removing space, number etc..."""
    termList=[]
    for term in artefact:
        term=term.strip()
        if(len(term)>1):
            term1=re.sub("^id[0-9]*$", '', term)
            if(len(term1)!=0):
                term2=re.sub("^[0-9]*", '', term)
                if(len(term2)>1):
                    termList.append(term)
    return termList

def dictionnary(sourceTexts, targetTexts,filename):
    
    """create the term dictionnary: a set to contain all terms in the source and target document
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    filename: file where the all terms and theirs synonym will be saved;
  Returns:
    None.
    """
    termList=[]
    allListe=[]
    for art in sourceTexts:
        art1=cleanArt(art) 
        allListe.append(art1)
    for art in targetTexts:
        art2=cleanArt(art) 
        allListe.append(art2)
    
    dictionary = set()
    for line in allListe:
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)
    
    for word in dictionary:
        termList.append(word)
        
    print(termList)
    print(len(termList))
    
        
    mat=np.zeros((len(termList),len(termList)))
    
    if not os.path.exists('helpers/GoogleNews-vectors-negative300.bin.gz'):
        raise ValueError("SKIP: You need to download the google news model and put it on the helpers directory")
    
    #if not os.path.exists('helpers/glove.6B.300d.txt'):
    #    raise ValueError("SKIP: You need to download the glove model")
    
    model = KeyedVectors.load_word2vec_format('helpers/GoogleNews-vectors-negative300.bin.gz', binary=True)
    #model = Word2Vec(allListe, min_count=1)
        
    # load the Stanford GloVe model
    #filename = 'glove.6B.300d.txt.word2vec'
    #glove2word2vec('helpers/glove.6B.300d.txt', filename)
    #model = KeyedVectors.load_word2vec_format(filename, binary=False)
    
    for i in range(0,len(termList)):
        #print(termList[i])
        for j in range(0, len(termList)):
            mat[i,j]=model.similarity(termList[i], termList[j]) 
    
    #print(mat)
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("term","synonym","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len(termList)):
            for j in range(0,len(termList)):
                if(mat[i,j]>=0.5):
                    if(termList[i]!=termList[j]):
                        writer.writerow({'term': str("{0}".format(termList[i])), 'synonym': str("{0}".format(termList[j])), 'probability': str("{0}".format(mat[i,j]))})  
      
           
    
    print("semantic network build")
    

def allDictionnary(sourceTexts, targetTexts, sourceArtefacts, targetArtefacts,filename):
    
    """create the pharse dictionnary: a set to contain all terms in the source and target document
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    sourceArtefacts: a list of source artefacts ;
    targetArtefacts: a list of target artefacts ;
    filename: file where the all noun or verbal pharses and theirs synonym will be saved.
  Returns:
    None.
    """
    termList=[]
    allListe=[]
    for art in sourceTexts:
        art1=cleanArt(art) 
        allListe.append(art1)
    for art in targetTexts:
        art2=cleanArt(art) 
        allListe.append(art2)
        

    #identify all pharses in the document
    allChunk=[]
    NLP_chunk=[]
    for req in sourceArtefacts:
        sent=prepareForNLP(req)
        for p in sent:
            NLP_chunk.append(p)
    
    for mod in targetArtefacts:
        sent=prepareForNLP(mod)
        for p in sent:
            NLP_chunk.append(p)
    
    for sentence in NLP_chunk:
        sent=chunk(sentence)
        for psent in sent:
            psent=re.sub("'",'', psent)
            qsent=Readingfile.camelCaseSplitter(psent)
            if (len(qsent)>1):
                chun=""
                for u in qsent:
                    u=re.sub('\W+','',u)   
                    u=re.sub('_','',u)  
                    u=u.strip()             
                    chun=chun + ' '+ u.lower()
                chun=re.sub(' u  ','',chun)
                allChunk.append(chun.strip())
    
    
    #build the term and phrase dictionnary   
    dictionary = set()
    for line in allListe:
        unique_terms = set(line)
        dictionary = dictionary.union(unique_terms)
    
        
    unique_terms1 = set(allChunk)
    dictionary = dictionary.union(unique_terms1)
    
        
    for word in dictionary:
        termList.append(word)
    
    print(termList)
        
    mat=np.zeros((len(termList),len(termList)))
    
    mat=simSentenceEmb(termList, mat) 
    
    
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("term","synonym","probability"))
        writer.writeheader()
        for i in range(0,len(termList)):
            for j in range(0,len(termList)):
                if(mat[i,j]>=0.5):
                    if(termList[i]!=termList[j]):
                        writer.writerow({'term': str("{0}".format(termList[i])), 'synonym': str("{0}".format(termList[j])), 'probability': str("{0}".format(mat[i,j]))})  
      
           
    
    print("semantic network build")
    




def prepareForNLP(text):
    """tokenisation and chunking of source and target texts"""
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


def chunk(sentence):
    """identifiy the noun and verbal pharses in sentence"""
    chunk=[]
    chunkToExtract = """
     NP:  {<JJ|NN.*>+|<V.*|NN.*>+}          # Chunk sequences of JJ, NN or VB, NN  
     """
    parser = nltk.RegexpParser(chunkToExtract)
    #print(parser)
    result = parser.parse(sentence)
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            t = subtree
            t = ' '.join(word for word, pos in t.leaves())
            #print(t)
            chunk.append(t)
    return chunk



def simSentenceEmb(termList, mat):  
    """compute similarity between a term and a pharse. this function uses SIFmaster code"""
    # input
    wordfile = 'helpers/glove.6B.300d.txt' # word vector file, can be downloaded from GloVe website
    weightfile = 'SIFmaster/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
    weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 1 # number :of principal components to remove in SIF weighting scheme
    #sentences = [pharse1, pharse2]
    
        
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
    
    x, m = data_io.sentences2idx(termList, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind) # get word weights
    
    # set parameters
    param = params.params()
    param.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, param) # embedding[i,:] is the embedding for sentence i
    
    
    for i in range(0,len(embedding)):
        for j in range(0, len(embedding)):
            mat[i,j]=sim2sentence(embedding[i], embedding[j])
            
        
    return mat

  
    
def sim2sentence(emb1, emb2):
    """compute similarity between two pharses. this function uses SIFmaster code"""
    #Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    scores=sim_algo.emb_weighted_average_sim_rmpc(emb1, emb2)
    preds = np.squeeze(scores)
    
       
    return preds

 
def showPlot(model):
    """show plot of the semantic network build"""
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    ## create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    