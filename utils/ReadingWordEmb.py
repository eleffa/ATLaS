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

from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import Word2Vec
import operator



def load_Cbow_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        min_count (int): min count of word in documents to consider for word vector creation
        embedding_dim (int): output wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=1, size=embedding_dim, sg=0)
    word_vectors = model.wv
    del model
    return word_vectors


def load_Skipgram_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        min_count (int): min count of word in documents to consider for word vector creation
        embedding_dim (int): output wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=1, size=embedding_dim, sg=1)
    word_vectors = model.wv
    del model
    return word_vectors



def compute_Cbow( sourceTexts, TargetTexts):
    """
    load cbow model and compute the embedding matrix for each pair of artefacts
    Args:
        sourceTexts: a list of source artefacts tokenized with stopword removed;
        targetTexts: a list of target artefacts tokenized with stopword removed.
    Returns:
        tokenizer: all the tokens
        embedding_matrix:matrix of embedding values of all words
        sequences:list of tokens of a pair of artefacts
        maxlen: maximal length of  pairs of artefacts. all pairs vectors are put in the same size.
        num_words: number of word of the artefacts pairs
        embedding_dim: output wordvector size;
        artifact_pairs:list of pairs of artefacts.
    """
       
    #always put the id of the artifact behind create function before using align_data function
    #align data by id
    sourceTexts=align_data(sourceTexts)
    TargetTexts=align_data(TargetTexts)  
     
    
    
    allwords=[]
    for i in sourceTexts:
        allwords.append(i)
    for j in TargetTexts:
        allwords.append(j)
      
      
    ###prepare embedding model for the cnn model  for text in source artifact
    #preprocessing text of artifact  
        
    artifact_pairs = [(x,y) for x in sourceTexts for y in TargetTexts] 
    
    #print (" ".join([" ".join(x) for x in artifact_pairs]))
    lstoken=[]
    for x in artifact_pairs:
        p=" ".join([" ".join(p) for p in x])        
        lstoken.append(p)
    
              
    print("all artifacts merged")
    
    
       
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lstoken)
    
    #hyperparameters
    num_words = len(tokenizer.word_index) + 1
    embedding_dim=100
    
    sequences = tokenizer.texts_to_sequences(lstoken)
        
    length = []
    for x in lstoken:
        length.append(len(x.split()))
        
    maxlen=max(length)
    
          
    #all vectors are put to the same size
    maxlen = maxlen + 5
     
    #load cbow model
    model_cbow = load_Cbow_word2vec(allwords, embedding_dim)    
   
             
    # build embedding matrix so that it is the same as vectors for each word in the model
    word_index = tokenizer.word_index
   
     
    embedding_matrix = np.zeros((num_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        embedding_vector = model_cbow[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
            
    #print(np.array_equal(embedding_matrix[1] ,model_cbow['get']))
    
    return tokenizer, embedding_matrix, sequences, maxlen, num_words, embedding_dim, artifact_pairs


def compute_Skipgram( sourceTexts, targetTexts):
    """
    load Skipgram model and compute the embedding matrix for each pair of artefacts
    Args:
        sourceTexts: a list of source artefacts tokenized with stopword removed;
        targetTexts: a list of target artefacts tokenized with stopword removed.
    Returns:
        tokenizer: all the tokens
        embedding_matrix:matrix of embedding values of all words
        sequences:list of tokens of a pair of artefacts
        maxlen: maximal length of  pairs of artefacts. all pairs vectors are put in the same size.
        num_words: number of word of the artefacts pairs
        embedding_dim: output wordvector size;
        artifact_pairs:list of pairs of artefacts.
    """
       
    
    #align data by id
    sourceTexts=align_data(sourceTexts)
    targetTexts=align_data(targetTexts)  
     
    
    
    allwords=[]
    for i in sourceTexts:
        allwords.append(i)
    for j in targetTexts:
        allwords.append(j)
      
      
    ###prepare embedding model for the cnn model  for text in source artifact
    #preprocessing text of artifact  
        
    artifact_pairs = [(x,y) for x in sourceTexts for y in targetTexts] 
    
    #print (" ".join([" ".join(x) for x in artifact_pairs]))
    lstoken=[]
    for x in artifact_pairs:
        p=" ".join([" ".join(p) for p in x])        
        lstoken.append(p)
    
              
    print("all artifacts merged")
    
    
       
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lstoken)
    
    #hyperparameters
    num_words = len(tokenizer.word_index) + 1
    embedding_dim=100
    
    sequences = tokenizer.texts_to_sequences(lstoken)
        
    length = []
    for x in lstoken:
        length.append(len(x.split()))
        
    maxlen=max(length)
    
          
    #all vectors are put to the same size
    maxlen = maxlen + 5
     
    #load cbow model
    model_cbow = load_Skipgram_word2vec(allwords, embedding_dim)
    
   
    
    #we padded all our sentences to have the same length    
    #artifactPairs_seq = pad_sequences(sequences, maxlen=maxlen)
           
    # build embedding matrix so that it is the same as vectors for each word in the model
    word_index = tokenizer.word_index
   
     
    embedding_matrix = np.zeros((num_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        embedding_vector = model_cbow[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
            
    #print(np.array_equal(embedding_matrix[1] ,model_cbow['get']))
    
    return tokenizer, embedding_matrix, sequences, maxlen, num_words, embedding_dim, artifact_pairs


def align_data(dataset):
    """
    align the pairs of artefacts by id
    Args:
        dataset: a list of source or target artefacts tokenized with stopword removed.
    Returns:
        dataset: the list of artefatct aligned
    """
       
    for i in range(0,len(dataset)):
        dataset[i][0]=int(dataset[i][0])
    dataset=sorted(dataset, key=operator.itemgetter(0,1))
    
    for i in range(0,len(dataset)):
        dataset[i][0]=str(dataset[i][0])
        
    return dataset
    


