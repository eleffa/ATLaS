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
from __future__ import unicode_literals
from gensim import corpora, models, similarities
import numpy as np
import csv





def lsi(sourceTexts, source_id, len_source, targetTexts, target_id, len_target, filename):
    """Read source and target artefacts and compute similarity for each pair of artefacts.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    source_id: a list of source artefacts ids;
    len_source: number of source artefacts;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    target_id: a list of target artefacts ids;
    len_target: number of target artefacts;
    filename: file where the ir model result are saved.
  Returns:
    None.
    """
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(sourceTexts)
    #dictionary = corpora.Dictionary(reqtexts)
    # convert tokenized documents into a document-term matrix
    reqCorpus = [dictionary.doc2bow(text) for text in sourceTexts]
    tfidf = models.TfidfModel(reqCorpus)
    req_tfidf = tfidf[reqCorpus]
    
    numExec=10
    lsiResult=[]
    #Compute LSI for each pair of artefacts
    for i in range(0,numExec):
        res=computeSim(req_tfidf,dictionary,targetTexts,len_target,len_source)
        lsiResult.append(res)
    
    
    transposeVect=np.zeros((len_source,len_target))
    
    
    for res in lsiResult:
        for i in range(0,len_source):
            for j in range(0,len_target): 
                transposeVect[i,j]=transposeVect[i,j]+res[i,j]
    
    transposeVect=transposeVect/numExec
        
        
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(transposeVect[i,j]))})  
       
    print("similarity matrix build")
    
    
def computeSim(source_tfidf,dictionary,targetTexts,len_target,len_source):
    """compute similarity for each pair of artefacts.
  Args:
    source_tfidf: a list of source artefacts in the tfidf model;
    dictionary: set of words of sources and artefacts ;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    len_target: number of target artefacts;
    len_source: number of source artefacts;
  Returns:
    transposeVect: transpose of the similarity vector.
    """
    model_bow=[]
    model=[]
    sims=[]
    lsi = models.LsiModel(corpus=source_tfidf, id2word=dictionary, num_topics=3)
    # convert tokenized documents into a document-term matrix
    for mod in targetTexts:
        bow = dictionary.doc2bow(mod)
        model_bow.append(bow)
    # convert the model corpus to req LSI space
    for bow in model_bow :
        vec_model = lsi[bow]
        model.append(vec_model)
    print("end")
    
    
    #compute similarities between requirements and models
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[source_tfidf])
    # perform a similarity query for each model against the requirements corpus
    for vec_model in model:
        sim = index[vec_model]
        sims.append(list(enumerate(sim)))
        
    vect=np.zeros((len_target,len_source))
            
    for x in range(0, len(sims)):
        for y in sims[x]:
            vect[x,y[0]]=y[1]
    transposeVect=vect.T
    return transposeVect