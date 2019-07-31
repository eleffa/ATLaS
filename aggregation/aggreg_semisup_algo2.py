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

import numpy as np
import operator
from nltk.tokenize import RegexpTokenizer
from functools import reduce
from sklearn.semi_supervised import LabelSpreading
import csv
from sklearn.neighbors import NearestNeighbors


tokenizer = RegexpTokenizer(r'\w+')

def alignIrmodel(irModel):
    """Read an IR model and align it with source and target ids.
  Args:
    irModel: the IR model to align.
  Returns:
    irModel_align: the IR model aligned.
  """     
    irModel_align=[]
    for i in range(0,len(irModel)):
        p=np.array([(irModel['X1'][i]),(irModel['X2'][i]),float(irModel['X3'][i])])
        irModel_align.append(p)
    
    irModel_align= sorted(irModel_align, key=operator.itemgetter(0,1))
    print("ok")
    return irModel_align

def read_similarity_score(lda,lsi,vsm):
    """Read an IR model and align it with source and target ids.
  Args:
    lda,lsi,vsm,emb: IR models to align.
  Returns:
    lsi_align, vsm_align, lda_align: IR models aligned.
  """             
  
    lda_align=alignIrmodel(lda)
    lsi_align=alignIrmodel(lsi)            
    vsm_align=alignIrmodel(vsm)
    
    
    return lsi_align, vsm_align, lda_align


def idFalseLinks(matrix_values):
    """ find the 20% of the lowest similarity measures 
    Args:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
  Returns:
    id_trueLinks_lsi, id_trueLinks_vsm, id_trueLinks_lda: id of 120% of the lowest  similarity measures of lsi, lda, vsm ir models """

    
    id_falseLinks_lsi=[]
    id_falseLinks_vsm=[]
    id_falseLinks_lda=[]
    
    length=len(matrix_values[:,0])
    fnum=round(length/20)
    
    lsi_values=np.array(matrix_values[:,0])
    id_lsi=np.argsort(lsi_values)
    
    
    vsm_values=np.array(matrix_values[:,1])
    id_vsm=np.argsort(vsm_values)
    
    lda_values=np.array(matrix_values[:,2])
    id_lda=np.argsort(lda_values)
    
    for i in range(0,fnum):
        id_falseLinks_lsi.append(id_lsi[i])
        id_falseLinks_vsm.append(id_vsm[i])
        id_falseLinks_lda.append(id_lda[i])
    
    return id_falseLinks_lsi, id_falseLinks_vsm, id_falseLinks_lda
    

def idTrueLinks(matrix_values):
    """ find the 10% of the highest similarity measures 
    Args:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
  Returns:
    id_trueLinks_lsi, id_trueLinks_vsm, id_trueLinks_lda: id of 10% of the highest similarity measures of lsi, lda, vsm ir models """
       
    id_trueLinks_lsi=[]
    id_trueLinks_vsm=[]
    id_trueLinks_lda=[]
    
    length=len(matrix_values[:,0])
    fnum=round(length/10)
    init=length-fnum
    
    lsi_values=np.array(matrix_values[:,0])
    id_lsi=np.argsort(lsi_values)
    
    
    vsm_values=np.array(matrix_values[:,1])
    id_vsm=np.argsort(vsm_values)
    
    lda_values=np.array(matrix_values[:,2])
    id_lda=np.argsort(lda_values)
    
    for i in range(init, length):
        id_trueLinks_lsi.append(id_lsi[i])
        id_trueLinks_vsm.append(id_vsm[i])
        id_trueLinks_lda.append(id_lda[i])
    
    return id_trueLinks_lsi, id_trueLinks_vsm, id_trueLinks_lda


def compValues(result):
    """extract only the probability in the ir model result 
    Args:
    result: IR models aligned
  Returns:
    values_r:  the probability of all a ir model"""
    
    values_r=[]
    for i in range (0,len(result)):
        values_r.append(result[i][2])
    return values_r

def building_aggreg_Matrix(lsi_align, vsm_align, lda_align, matrixSize):
    """ build the descriptor matrix for each pair of artefacts    
    Args:
    lsi_align, vsm_align, lda_align: IR models aligned;
    matrixSize: size of the descriptor matrix
  Returns:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
  """   
    
    
    matrix_values=np.zeros((len(lsi_align),matrixSize))
    
    lsi_values=compValues(lsi_align)
    vsm_values=compValues(vsm_align)
    lda_values=compValues(lda_align)    
   
    
    matrix_values[:,0]=lsi_values
    matrix_values[:,1]=vsm_values
    matrix_values[:,2]=lda_values
   
    
    return matrix_values
    


def buildingTrainingSet2(lsi_align, vsm_align, lda_align, numLine,matrixSize=3):
    """ automatically annotated the true and false links     
    Args:
    lsi_align, vsm_align, lda_align: IR models aligned;
    numLine: number of pairs of artefacts
    matrixSize: size of the descriptor matrix
  Returns:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
    vect: the answer vector (value 0 for false links and 1 for true links);
    matrix_values0 : sub set of the descriptor matrix based on links neighbourhood
  """  

    falseLinks=[]
    trueLinks=[]
    
    #building training set with IR model result
    matrix_values0=building_aggreg_Matrix(lsi_align, vsm_align, lda_align,matrixSize)
    #le calcul du laplacien etant impossible on construit une sous matice represntatrice 
    #de la grande nbre de lien est limité à 50000
    matrix_values,list_index_partitions =computeRandom(matrix_values0, numLine)
    
        
    lsi_align0=[]    
    
    
    for i in list_index_partitions:
        lsi_align0.append(lsi_align[i])
        
        
        
    #building training set with IR model result
    print("building training set with IR model")
    lsi_trueLinks_id, vsm_trueLinks_id, lda_trueLinks_id=idTrueLinks(matrix_values)
    lsi_falseLinks_id,vsm_falseLinks_id, lda_falseLinks_id =idFalseLinks(matrix_values)
    
    print('lsi')
    trueLinks.append(lsi_trueLinks_id)
    print('lda')
    trueLinks.append(lda_trueLinks_id)
    print('vsm')
    trueLinks.append(vsm_trueLinks_id)
    
    trueL=list(reduce(set.intersection, [set(item) for item in trueLinks ]))
    

    falseL=list(set().union(lda_falseLinks_id,vsm_falseLinks_id, lsi_falseLinks_id))
    
    
    print(len(falseL))
    print(len(trueL))
    
    #build the answers vector
    vect=np.zeros((len(lsi_align0),1))
    vect[:]=-1
    for i in falseL:
        vect[i]=0
        
    for i in trueL:
        vect[i]=1
    
    print("annoted_vectors_ok")
            
    return vect, matrix_values, matrix_values0

def partition_ker(neighbors_mat):
    """The neighborhood is defined in terms of Euclidean distance between the links descriptor vectors
    this function computes the neighborhood graph of the complete dataset"""
    
    nb_elements = neighbors_mat.shape[0]
    print(nb_elements)
    nb_neigh = neighbors_mat.shape[1]
    print(nb_neigh)
    numb = np.arange(0,nb_elements)
    numb = np.random.choice(numb,nb_elements,replace=False)              
    
    flag_remove = np.zeros((nb_elements,))
    list_output = []
    for i in range(0,nb_elements):
        if flag_remove[numb[i]]==0:
            list_output.append(numb[i])
            flag_remove[numb[i]] = 1
            flag_remove[neighbors_mat[numb[i],:]] = 1
    return list_output

def generate_partitions(desc_mat,downsampling_fact,nb_part):
    """the neighborhood graph is divided in multiple neighborhoods. The number of neighborhoods is
    equal to a defined down-sampling factor. When a link is randomly selected, its nearest neighbors are discarded. This
    operation is repeated until there is no link left."""
    
    nbrs = NearestNeighbors(n_neighbors=downsampling_fact+1, algorithm='ball_tree').\
    fit(desc_mat)
    dists, results = nbrs.kneighbors(desc_mat)
    partitions = []
    for i in range(0,nb_part):
        partitions.append(partition_ker(results[:,1:]))
    print("fin")
    return partitions


def computeRandom(matrix_values, numLine):
    """radomly choose a subset for the learning"""
    nb_part=1
    list_index_partition=generate_partitions(matrix_values,numLine, nb_part)
    subMatrix=matrix_values[list_index_partition]
            
    return subMatrix, list_index_partition[0]

    
    

def vote(allPrediction, length,numRan):   
    """find the average of all the result"""
    preds=np.zeros((length,1))
    for res in allPrediction:
        for i in range(0,len(res)):
            preds[i]=res[i]+preds[i]
    
    preds=preds/numRan
    
    print(preds)
           
    return preds


def computeSimilarities(vect,matrix_values):
    """ build the model with the semi supervised approach labelSpreading     
    Args:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
    vect: the answer vector (value 0 for false links and 1 for true links);
  Returns:
    preds: probability that a pair of artefact is linked
  """  
    model = LabelSpreading()
    computeModel=model.fit(matrix_values, vect)
    print("model built")
    preds = computeModel.predict_proba(matrix_values)
    print(preds)
    #print(len(preds))
    
    return preds[:,1]

def computeSimilarities2(lsi_align, vsm_align, lda_align, numLine,numRan=5):
    """ build the model with the semi supervised approach labelSpreading     
    Args:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
    vect: the answer vector (value 0 for false links and 1 for true links);
    numLine: number of pairs of artefacts
  Returns:
    preds: probability that a pair of artefact is linked
  """ 
    
    allPrediction=[]
    model = LabelSpreading()
    #np.memmap
    #compute multiple (10) random vector of the matrix_values 
    for i in range(0,numRan):
        subVect,subMatrix_values,matrix_values = buildingTrainingSet2(lsi_align, vsm_align, lda_align, numLine)
        #compute the prediction function of each random vector 
        print(len(subVect))
        print(len(subMatrix_values))
        computeModel=model.fit(subMatrix_values, subVect)
        print("new predicted function computed")
        #compute the prediction of each pair of artefact with the random model
        preds0 = computeModel.predict_proba(matrix_values)
        allPrediction.append(preds0[:,1])
        
    # by the "vote majoritaire"
    preds=vote(allPrediction, len(lsi_align),numRan)
    
    
    result=[]
    
    print(len(preds))
    
    for p in preds:
        for i in range(0, len(p)):
            result.append(p[i])
    
    
    print(len(result))
          
    return result

def compute_aggreg (pred_grd, lsi_align,filename ) :
    """ compute confidence measure for each pair of artefacts.
  Args:
    pred_grd: all probability between pairs of artefact  ;
    irmodel_align: an IR models aligneds;
    filename: file where the aggreg similarity scores results will be saved;
  Returns:
    None."""
    
    
    print("compute confident measure")
    
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len(lsi_align)):
            writer.writerow({'Artifact1': str("{0}".format(lsi_align[i][0])), 'Artifact2': str("{0}".format(lsi_align[i][1])), 'probability': str("{0}".format(pred_grd[i]))})  
       
    
    print("similarity matrix build")
    
   
def aggreg(lda,lsi,vsm, emb,filename):
    """ compute similarity between pairs of artefacts.
  Args:
    lda,lsi,vsm: ir model result  ;
    filename: file where aggreg similarity scores results will be saved;
  Returns:
    None."""
    #we fixe the randow matrix at 12000 line
    #numLine=12000
    numLine=2
    #matrixSize=9
    
    lsi_align, vsm_align, lda_align,emb_align =read_similarity_score(lda,lsi,vsm, emb)
    
        
    pred_grd=computeSimilarities2(lsi_align, vsm_align, lda_align, numLine)
    
    #pred_grd=computeSimilarities(vect, matrix_values)        
    compute_aggreg (pred_grd, lsi_align,filename )
    