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
import gc
import csv


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

def read_similarity_score(lda,lsi,vsm, f1,f2, f3, f4, f5, f6):
    """Read an IR model and align it with source and target ids.
  Args:
    lda,lsi,vsm,f1,f2, f3, f4, f5, f6: IR models to align.
  Returns:
    lsi_align, vsm_align, lda_align,f1_align, f2_align, f3_align, f4_align, f5_align, f6_align: IR models aligned.
  """              
      
    lda_align=alignIrmodel(lda)
    lsi_align=alignIrmodel(lsi)
    vsm_align=alignIrmodel(vsm)    
    f1_align=alignIrmodel(f1)
    f2_align=alignIrmodel(f2)
    f3_align=alignIrmodel(f3)
    f4_align=alignIrmodel(f4)   
    f5_align=alignIrmodel(f5)
    f6_align=alignIrmodel(f6)    
    
    
    return lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align


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

def building_aggreg_Matrix(lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align, matrixSize):
    """ build the descriptor matrix for each pair of artefacts    
    Args:
    lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align: IR models aligned;
    matrixSize: size of the descriptor matrix
  Returns:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
  """   
    
    
    matrix_values=np.zeros((len(lsi_align),matrixSize))
    
    lsi_values=compValues(lsi_align)
    vsm_values=compValues(vsm_align)
    lda_values=compValues(lda_align)    
    f1_values=compValues(f1_align)
    f2_values=compValues(f2_align)
    f3_values=compValues(f3_align)
    f4_values=compValues(f4_align)
    f5_values=compValues(f5_align)
    f6_values=compValues(f6_align)
    
    matrix_values[:,0]=lsi_values
    matrix_values[:,1]=vsm_values
    matrix_values[:,2]=lda_values
    matrix_values[:,3]=f1_values
    matrix_values[:,4]=f2_values
    matrix_values[:,5]=f3_values
    matrix_values[:,6]=f4_values
    matrix_values[:,7]=f5_values
    matrix_values[:,8]=f6_values
    
    return matrix_values
    


def buildingTrainingSet(lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align,numLine,matrixSize=9):
    """ automatically annotated the true and false links     
    Args:
    lsi_align, vsm_align, lda_align: IR models aligned;
    matrixSize: size of the descriptor matrix
  Returns:
    matrix_values: descriptor matrix i.e. all the probability of all ir models
    vect: the answer vector (value 0 for false links and 1 for true links);
  """    
    
    trueLinks=[]
    
    #building training set with IR model result
    matrix_values=building_aggreg_Matrix(lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align,matrixSize)

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
    
    falseL=list(set().union(lda_falseLinks_id,vsm_falseLinks_id, lsi_falseLinks_id))
    trueL=list(reduce(set.intersection, [set(item) for item in trueLinks ]))
    
    del lda_falseLinks_id
    del vsm_falseLinks_id
    del lsi_falseLinks_id
    del trueLinks
    gc.collect()
    
    print(len(falseL))
    print(len(trueL))
    
    #build the answers vector
    vect=np.zeros((len(lsi_align),1))
    vect[:]=-1
    for i in falseL:
        vect[i]=0
        
    for i in trueL:
        vect[i]=1
    
    print("annoted_vectors_ok")
    
        
    return vect, matrix_values
 
def computeRandom(vect, matrix_values, numLine):
    """this function consists of building the subset by selecting the
    reference links of the heuristic and randomly adding to these
    links, a defined number of links that have not been chosen"""
    
    
    small_mat=[]

    trueLink=np.where(vect==1)[0]
    falseLink=np.where(vect==0)[0]
    unlabelLink=np.where(vect==-1)[0]
    
    
    numUnlabelLink=numLine-(len(trueLink)+len(falseLink))
    print(numUnlabelLink)
    randomLink=np.random.choice(unlabelLink, numUnlabelLink, replace=False)
    
    small_mat.extend(trueLink)
    small_mat.extend(falseLink)
    small_mat.extend(randomLink)
    subVect=vect[small_mat]
    subMatrix=matrix_values[small_mat]
        
    return subVect, subMatrix
    
    
    

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
    #print(preds)
    print(len(preds))
    
    return preds[:,1]

def computeSimilarities2(vect,matrix_values, numLine,numRan=2):
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
    
    #compute multiple (10) random vector of the matrix_values 
    for i in range(0,numRan):
        
        subVect, subMatrix_values= computeRandom(vect, matrix_values, numLine)
        #compute the prediction function of each random vector 
        print(len(subVect))
        print(len(subMatrix_values))
        computeModel=model.fit(subMatrix_values, subVect)
        print("new predicted function computed")
        #compute the prediction of each pair of artefact with the random model
        preds0 = computeModel.predict_proba(matrix_values)
        allPrediction.append(preds0[:,1])
        
    # by the "vote majoritaire"
    preds=vote(allPrediction, len(vect),numRan)
    
    
    result=[]
    
    print(len(preds))
    
    for p in preds:
        for i in range(0, len(p)):
            #print(p[i])
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
    
    
def aggreg(lda,lsi,vsm, f1,f2, f3, f4, f5, f6,filename):
    """ compute similarity between pairs of artefacts.
  Args:
    lda,lsi,vsm: ir model result  ;
    filename: file where aggreg similarity scores results will be saved;
  Returns:
    None."""
    
    #we fixe the randow matrix at 50000 line
    numLine=50000
    
    lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align =read_similarity_score(lda,lsi,vsm, f1,f2, f3, f4, f5, f6)
    vect,matrix_values = buildingTrainingSet(lsi_align, vsm_align, lda_align, f1_align, f2_align, f3_align, f4_align, f5_align, f6_align,numLine)
    
    del vsm_align
    del lda_align
    gc.collect()
    
    if (len(vect)<numLine):
        pred_grd=computeSimilarities(vect, matrix_values)    
    else:
        pred_grd=computeSimilarities2(vect, matrix_values, numLine)
     
    compute_aggreg (pred_grd, lsi_align,filename )
    