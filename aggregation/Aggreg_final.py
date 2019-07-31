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
import pandas
import operator
from nltk.tokenize import RegexpTokenizer
from utils import CreateTraining_set
from sklearn.semi_supervised import LabelSpreading
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

def read_similarity_score(lda, lsi, vsm, answers, emb ):
    """Read an IR model and align it with source and target ids.
  Args:
    lda,lsi,vsm,emb: IR models to align.
    answers: the list of true links;
  Returns:
    lsi_align, vsm_align, lda_align,emb_align: IR models aligned.
    vect : the answer vector (value 0 for false links and 1 for true links);
  """       
    #build the answers vector
    vect=np.zeros((len(lsi),1))
    
 
    lda_align=alignIrmodel(lda)
    lsi_align=alignIrmodel(lsi)      
    vsm_align=alignIrmodel(vsm)                        
    emb_align=alignIrmodel(emb)
    
    
    lsi_list=[]
    for i in range(0,len(lsi_align)):
        p=((lsi_align[i][0]), (lsi_align[i][1]))
        lsi_list.append(p)
        
    list_answers=[]
    for j in range(0,len(answers)):
        p=(int(answers.values[j][0]), int(answers.values[j][1]))
        list_answers.append(p)
    
    
    
    for i in range(0,len(answers)):
        x=lsi_list.index(list_answers[i]) if list_answers[i] in lsi_list else -1
        if(x!=-1):
            vect[x]=1
    
    print("annoted_vectors_ok")
    
    
    return vect, lsi_align, vsm_align, lda_align, emb_align



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

def compute_confident_measure (vect,lsi_align,lda_align, vsm_align,emb_align,matrixSize=4):
    """ compute confidence measure for each pair of artefacts.
  Args:
    vect: the answer vector (value 0 for false links and 1 for true links);
    lsi_align, vsm_align, lda_align: IR models aligned;
    filename: file where the aggreg similarity scores results will be saved;
  Returns:
     preds: probability that a pair of artefact is linked."""
    
    
    
    vect0=np.zeros((len(lsi_align),1))
    vect0[:]=-1
    
    trueL, falseL=CreateTraining_set.create_link_class(vect, lsi_align)
    print(len(trueL))
    print(len(falseL))
    for i in falseL:
        vect0[i]=0
        
    for i in trueL:
        vect0[i]=1
    
    print("annoted_vectors_ok")
    
        

    emb_values=compValues(emb_align)
    vsm_values=compValues(vsm_align)
    lda_values=compValues(lda_align)
    lsi_values=compValues(lsi_align)
    
    matrix_values=np.zeros((len(lsi_align),matrixSize))
    
    
    matrix_values[:,0]=lda_values
    matrix_values[:,1]=emb_values
    matrix_values[:,2]=vsm_values
    matrix_values[:,3]=lsi_values
    
    model = LabelSpreading()
    computeModel=model.fit(matrix_values, vect0)
    print("models built")
    preds = computeModel.predict_proba(matrix_values)
    
    
    print(preds)
    
    
    return preds[:,1] 
    
    
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



def aggreg(lda, lsi, vsm, answers, emb,filename):
    """ compute similarity between pairs of artefacts.
  Args:
    lda,lsi,vsm: ir model result  ;
    filename: file where aggreg similarity scores results will be saved;
  Returns:
    None."""
    vect, lsi_align, vsm_align, lda_align, emb_align = read_similarity_score(lda, lsi, vsm, answers, emb)
    pred_grd = compute_confident_measure (vect,lsi_align,lda_align, vsm_align,emb_align)
    compute_aggreg (pred_grd, lsi_align,filename )
    