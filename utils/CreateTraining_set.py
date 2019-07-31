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
import pickle
import operator

def alignIrmodel(irModel):
    """Read an IR model and align it with source and target ids.
  Args:
    irModel: the IR model to align.
  Returns:
    irModel_align: the IR model aligned.
  """     
    irModel_align=[]
    for i in range(0,len(irModel)):
        p=np.array([int(irModel['X1'][i]),int(irModel['X2'][i]),float(irModel['X3'][i])])
        irModel_align.append(p)
    
    irModel_align= sorted(irModel_align, key=operator.itemgetter(0,1))
    print("ok")
    return irModel_align

def alignTraining_set(answers, irModel):
    """Read an ir model and align the true link answers with the ir model vector.
  Args:
    answers: the list of true links;
    irModel: a result of an ir model.
  Returns:
    vect: the answer vector (value 0 for false links and 1 for true links) 
  """   
    
    #build the answers vector
    vect=np.zeros((len(irModel),1))
    
    irModel_align=alignIrmodel(irModel)  
    
    irModel_list=[]
    for i in range(0,len(irModel_align)):
        p=((irModel_align[i][0]), (irModel_align[i][1]))
        irModel_list.append(p)
        
        
    list_answers=[]
    for j in range(0,len(answers)):
        p=(int(answers.values[j][0]), int(answers.values[j][1]))
        list_answers.append(p)
    
    
    #identify the position of the true link in the ir model vector and build the answer vector based on those information
    for i in range(0,len(answers)):
        x=irModel_list.index(list_answers[i]) if list_answers[i] in irModel_list else -1
        if(x!=-1):
            vect[x]=1

    return vect


def create_training_set (vect, nbTrueLinks=250, nbFalseLinks=750):
    """split the dataset into two set (training set and evaluation set)
        randomly choose a set of true and false links for the training
       the dataset do not have the same size so yiu can modify the nbtruelinks and nbfalseliks
       for example hipaa(120, 880), easyclinic (450, 750), icebreaker (250, 750), cm1-nasa(20, 200) 
  Args:
    vect: the answer vector (value 0 for false links and 1 for true links);
    nbTrueLinks: number of true links for the training
    nbFalseLinks: number of false links for the training
  Returns:
    none
  """ 
    train_data=[]
    #training data 250 true links and 500 false links
    trueL=np.where(vect==1)[0]
    falseL=np.where(vect==0)[0]
    
    train_data_true=np.random.choice(trueL, nbTrueLinks, replace=False)
    train_data_false=np.random.choice(falseL, nbFalseLinks, replace=False)
    
    train_data.extend(train_data_false)
    train_data.extend(train_data_true)
    pickle.dump(train_data, open("outputs/trainingSet.p","wb"))
        
    data_id=[i for i in range(0,len(vect)) if not i in train_data]    
    pickle.dump(data_id, open("outputs/evaluationSet.p","wb"))
        
    print("training file ok")
    
    
 
def create_link_class (vect,nbTrueLinks=250,nbFalseLinks=1500):
    """randomly choose a set of true and false links for the training
       the dataset do not have the same size so yiu can modify the nbtruelinks and nbfalseliks
       for example hipaa(100, 1200), easyclinic(250, 1500), icebreaker(150, 1500), cm1-nasa(15,150)
  Args:
    vect: the answer vector (value 0 for false links and 1 for true links);
    nbTrueLinks: number of true links for the training
    nbFalseLinks: number of false links for the training
  Returns:
    train_data_true:list of true links chose,
    train_data_false:list of false links chose.
  """ 
    train_data=[]
    #training data 250 true links and 500 false links
    trueL=np.where(vect==1)[0]
    falseL=np.where(vect==0)[0]
    
    train_data_true=np.random.choice(trueL, nbTrueLinks, replace=False)
    train_data_false=np.random.choice(falseL, nbFalseLinks, replace=False)
    return train_data_true, train_data_false