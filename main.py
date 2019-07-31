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
from utils import Readingfile, CreateTraining_set
from irmodels import Lsiwithgensim, WordEmbbeding, VsmTracelab,\
	 Ldawithgensim, Rnn_Cbow, Rnn_Skipgram, Cnn_Cbow, WordEmbcbow, WordEmbSkipgram
from aggregation import aggreg_semisup, Aggreg_features, Aggreg_final

#algo 2 is used for insdustrial/big datasets (ARC-IT1 & 2) to find a pertinent subset. It is used on a supercomputer
from aggregation import aggreg_semisup_algo2, Aggreg_features_algo2

from features import similarities, extractDict

import pandas
import time

 
# read the input documents 
""" choose the dataset you want to use
"""
dataset = {   0 : "inputs/IceBreakerText/requirements.txt",
              1 : "inputs/IceBreakerText/ClassDiagram.txt",
              2 : "inputs/IceBreakerText/Requirements2ClassMatrix.csv",
              3 : "outputs/IceBreakerText/"
             }
"""
dataset =    {0 : "inputs/CM1-NASA/CM1-sourceArtifacts.txt",
              1 : "inputs/CM1-NASA/CM1-targetArtifacts.txt",
              2 : "inputs/CM1-NASA/CM1-source2target_answers.csv",
              3 : "outputs/CM1-NASA/"
             }

dataset = {   0 : "inputs/EasyClinic/all_model_artifactsSrc.txt",
              1 : "inputs/EasyClinic/all_model_artifactsTg.txt",
              2 : "inputs/EasyClinic/allArtifact2allArtifact_answers.csv",
              3 : "outputs/EasyClinic/"
             }

dataset =     { 0 : "inputs/ARC-IT/requirements-functions/arc-it_requirements.txt",
                1 : "inputs/ARC-IT/requirements-functions/arc-it_functionalObjects.txt",
                2 : "inputs/ARC-IT/requirements-functions/arc-it_links-R-F.csv",
                3 : "outputs/ARC-IT/requirements-functions/"
             }

dataset = {   0 : "inputs/ARC-IT/need-requirements/arc-it_requirements.txt",
              1 : "inputs/ARC-IT/need-requirements/arc-it_needs.txt",
              2 : "inputs/ARC-IT/need-requirements/arc-it_links-R-N.csv",
              3 : "outputs/ARC-IT/ARC-IT/need-requirements/"
             }

dataset =    {0 : "inputs/HIPAA/HIPAA_Goal_Model.txt",
              1 : "inputs/HIPAA/all_requirements.txt",
              2 : "inputs/HIPAA/hipaa2goal_model_answers.csv",
              3 : "outputs/HIPAA/"
             }
"""


""" the purpose of this code is to find similarities between source and target artefacts 
"""

#reading source and target artefacts   
sourceArtefacts=Readingfile.readLinksTxt(dataset[0])
targetArtefacts=Readingfile.readLinksTxt(dataset[1])
sourceArtefactsTexts, sourceArtefacts_id, nbOfSourceArtefacts, sourceArtefactsTextsWithNoId, sourceArtefactsList=Readingfile.reqTxtFile (dataset[0])
targetArtefactsTexts, targetArtefacts_id, nbOfTargetArtefacts, targetArtefactsWithNoId, targetArtefactsList=Readingfile.modTxtFile(dataset[1])


"""compute the information retrieval similarities and the defined similarities based on advanced natural processing 
   techniques (embeddings)
"""


#compute information retrieval model VSM
vsmFilename=dataset[3]+'vsmResults.csv'
VsmTracelab.TLSimilarityMatrixCompute(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
									 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,vsmFilename)


#compute information retrieval model LSI
lsiFilename=dataset[3]+"lsiResults.csv"
Lsiwithgensim.lsi(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
									 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,lsiFilename)

#compute information retrieval model LDA
ldaFilename=dataset[3]+"ldaResults.csv"
Ldawithgensim.lda(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
									 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,ldaFilename)

###create and semantic models  ###
#compute word embbeding with CBOW model
cbowModel=WordEmbbeding.load_Cbow(sourceArtefactsTextsWithNoId, targetArtefactsWithNoId)

#compute word embbeding with SKIPGRAM model
skipgramModel=WordEmbbeding.load_Skipgram(sourceArtefactsTextsWithNoId, targetArtefactsWithNoId)


#compute move distance with word embedding Cbow model
cbowFilename=dataset[3]+'wordEmbCbowResults.csv'
WordEmbcbow.wordCbow(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
									 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,cbowFilename)

#compute move distance with word embedding Skipgram model
skipFilename=dataset[3]+'wordEmbSkipgramResults.csv'
WordEmbSkipgram.wordSkipgram(sourceArtefactsTextsWithNoId, sourceArtefacts_id, targetArtefactsWithNoId, targetArtefacts_id,skipFilename)


#build the term and pharses dictionnaries
dictTermFilename=dataset[3]+'dictionnary.csv'
dictPharseFilename=dataset[3]+'pharseDict.csv'
extractDict.dictionnary(sourceArtefactsTextsWithNoId, targetArtefactsWithNoId,dictTermFilename)
extractDict.allDictionnary(sourceArtefactsTextsWithNoId, targetArtefactsWithNoId, sourceArtefacts, targetArtefacts,dictPharseFilename)

#create the word-synonym file and phrase-synonym file
dictTerm=pandas.read_csv(dictTermFilename, names = ['term','synonym', 'probability'], header = 0) 
dictPharse=pandas.read_csv(dictPharseFilename, names = ['term','synonym', 'probability'], header = 0)  
synonymTermsFilename =dataset[3]+'synonymsterms.csv'
synonymPhraseFilename=dataset[3]+'synonymsPhrases.csv'
similarities.synonymFile(dictTerm,sourceArtefactsTextsWithNoId, targetArtefactsWithNoId, synonymTermsFilename)
similarities.synonymPhraseFile(dictPharse,sourceArtefacts, targetArtefacts, synonymPhraseFilename)

#compute similarities scores : f1,f2,f3,f4,f5,f6
#f1,f2,f3 are the similarities scores and f4,f5,f6 are the dissimilarities scores
f1Filename=dataset[3]+'simTerms.csv'
f2Filename=dataset[3]+'commonSystemTerms.csv'
f3Filename=dataset[3]+'simPharses.csv'
f4Filename=dataset[3]+'diffTerms.csv'
f5Filename=dataset[3]+'diffSystemTerms.csv'
f6Filename=dataset[3]+'diffPharses.csv'

#this computation can take some hours 
start = time.time()
#compute f1 and f3 similarities scores
similarities.simDiffTerm(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
						 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,f1Filename,f4Filename,synonymTermsFilename)

#compute f2 and f4 similarities scores
similarities.TFIDFsimDiffTerm(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts, \
							targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,f2Filename,f3Filename,synonymTermsFilename)

#compute f3 and f6 similarities scores
similarities.simDiffPharse(sourceArtefactsTextsWithNoId, sourceArtefacts_id, nbOfSourceArtefacts,\
						 targetArtefactsWithNoId, targetArtefacts_id, nbOfTargetArtefacts,f3Filename,f6Filename,synonymPhraseFilename)
end = time. time()
print(end - start)


"""first experimentation : recovering links with supervised approaches """
answers=pandas.read_csv(dataset[2],names = ['X1','X2'], header = 0)

vect=CreateTraining_set.alignTraining_set()
CreateTraining_set.create_training_set(vect)


#compute information retrieval model RNN+CBOW
rnnCbowFilename=dataset[3]+'rnnCbowResults.csv'
irmodel=dataset[3]+'lsiResults.csv' 
#TODO align rnn vector without a statistic model result
Rnn_Cbow.computeRnnCbow(sourceArtefactsTexts,targetArtefactsTexts,answers,irmodel,rnnCbowFilename)

#compute information retrieval model CNN+CBOW
cnnCbowFilename=dataset[3]+'cnncbowResults.csv'
Cnn_Cbow.computeCnnCbow(sourceArtefactsTexts,targetArtefactsTexts,cnnCbowFilename)

#compute information retrieval model RNN+Skipgram
rnnSkipFilename=dataset[3]+'rnnSkipgramResults.csv'
irmodel=dataset[3]+'lsiResults.csv'
#TODO align rnn vector without a statistic model result
Rnn_Skipgram.computeRnnSkipgram(sourceArtefactsTexts,targetArtefactsTexts,irmodel,answers,rnnSkipFilename)


"""second experimentation : recovering links with semi supervised approaches """

# read the Information retrieval model result
lda=pandas.read_csv(ldaFilename,names = ['X1','X2','X3'], header = 0)
lsi=pandas.read_csv(lsiFilename,names = ['X1','X2','X3'], header = 0)
vsm=pandas.read_csv(vsmFilename,names = ['X1','X2','X3'], header = 0)
emb=pandas.read_csv(cbowFilename,names = ['X1','X2','X3'], header = 0)
f1=pandas.read_csv(f1Filename, names = ['X1','X2','X3'], header = 0)
f2=pandas.read_csv(f2Filename, names = ['X1','X2','X3'], header = 0)
f3=pandas.read_csv(f3Filename, names = ['X1','X2','X3'], header = 0)
f4=pandas.read_csv(f4Filename, names = ['X1','X2','X3'], header = 0)
f5=pandas.read_csv(f5Filename, names = ['X1','X2','X3'], header = 0)
f6=pandas.read_csv(f6Filename, names = ['X1','X2','X3'], header = 0)



#aggregate all the information retrieval model 

#aggreg combinaision 1 only uses statistic measures lda,lsi,vsm
#with real answers
realvalueFilename="aggregResultsRealValues.csv"
Aggreg_final.aggreg(lda, lsi, vsm, answers, emb,realvalueFilename)

#with the proposed heuristic to automatically annotated the data 
combi1Filename="aggregResultsV40"
aggreg_semisup.aggreg(lda,lsi,vsm,combi1Filename)

#algo for industrial/big datasets uses on the supercomputer
#combi1Filename1="aggregResultsV4"
#aggreg_semisup_algo2.aggreg(lda,lsi,vsm,combi1Filename1)

#aggreg combinaision 2 uses all the 3 statistic measures (lda,lsi,vsm) and the 6 defined similarity scores
combi2Filename="aggregResultsV50"
Aggreg_features.aggreg(lda, lsi, vsm, f1, f2, f3, f4, f5, f6,combi2Filename)

#algo for insdustrial/big datasets uses on the supercomputer
#combi2Filename2="aggregResultsV5"
#Aggreg_features_algo2.aggreg(lda, lsi, vsm, f1, f2, f3, f4, f5, f6,combi2Filename2)