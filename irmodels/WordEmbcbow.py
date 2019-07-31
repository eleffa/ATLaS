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
from __future__ import unicode_literals
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
import os
from stop_words import get_stop_words
import csv

# create English stop words list
stop_words = get_stop_words('en')

def wordCbow(sourceTexts, source_id, len_source, targetTexts, target_id, len_target,filename):
    """Read source and target artefacts and compute word move distance similarity for each pair of artefacts.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    source_id: a list of source artefacts ids;
    len_source: number of source artefacts
    targetTexts: a list of target artefacts tokenized with stopword removed;
    target_id: a list of target artefacts ids;
    len_target: number of target artefacts
    filename: file where the ir model result are saved.
  Returns:
    None.
    """
    allwords=[]
    for i in sourceTexts:
        allwords.append(i)
    for j in targetTexts:
        allwords.append(j)
      
    
    if not os.path.exists('helpers/GoogleNews-vectors-negative300.bin.gz'):
        raise ValueError("SKIP: You need to download the google news model and put it on helpers directory")
    
    model = KeyedVectors.load_word2vec_format('helpers/GoogleNews-vectors-negative300.bin.gz', binary=True)
    instance = WmdSimilarity(sourceTexts, model)   
    
        
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for i in range(0,len_source):
            for j in range(0,len_target):
                print(i)
                sim=model.wmdistance(sourceTexts[i], targetTexts[j]) 
                writer.writerow({'Artifact1': str("{0}".format(source_id[i])), 'Artifact2': str("{0}".format(target_id[j])), 'probability': str("{0}".format(sim))})  
       

    
    
    print("similarity matrix build")
    
    