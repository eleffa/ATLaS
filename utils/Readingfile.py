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
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re




# list for tokenized documents in loop
sourceTexts = []
targetTexts = []

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


def scrTxtFile (filename):
    """Read source data (requirements, models elements) from source txt files and put them into buckets.
       each line represents an artefact.
  Args:
    filename: path to the source files.
  Returns:
    source_set: a list of source artefacts with theirs ids.
  """
    source_set=[]
    file = open(filename, "r")
    for line in file:
        source_set.append(line)
    file.close()
    return source_set

def tgTxtFile(filename):
    """Read target data (requirements, models elements) from target txt files and put them into buckets.
  Args:
    filename: path to the target files.
  Returns:
    target_set: a list of target artefacts with theirs ids. 
  """
    target_set=[]
    file = open(filename, "r")
    for line in file:
        target_set.append(line)
    file.close()
    return target_set


def reqTxtFile (filenames):
    """Read source data (requirements, models elements) from source txt files and put them into buckets.
       Each artefact is tokenized and the stopword are removed in order to build the corpus.
  Args:
    filenames: path to the source files.
  Returns:
    sourceTexts:a list of source artefacts tokenized with stopword removed;
    source_ids_set: a list of source artefacts ids;
    len_source: number of source artefacts ;
    textNoId: a list of source artefacts tokenized without theirs ids;
    source_set: a list of source artefacts with theirs ids.
  """
    source_set=[]
    source_ids_set=[]
    #artefact_id=[]
    textNoId=[]
    file = open(filenames, "r")
    for ligne in file:
        source_set.append(ligne)
    file.close()
    len_source=len(source_set)
    #retrieve requirement id 
    for i in range(0,len(source_set)):
        artefact_id=tokenizer.tokenize(source_set[i])
        source_ids_set.append(artefact_id[0])
    # loop through requirements document list
    for i in source_set:
    # clean and tokenize document string
    # format input data
        i=format(i)
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
    # add tokens to list
    #texts.append(stemmed_tokens)
        length=len(stopped_tokens)+1
        sourceTexts.append(stopped_tokens)
        textNoId.append(stopped_tokens[1:length])
    return sourceTexts, source_ids_set, len_source, textNoId, source_set
    


def modTxtFile(filename):
    """Read target data (requirements, models elements) from target txt files and put them into buckets.
       Each artefact is tokenized and the stopword are removed in order to build the corpus.
  Args:
    filename: path to the source files.
  Returns:
    targetTexts:a list of target artefacts tokenized with stopword removed;
    target_ids_set: a list of target artefacts ids;
    len_target: number of target artefacts ;
    textNoId: a list of target artefacts tokenized without theirs ids;
    target_set: a list of target artefacts with theirs ids.
  """
    target_set=[]
    target_ids_set=[]
    modelNoId=[]
    #Reads in models document
    file = open(filename, "r")
    for line in file:
        target_set.append(line)
    file.close()
    len_target=len(target_set)
    
    for i in range(0,len(target_set)):
        artefact_id=tokenizer.tokenize(target_set[i])
        target_ids_set.append(artefact_id[0])
    
    # loop through models document list
    for i in target_set:
    # clean and tokenize document string
        # format input data
        i=format(i)
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

    # add tokens to list
        l=len(stopped_tokens)+1
        targetTexts.append(stopped_tokens)
        modelNoId.append(stopped_tokens[1:l])
    return targetTexts, target_ids_set, len_target, modelNoId, target_set


def readLinksTxt (filename):
    """Read source or target data (requirements, models elements) from txt files and put them into buckets.
  Args:
    filename: path to the txt files.
  Returns:
    data_set: a list of artefacts with theirs ids. 
  """
    data_set=[]
    file = open(filename, "r")
    for line in file:
        data_set.append(line)
    file.close()
    return data_set


def camelCaseSplitter(string):
    """split compose word into atomic word for example 'RemoteDriving' into 'Remote','Driving'
    or 'TMS_Driving_States' into 'TMS','Driving','States' using the regex '(?!^)([A-Z][a-z]+)'.    
  Args:
    string: the string to split.
  Returns:
    splitted: the splitted string 
    """
    splitted = re.sub('(?!^)([A-Z][a-z]+)', r' \1', string).split()
    return splitted


def format(line):
    """read each token of an artefact and split compose words if there are any and return the artefact with the tokens'.    
  Args:
    line: the artefact text.
  Returns:
    new_artefact_line: the artefact text with only atomic word
    """
    #new_artefact_line=''
    
    tokens=tokenizer.tokenize(line)
    new_artefact_line=tokens[0]
    
    for i in range(1, len(tokens)):
        word=tokens[i]
        listOfWord = camelCaseSplitter(word)
        if (len(listOfWord)>1):
            for w in listOfWord:
                w=re.sub('[^A-Za-z0-9]+', ' ', w)
                new_artefact_line=new_artefact_line + ' '+ w
        else:
            word=re.sub('[^A-Za-z0-9]+', ' ', word)
            new_artefact_line=new_artefact_line + ' ' + word
    return new_artefact_line
            
    