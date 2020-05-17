What Is This?
-------------
Aggregation Trace Links Support (ATLaS) version 1.0.

This program is a prototype which role is to recover links between artefact (requirement, models elements) produce in the differents tools
of the collaborative space in order to check the consistency of data. It uses a semi-supervised method with a combination of similarities 
measure defined by IR techniques (VSM, LSI, LDA).

This is the code for the paper 

Requirements
------------
The code is written in python and requires numpy, pandas, pickle, sklearn, csv, gensim, keras, sklearn, matplotlib, stop_words,
										   math, scipy, and nltk. 


It also require some file available online :

	-glove  download the file glove.840B.300d.txt and put it on ATLaS\SIFmaster\data available on https://nlp.stanford.edu/projects/glove/
		    download the file glove.6B.300d.txt and put it on ATLaS\Helper  available on https://nlp.stanford.edu/projects/glove/
		
	-google download the file GoogleNews-vectors-negative300.bin.gzword2vec and put it on ATLaS\Helper on available on https://code.google.com/archive/p/word2vec/

	-ATLaS\SIFmaster is the code of Arora Sanjeev for the paper "A Simple but Tough-to-Beat Baseline for Sentence Embeddings".
	 Some functions/classes are based on this code. It requires numpy, scipy, pickle, sklearn, theano and the lasagne library. 

The inputs data sets are also available on online as describe below.

-ARC-IT 1 & 2 : The Architecture Reference for Cooperative and Intelligent Transportation (ARC-IT) includes a set 
of interconnected components that are organized into four views that focus on four different architecture perspectives. 
available on https://local.iteris.com/arc-it/html/architecture/architecture.html

-CM1-NASA : Data is provided courtesy of NASA collated by Jane Huffman Hayes. available on http://coest.org/

-EASYCLINIC : small student created dataset in english, italian containing diverses artifacts. Includes use case, intercation diagrams, 
test cases and class description. available on http://coest.org/
  
-HIPAA : HIPAA (Healthcare Insurance Portability and Accountability Act). Provides traceability from 10 HIPAA Technical safeguards 
to requirements in 10 different EHR systems. available on http://coest.org/

-ICEBREAKER : IBS  Ice Breaker System. Provides traceability form High level requirements to UML Classes. available on http://coest.org/
 

Get started
------------
execute the main.py file.


Source code
-----------
The code is separated into the following parts:

aggregation : it provides the code for the semi-supervised training
SIFmaster : this is the code of the code of Arora Sanjeev for the paper "A Simple but Tough-to-Beat Baseline for Sentence Embeddings".
feature : it provides the code for extraction of words, noun and verbal pharses and the computation of the defined similarity scores.
helpers : it provides dictionaries, stopword files and the architeture of VSM in tracelab.
inputs : it provides all the evaluated datasets.
outputs : it provides the results of the inputs data. 
irmodels : it provides all the information retrieval and machine learning models. VsmTracelab.py is the python version of the SEMERU Component Library.
utils : it provides utilisty functions

References
----------
For technical details and full experimental results, see the papers.

	-Semi-supervised Approach for Recovering Traceability Links in Complex Systems, 
	Conference Proceedings, 2018 23rd International Conference on Engineering of Complex Computer Systems (ICECCS), 
	Emma Effa Bella, Marie-Pierre Gervais, Reda Bendraou, Laurent Wouters and Ali Koudri.

	-ATLaS: A Framework for Traceability Links Recovery Combining Information Retrieval and Semi-supervised Techniques, 
	Conference Proceedings, 2019 23rd ieee international edoc conference - the enterprise computing conference, 
	Emma Effa Bella, Stephen Creff, Marie-Pierre Gervais and Reda Bendraou.

Acknowledgment
--------------
This research work has been carried out in the framework of the  Technological  Research Institute  SystemX,
and therefore granted with public funds within the scope of the French Program "Investissements d’Avenir".
