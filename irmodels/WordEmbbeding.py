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
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from stop_words import get_stop_words


# create English stop words list
stop_words = get_stop_words('en')

def load_Cbow(sourceTexts, TargetTexts):
    """Read source and target artefacts, build the CBOW model cand plot the semantic network with all the words.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    targetTexts: a list of target artefacts tokenized with stopword removed;
  Returns:
    model: the CBOW model computes.
    """
    
    # train model
    model = Word2Vec(sourceTexts, min_count=1,sg=0)
    model.save('outputs/model.bin')
    #Incremental Training
    model.build_vocab(TargetTexts, update=True)
    model.train(TargetTexts,total_examples=model.corpus_count,epochs=model.iter)
    # save model
    model.save('model.bin')
    model.build_vocab(sourceTexts, update=True)
    model.train(sourceTexts,total_examples=model.corpus_count,epochs=model.iter)
    #print(model)
    #print("dictionary")
    model.wv.save_word2vec_format('outputs/model_cbow.bin')
    
    #model = Word2Vec(sentence, min_count=1)
    print("model build")
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    ## create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    
    return model
    
    
    
def load_Skipgram(sourceTexts, TargetTexts):
    """Read source and target artefacts, build the SKIPGRAM model cand plot the semantic network with all the words.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    targetTexts: a list of target artefacts tokenized with stopword removed;
  Returns:
    model: the SKIPGRAM model computes.
    """
    # train model
    model = Word2Vec(sourceTexts, min_count=1,sg=1)
    model.save('outputs/model.bin')
    print(model)
    #Incremental Training
    model.build_vocab(TargetTexts, update=True)
    model.train(TargetTexts,total_examples=model.corpus_count,epochs=model.iter)
    # save model
    model.save('outputs/model.bin')
    print(model)
    model.wv.save_word2vec_format('outputs/model_skipgram.bin')
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    ## create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    return model
    