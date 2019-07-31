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
from xlwt import Workbook
from utils import ReadingWordEmb
import os
from utils import CreateTraining_set
import pickle
import h5py
from keras.layers import Dense
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import average_precision_score as average_precision
import collections
import csv



DataSet = collections.namedtuple('DataSet', ['artifact1',  'artifact2','labels'])

   

def computeCnnCbow(texts, modeltexts, cnnfilename):
    """Read source and target artefacts and built the RNN CBOW model and compute similarity for each pair of artefacts.
  Args:
    sourceTexts: a list of source artefacts tokenized with stopword removed;
    targetTexts: a list of target artefacts tokenized with stopword removed;
    answers: list of true links;
    irmodel: a statistic model result(LSI,LDA or VSM);
    #TODO align rnn vector without a statistic model result
    filename: file where the ir model result are saved.
  Returns:
    None.
    """
    
    #align data by id
    texts=ReadingWordEmb.align_data(texts)
    modeltexts=ReadingWordEmb.align_data(modeltexts)       
    
    
    allwords=[]
    for i in texts:
        allwords.append(i)
    for j in modeltexts:
        allwords.append(j)
        
    ###prepare embedding model for the cnn model  for text in source artifact
    #preprocessing text of artifact  
        
    artifact_pairs = [(x,y) for x in texts for y in modeltexts] 
    
    #print (" ".join([" ".join(x) for x in artifact_pairs]))
    lstoken=[]
    for x in artifact_pairs:
        p=" ".join([" ".join(p) for p in x])        
        lstoken.append(p)
    
              
    print("all artifacts merged")
     
    embedding_dim=100
    channels = 2
     
    length = []
    for x in lstoken:
        length.append(len(x.split()))
        
    maxlen=max(length)
    
          
    #all vectors are put to the same size
    maxlen = maxlen + 5
     
    #load cbow model
    model_cbow = ReadingWordEmb.load_Cbow_word2vec(allwords, embedding_dim)   
       
    #build the answers vector  
    annoted_vect=CreateTraining_set.alignTraining_set()    
    
    #building training set
    if(os.path.exists("outputs/trainingSet.p")):
        print("training set already created")
        print("evaluation set already created")
    else:
        CreateTraining_set.create_training_set(annoted_vect)  
        
    all_dataset=[]
        
    for i in range(0,len(artifact_pairs)):
        p=np.array([artifact_pairs[i][0],artifact_pairs[i][1],annoted_vect[i]])
        all_dataset.append(p)
        
            
    # retrieved training data
    train_indices=pickle.load(open("outputs/trainingSet.p","rb"))
    train_indices.sort()    
    #train_labels=np.array(annoted_vect)[train_indices,:]        
      
    # retrieved evaluated data for testing the model
    test_indices=pickle.load(open("outputs/evaluationSet.p","rb"))
    #test_indices.sort()
    #test_labels=np.array(annoted_vect)[test_indices,:]
    
    train_data=np.array(all_dataset)[train_indices,:]
    test_data=np.array(all_dataset)[test_indices,:]
    
    
    TEST_SPLIT = 1500 
    #splitting test data in multiple files of length=1500   
    test_chunks = [test_data[x:x+TEST_SPLIT] for x in range(0, len(test_data), TEST_SPLIT)]
    #label_chunks= [test_labels[x:x+TEST_SPLIT] for x in range(0, len(test_labels), TEST_SPLIT)] 
    
    
    
    
    #n_sentences=len(artifact_pairs)
    #n_sentences=10
    if not os.path.exists('outputs/cnnfiles/cbow/training.h5py'):
        builtDecompFile('outputs/cnnfiles/cbow/training.h5py',train_data,model_cbow, maxlen,embedding_dim,channels)
        for i in range(0,len(test_chunks)):
            filename='outputs/cnnfiles/cbow/test'+str(i)+'.h5py'
            builtDecompFile(filename,test_chunks[i],model_cbow, maxlen,embedding_dim,channels)   
    
    
    # hyper parameters from Wang et al. (2016):
    FILTER_SIZES = [1, 2, 3]
    NUM_FEATURE_MAPS = 500
    KEEP_PROB = 1.0
    STOP_AFTER = 5 
    
    print('building the graph')
    # placeholders
    input_1 = tf.placeholder(tf.float32, [None,embedding_dim, maxlen, channels], name='input_1')
    input_2 = tf.placeholder(tf.float32, [None,embedding_dim, maxlen, channels], name='input_2')
    cnn_labels = tf.placeholder(tf.float32, [None], name='cnn_labels')
   
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # building the graph
    logits,prob_scores = inference(input_1, input_2, keep_prob,FILTER_SIZES,NUM_FEATURE_MAPS,embedding_dim, channels, maxlen)
    losses = loss(prob_scores, cnn_labels)
    train_op = training(losses)
    saver = tf.train.Saver()
    bestdev_model_file = Path('outputs/models', 'best-dev_model.ckpt').as_posix()
    
   
    best_map = 0.0
    last_improvement = 0
    EPOCHS=20
    all_scores=[]  
    placeholders = (input_1, input_2, keep_prob)
    num_train_data=len(train_data)
    
    init = tf.global_variables_initializer()
    trainfile=h5py.File('outputs/cnnfiles/cbow/training.h5py', 'r')    
    #decompfile=h5py.File('decomp.h5py', 'r')
    
    
    with tf.Session() as sess:
        sess.run(init)
        
        step = 0
        
        for batch in question_epochs(trainfile, EPOCHS, num_train_data):
            step += 1
            #print('step:', step)    
            feed_train = { input_1:batch.artifact1, input_2:batch.artifact2, cnn_labels:batch.labels, keep_prob:KEEP_PROB}
            #print('training feed ok')
            
            _, loss_value = sess.run([train_op, losses], feed_dict=feed_train)
            print(loss_value)         
            
            if step % 10 == 0:
                placeholders = (input_1, input_2, keep_prob)
                map_dev, train_scores = compute_metrics(sess, prob_scores, placeholders, trainfile, num_train_data)
                print('# Step %d, MAP(dev_train): %.3f' % (step, map_dev))
                last_improvement += 1
                print('last improvement: %d' %(last_improvement))
                if map_dev > best_map:
                    improvement = map_dev - best_map
                    best_map = map_dev
                    last_improvement = 0
                    #train_scores=all_score
                    saver.save(sess, bestdev_model_file)
                    print('Model saved (improved: %.3f)' % improvement)

            if  last_improvement >= STOP_AFTER:
                print('Early stopping...')
                break
               
        print('Restoring best dev model...')
        saver.restore(sess, bestdev_model_file)
        
        for i in range(0, len(test_chunks)):
            filename='outputs/cnnfiles/cbow/test'+str(i)+'.h5py'
            testfile=h5py.File(filename, 'r')
            num_test_data=len(test_chunks[i])
            map_dev_test,  test_scores = compute_metrics(sess,  prob_scores, placeholders, testfile,num_test_data)
            print('# MAP(dev_test): %.3f' % ( map_dev_test))
            all_scores.append(test_scores)
       
        
    
    sess.close()
    
    
    
    print(len(train_scores))
    print(len(all_scores))
    
        
    results_train = [(x[0], y[0], z) for (x, y), z in zip(train_data, train_scores)]
    
    results_test = [(x[0], y[0], z) for (x, y), z in zip(test_data, all_scores)]
    
    results = results_train + results_test
    
    print("CNN Cbow model compute")
    
    
    #creation of the csv file
    with open(cnnfilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for res in results:
            writer.writerow({'Artifact1': str("{0}".format(res[0])), 'Artifact2': str("{0}".format(res[1])), 'probability': str("{0}".format(res[2]))})  
       

    print("similarity matrix build")
    
    

    
def vectorize(sentence, embeddings, size_limit, embedding_dim, ignore_case=False):
    """Creates a vectorized representation (embedding matrix) of a sentence."""
    if ignore_case:
        sentence = sentence.lower()
    tokens = [w for w in sentence.strip().split() ]
    tokens = tokens[:size_limit]  # truncating sentence
    
    vectors = []
    for t in tokens:
        if t in embeddings:
            vectors.append(embeddings[t])
        else:  # OOV word
            random_vector = np.random.uniform(-0.25, 0.25, size=embedding_dim)
            vectors.append(random_vector)
    vectorized = np.array(vectors).T  # dimension lines x len(tokens) columns
    
    return vectorized

    
def compute_attention(question_matrix, sentence_matrix):
    """Computes the Attention Matrix of `question_matrix` (S) and
    `sentence_matrix` (T).
    """
    # computing unit vectors
    norms = np.linalg.norm(question_matrix, axis=0)
    norms = norms[np.newaxis, :]
    qmat = question_matrix / norms

    # computing unit vectors
    norms = np.linalg.norm(sentence_matrix, axis=0)
    norms = norms[np.newaxis, :]
    smat = sentence_matrix / norms

    # With unit vectors, the cosine similarity between every word vector can
    # be computed as a series of dot products, implemented as
    # matrix multiplication
    attention = np.matmul(qmat.T, smat)
    
    #print("Computes the Attention Matrix done")
    return attention


def semantic_match(sentence, attention_vec):
    """Computes the semantic matching between the i-th word of a sentence (s_i)
    and the words of a target `sentence` (T).

    Implements the strategy local-w from Wang et al. (2016, sec.3.1), with w=3.

    :param sentence: a sentence matrix of shape (dim, n_tokens).
    :param attention_vec: a row (or column) from the attention matrix
    containing similarity scores of the i-th word with all the words from
    `sentence`.

    :returns: a semantic matching vector of shape (dim,), where dim is the
    embedding dimension.
    """
    w = 3  # window size
    dimension, n_tokens = sentence.shape
    k = np.argmax(attention_vec)  # position k, of the most similar word
    window = range(max(0, k-w), min(k+w, n_tokens))  # window centered on k
    vector = np.zeros(dimension)
    for i in window:
        vector += attention_vec[i] * sentence[:, i]
    #print("Computes the semantic matching")
    return vector


def decompose(word_vec, match_vec, method='linear'):
    """Decompose `word_vec` (s_i) in two channels (similar, dissimilar) using
    the `match_vec` (s^_i).

    Implements decomposition methods from Wang et al. (2016, sec.3.2).
    """
    assert method in ('orthogonal', 'linear')

    if method == 'orthogonal':
        # About the formulas: https://en.wikipedia.org/wiki/Vector_projection#Vector_rejection  # noqa
        alpha = np.dot(word_vec, match_vec) / np.dot(match_vec, match_vec)
        positive = alpha * match_vec  # parallel to match_vec
        negative = word_vec - positive  # perpendicular
    elif method == 'linear':
        # cosine
        alpha = np.dot(word_vec, match_vec) / ( np.linalg.norm(word_vec) * np.linalg.norm(match_vec))
        positive = alpha * word_vec
        negative = (1.0 - alpha) * word_vec
    
    return positive, negative


def decomp_channels(question, sentence, maxlen,):
    """Decomposes `question` and `sentence` (both vectorized sentences) in
    two-channel matrices.

    :param question: vectorized representation of a question, generated by
    :function:vectorize. It's a matrix of shape (`dim`, n_tokens), where
    1 <= n_tokens <= MAX_SIZE.
    :param sentence: vectorized representation of a sentence, like `question`.

    :returns: a tuple of two arrays of shape (`dim`, MAX_SIZE, `channels`),
    where MAX_SIZE is the maximum number of tokens allowed for that each
    sentence (either MAX_QUESTION or MAX_SENTENCE). Channel 0 will hold
    positive vectors and channel 1 the negative.
    """
    assert question.shape[0] == sentence.shape[0]
    dim = question.shape[0]
    channels = 2

    ch_question = np.zeros((dim, maxlen, channels), dtype=np.float32)
    ch_sentence = np.zeros((dim, maxlen, channels), dtype=np.float32)

    # decomposition: similar and dissimilar channels
    attention = compute_attention(question, sentence)
    for i in range(question.shape[1]):
        word = question[:, i]  # word vector
        match_vec = semantic_match(sentence, attention[i, :])
        pos, neg = decompose(word, match_vec)
        ch_question[:, i, 0] = pos  # channel 0: positive
        ch_question[:, i, 1] = neg  # channel 1: negative

    for i in range(sentence.shape[1]):
        word = sentence[:, i]  # word vector
        match_vec = semantic_match(question, attention[:, i])
        pos, neg = decompose(word, match_vec)
        ch_sentence[:, i, 0] = pos  # channel 0: positive
        ch_sentence[:, i, 1] = neg  # channel 1: negative
    

    return ch_question, ch_sentence

def conv_filter(filter_size, embedding_size, in_channels, NUM_FEATURE_MAPS):
    """Creates a convolutional filter."""
    
    filter_shape = [embedding_size, filter_size, in_channels, NUM_FEATURE_MAPS]
    weights = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[NUM_FEATURE_MAPS]), dtype=tf.float32)
        
    print("decomposition convolutional filter created")
    
    return weights, biases


def conv_layer(in_data, FILTER_SIZES,NUM_FEATURE_MAPS,embedding_size,in_channels,sequence_length):
    """Apply a set of convolutional filters over input data.

    :param in_data: a tensor of shape [batch, EMBEDDING_SIZE,
    `sequence_length`, IN_CHANNELS]
    :returns: a tensor of shape [batch, total_feature_maps]
    
    
    if hasattr(in_data, 'shape'):  # a Numpy array
        embedding_size = in_data.shape[1]
        sequence_length = in_data.shape[2]
        in_channels = in_data.shape[3]
    elif hasattr(in_data, 'get_shape'):  # a TensorFlow placeholder
        embedding_size = in_data.get_shape()[1].value
        sequence_length = in_data.get_shape()[2].value
        in_channels = in_data.get_shape()[3].value
    else:
        raise TypeError('in_data must be either a Numpy array or a tensorFlow placeholder')
    
    
    print("reading input data done")
    """
   

    # convolutional layer
    pooled_outputs = []
    for filter_size in FILTER_SIZES:
        weights, biases = conv_filter(filter_size, embedding_size, in_channels, NUM_FEATURE_MAPS)
        conv = tf.nn.conv2d(in_data, weights, strides=[1, 1, 1, 1], padding='VALID')
        feature_map = tf.tanh(tf.nn.bias_add(conv, biases))
        # feature_map = tf.nn.relu(tf.nn.bias_add(conv, biases))
        width_conv = sequence_length - filter_size + 1
        pooled = tf.nn.max_pool(feature_map, ksize=[1, 1, width_conv, 1],strides=[1, 1, 1, 1], padding='VALID')
        # pooled.shape = [batch, 1, 1, NUM_FEATURE_MAPS]
        pooled_outputs.append(pooled)
        
    
    # concatenating feature maps
    features = tf.concat(pooled_outputs,3)
    total_feature_maps = len(FILTER_SIZES) * NUM_FEATURE_MAPS
    features_flat = tf.reshape(features, [-1, total_feature_maps])
    # features_flat.shape = [batch, total_feature_maps]
    print("convolutional layer done")
    
    return features_flat


def hidden_layer(features, keep_prob, FILTER_SIZES,NUM_FEATURE_MAPS ):
    """Build the fully-connected layer.
    """
    NUM_CLASSES = 1
    input_neurons = 2 * len(FILTER_SIZES) * NUM_FEATURE_MAPS

    weights = tf.Variable(tf.truncated_normal([input_neurons, NUM_CLASSES], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]),  dtype=tf.float32)
    features_drop = tf.nn.dropout(features, keep_prob=keep_prob)
    scores = tf.nn.bias_add(tf.matmul(features_drop, weights), biases)
    scores_flat = tf.reshape(scores, [-1])
    preds = Dense(1, activation='sigmoid')(scores)
    prob_scores = tf.nn.softmax(scores_flat)
    
    print("fully-connected layer Built")
    #print(prob_scores)
    return prob_scores,  preds


def inference(questions, sentences, keep_prob,FILTER_SIZES,NUM_FEATURE_MAPS,embedding_size,in_channels,sequence_length):
    """Build the composition model (CNN).
    """
    question_features = conv_layer(questions,FILTER_SIZES,NUM_FEATURE_MAPS,embedding_size,in_channels,sequence_length)
    sentence_features = conv_layer(sentences,FILTER_SIZES,NUM_FEATURE_MAPS,embedding_size,in_channels,sequence_length)
    features = tf.concat([question_features, sentence_features],1)
    prob_scores, preds = hidden_layer(features, keep_prob,FILTER_SIZES,NUM_FEATURE_MAPS)
    print("composition model (CNN) built")
    #print(scores)
    return prob_scores, preds

def get_proba(scores, sess, feed_dict,labels):
    
    proba=sess.run([scores], feed_dict=feed_dict)

    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print ("Accuracy:", accuracy.eval({scores, labels}))
    
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    return proba
    


def loss(logits, labels):
    """Computes the loss from the logits and the labels.

    :param logits: scores predicted by the model - [batch,]
    :param labels: expected labels - [batch,]
    """
    # cross entropy
    expected = tf.nn.softmax(labels)
    loss = tf.reduce_mean(-tf.reduce_sum(expected * tf.log(logits)))
    #loss = tf.reduce_mean(-tf.reduce_sum(expected * tf.log(logits), reduction_indices=1))                      
    #loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    #print(logits)
    return loss


def training(loss):
    tf.summary.scalar('loss', loss)
    # optimizer
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def compute_metrics(sess, preds, placeholders, data_file, num_test_data,exporter=None):
    """Compute metrics MAP and MRR over a dataset.

    :param sess: TensorFlow session
    :param logits_op: an operation that returns the scores for a given set of sentences
    :param placeholders: placeholders defined for `logits_op`
    :data_file: a HDF5 file object holding the dataset

    :returns: the values of MAP and MRR as a tuple: (MAP, MRR)
    """
    questions_ph, sentences_ph, keep_prob_ph = placeholders

    if exporter is None:
        exporter = no_op()
    next(exporter)  # priming the coroutine
    
    all_score=[]
    total_avep = 0.0
    
    n_questions = 0
    for batch in question_batches(data_file,num_test_data):
        feed_dict = {
            questions_ph: batch.artifact1,
            sentences_ph: batch.artifact2,
            keep_prob_ph: 1.0
        }
        scores = preds.eval(session=sess, feed_dict=feed_dict)
        exporter.send(scores)
        
        proba=get_proba(preds, sess, feed_dict, batch.labels)
        
        all_score.append(proba)
        
    
    n_questions += 1
    avep = average_precision(batch.labels, scores)
    print(avep)
    #print(scores)
    #print(len(scores))
    #print(len(batch.labels))
    #print(n_questions)
    total_avep += avep
    
       
    exporter.close()
    
    

    mean_avep = total_avep / n_questions
    
    return mean_avep, all_score


def question_batches(data_file,n_questions):
    """Iterates over a dat_file dataset returning batches composed by a all the training data 
    single question and its candidate answers.

    :data_file: a HDF5 file object holding the dataset
    :returns: a DataSet namedtuple of arrays (questions, sentences, labels).
    """
    
    questions_ds = data_file['artifact1']
    sentences_ds = data_file['artifact2']
    labels_ds    = data_file['labels']
    train_questions=[]
    train_sentences=[]
    train_labels=[]
    
    for rows in range(0, n_questions):        
        #print('rows= %d' %(rows))
        #labels = data_file['labels/q%d' % rows][...]
        questions = questions_ds[rows]
        sentences = sentences_ds[rows]
        labels    = labels_ds[rows]
        train_questions.append(questions)
        train_sentences.append(sentences)
        train_labels.append(labels)
        
    #print('load all %d the training data'%(n_questions))    
    yield DataSet(train_questions, train_sentences, train_labels)
        
        
def builtDecompFile(filename,artifact_pairs,model_cbow, maxlen,embedding_dim,channels):
    hdf = h5py.File(filename, mode='w') 
    tab_art1=[]
    tab_art2=[]
    labels=[]
    n_sentences=len(artifact_pairs)
    
    #grp_labels=hdf.create_group('labels')
    for i in range(0,len(artifact_pairs)):
        x1 = " ".join(artifact_pairs[i][0])
        x2 = " ".join(artifact_pairs[i][1])
        
        artifact1 = vectorize(x1,  model_cbow, maxlen,embedding_dim, ignore_case="TRUE")
        artifact2 = vectorize(x2,  model_cbow, maxlen,embedding_dim, ignore_case="TRUE")    
        # channel decomposition
        ch_artifact1, ch_artifact2 = decomp_channels(artifact1, artifact2,maxlen)
        tab_art1.append(ch_artifact1)
        tab_art2.append(ch_artifact2)
        labels.append(artifact_pairs[i][2])
        
    ds_artifact1=hdf.create_dataset('artifact1', data=tab_art1, dtype=np.float32, shape=(n_sentences, embedding_dim, maxlen, channels))
    ds_artifact2=hdf.create_dataset('artifact2', data=tab_art2, dtype=np.float32, shape=(n_sentences, embedding_dim, maxlen, channels))
    ds_label=hdf.create_dataset('labels', data=labels, dtype=np.float32, shape=(n_sentences, ))      
    #grp_labels = labels
    hdf.close()
    #print(ds_artifact1.shape)
    print("computing decomposition done")
    print("h5py file created")


def question_epochs(data_file, n_epochs,n_questions):
    """Iterates over a dataset for `n_epochs` epochs.
    """
    for i in range(0, n_epochs):
        yield from question_batches(data_file,n_questions) 

def no_op(*args, **kwargs):
    """A convenience no-op coroutine to avoid `if is not None` checks.
    """
    dummy = None
    while True:
        dummy = yield dummy