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

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from utils import ReadingWordEmb
from keras.models import Model
import time
import os
from utils import CreateTraining_set
import pickle
import csv



def computeRnnCbow(sourceTexts, targetTexts, answers,irmodel,filename):
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
    #load cbow model
    tokenizer, embedding_matrix, sequences, maxlen, num_words, embedding_dim, artifact_pairs = ReadingWordEmb.compute_Cbow(sourceTexts, targetTexts)
    
    artifacts1 = [x[0] for x in artifact_pairs]
    artifacts2 = [x[1] for x in artifact_pairs]
    sequences_1 = tokenizer.texts_to_sequences(artifacts1)
    sequences_2 = tokenizer.texts_to_sequences(artifacts2)
    #compute the number of common words (syntaxic) for each pair of artifacts
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(sequences_1, sequences_2)]
    

    #we padded all our sentences to have the same length  
    padded_data_1 = pad_sequences(sequences_1, maxlen=maxlen)
    padded_data_2 = pad_sequences(sequences_2, maxlen=maxlen)
    leaks = np.array(leaks)
    
    print(len(padded_data_1))
    print(len(padded_data_2))
    
    
    #build the answers vector
    labels=[]
    
    labels=CreateTraining_set.alignTraining_set(answers,irmodel)    
    #building training set
    if(os.path.exists("outputs/trainingSet.p")):
        print("training set already created")
        print("evaluation set already created")
    else:
        CreateTraining_set.create_training_set(labels,irmodel)  
        
          
    # retrieved training data
    train_indices=pickle.load(open("outputs/trainingSet.p","rb"))
    train_indices.sort()    
    train_data_1_all=np.array(padded_data_1)[train_indices,:]
    train_data_2_all=np.array(padded_data_2)[train_indices,:]
    train_labels_all=np.array(labels)[train_indices,:]
    train_leaks_all=np.array(leaks)[train_indices,:]
    
    VALIDATION_SPLIT = 0.1    
    dev_idx = max(1, int(len(train_labels_all) * VALIDATION_SPLIT))
   
    #splitting training data and validation data    
    train_data_1, val_data_1=train_data_1_all[:-dev_idx], train_data_1_all[-dev_idx:]
    train_data_2, val_data_2=train_data_2_all[:-dev_idx], train_data_2_all[-dev_idx:]
    train_labels, val_labels=train_labels_all[:-dev_idx], train_labels_all[-dev_idx:]
    train_leaks , val_leaks =train_leaks_all[:-dev_idx],  train_leaks_all[-dev_idx:]
    
    
    
    #building Rnn model with LSTM 
    RATE_DROP_LSTM = 0.17
    RATE_DROP_DENSE = 0.25
    NUMBER_LSTM = 50
    NUMBER_DENSE_UNITS = 50
    NUMBER_DENSE_UNITS_1 = 25
    ACTIVATION_FUNCTION = 'relu'
    
    
    # Creating word embedding layer
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    
    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(NUMBER_LSTM, dropout=RATE_DROP_LSTM, recurrent_dropout=RATE_DROP_LSTM))
    
    # Creating LSTM Encoder layer for source artifact
    sequence_1_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    
    
    # Creating LSTM Encoder layer for for target artifact
    sequence_2_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)
    
        
    # Creating leaks input
    leaks_input = Input(shape=(leaks.shape[1],))   
    
    
    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    
    merged = concatenate([x1, x2, leaks_input])
    merged = BatchNormalization()(merged)
    merged = Dropout(RATE_DROP_DENSE)(merged)
    merged = Dense(NUMBER_DENSE_UNITS, activation=ACTIVATION_FUNCTION)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(RATE_DROP_DENSE)(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (NUMBER_LSTM, NUMBER_DENSE_UNITS, RATE_DROP_LSTM, RATE_DROP_DENSE)
    model_save_directory='outputs/'
    checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    
    
    numExec=1
    RnnResult=[]
    for i in range(0,numExec):
    #tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
    
        model.fit([train_data_1, train_data_2, train_leaks], train_labels, 
                   validation_data=([val_data_1, val_data_2, val_leaks], val_labels),
                    epochs=200, batch_size=64, shuffle=True, callbacks=[early_stopping, model_checkpoint])
        
        loss0, accuracy0 = model.evaluate([train_data_1_all, train_data_2_all, train_leaks_all], train_labels_all, verbose=1)
        print('Accuracy: %f' % (accuracy0*100))
        
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        
       
        preds = list(model.predict([padded_data_1, padded_data_2, leaks], verbose=1).ravel())
        RnnResult.append(preds)
    
        loss, accuracy = model.evaluate([padded_data_1, padded_data_2, leaks], labels, verbose=1)
        print('Accuracy: %f' % (accuracy*100))
    
    
    resvect=np.zeros((len(preds),1))
    
    for res in RnnResult:
        for i in range(0,len(preds)):
                resvect[i]=resvect[i]+res[i]
    
    resvect=resvect/numExec
    
        
    results = [(x[0], y[0], z) for (x, y), z in zip(artifact_pairs, resvect)]
    
    
    print("RNN Cbow model compute")
    
    #creation of the csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("Artifact1","Artifact2","probability"))
        writer.writeheader()
        # in each row # add requirements names, model name , and value
        for res in results:
            writer.writerow({'Artifact1': str("{0}".format(res[0])), 'Artifact2': str("{0}".format(res[1])), 'probability': str("{0}".format(res[2]))})  
       

    print("similarity matrix build")
    
    