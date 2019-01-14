import math
import os
# import ipdb
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle  
import os

import tensorflow.python.platform
from keras.preprocessing import sequence
from sklearn.model_selection import KFold

class Caption_Generator():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, n_words, dim_embed, dim_ctx, dim_hidden, n_lstm_steps, batch_size, ctx_shape, bias_init_vector=None):
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.ctx_shape = ctx_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

        self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

        self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')
        
        self.init_hidden_W_glstm = self.init_weight(dim_hidden, dim_hidden, name='init_hidden_W_glstm')
        self.init_hidden_b_glstm = self.init_bias(dim_hidden, name='init_hidden_b_glstm')

        self.init_memory_W_glstm = self.init_weight(dim_hidden, dim_hidden, name='init_memory_W_glstm')
        self.init_memory_b_glstm = self.init_bias(dim_hidden, name='init_memory_b_glstm')

        self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W')
        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')
        
        self.lstm_W_glstm = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W_glstm')
        self.lstm_U_glstm = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U_glstm')
        self.lstm_b_glstm = self.init_bias(dim_hidden*4, name='lstm_b_glstm')
        
        self.guidance_W_glstm = self.init_weight(1, dim_hidden*4, name='guidance_W_glstm')

        self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W')
        
        self.context_encode_W_glstm = self.init_weight(dim_hidden, dim_hidden*4, name='context_encode_W_glstm')
        
        self.non_image_vector = tf.get_variable("non_image_vector", shape=[non_shape[0], non_shape[1]], initializer=tf.contrib.layers.xavier_initializer())

        self.image_att_W = self.init_weight(dim_ctx, dim_att, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_att, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_att, name='pre_att_b')

        self.att_W = self.init_weight(dim_att, dim_ctx, name='att_W')
        self.att_b = self.init_bias(dim_ctx, name='att_b')

        self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')
        
        self.decode_lstm_W_glstm = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W_glstm')
        self.decode_lstm_b_glstm = self.init_bias(dim_embed, name='decode_lstm_b_glstm')

        self.decode_word_W = self.init_weight(dim_embed, n_words, name='decode_word_W')

        if bias_init_vector is not None:
            self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(n_words, name='decode_word_b')
            
        self.decode_alpha_W_glstm = self.init_weight(dim_embed, dim_ctx, name='decode_alpha_W_glstm')
        self.decode_alpha_b_glstm = self.init_bias(dim_ctx, name='decode_alpha_b_glstm')


    def get_initial_lstm(self, mean_context):
        initial_hidden = tf.nn.tanh(tf.matmul(tf.squeeze(mean_context, 2), self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(tf.squeeze(mean_context, 2), self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    def get_initial_glstm(self, mean_context):
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W_glstm) + self.init_hidden_b_glstm)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W_glstm) + self.init_memory_b_glstm)

        return initial_hidden, initial_memory

    def build_model(self):
        with tf.variable_scope("build_model") as scope:
          context = tf.placeholder("float32", [self.batch_size - 0, self.ctx_shape[0], self.ctx_shape[1]])
          rating = tf.placeholder("float32", [self.batch_size - 0, 1, 1])
          sentence = tf.placeholder("int32", [self.batch_size - 0, self.n_lstm_steps])
          mask = tf.placeholder("float32", [self.batch_size - 0, self.n_lstm_steps])
  
          #h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))
          all_context = tf.concat([context, tf.expand_dims(tf.matmul(tf.squeeze(rating, 2), tf.transpose(self.non_image_vector)), 2)], 1) # (80, 5120)
          h, c = self.get_initial_lstm(all_context) # (80, 256)
  
          context_flat = tf.reshape(all_context, [-1, self.dim_ctx])
          context_encode = tf.matmul(context_flat, self.image_att_W) # (batch_size, 196, 512)
          context_encode = tf.reshape(context_encode, [-1, dim_att])
  
          loss = 0.0
          
          for ind in range(self.n_lstm_steps):
  
              if ind == 0:
                  word_emb = tf.zeros([self.batch_size - 0, self.dim_embed])
              else:
                  tf.get_variable_scope().reuse_variables()
                  with tf.device("/cpu:0"):
                      word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1])
  
              x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # [80, 256 * 4]
  
              labels = tf.expand_dims(sentence[:,ind], 1)
              indices = tf.expand_dims(tf.range(0, self.batch_size - 0, 1), 1)
              concated = tf.concat([indices, labels], 1)
              onehot_labels = tf.sparse_to_dense( concated, tf.stack([self.batch_size - 0, self.n_words]), 1.0, 0.0)
              context_encode = context_encode + \
                   tf.matmul(h, self.hidden_att_W) + \
                   self.pre_att_b
  
              context_encode = tf.nn.tanh(context_encode) # [80, 256]
              context_encode_flat = tf.reshape(context_encode, [-1, dim_att]) # [80, 256]
              
              
              
              
              
              x_t_glstm = tf.matmul(word_emb, self.lstm_W_glstm) + self.lstm_b_glstm # [80, 256 * 4]
              h_glstm, c_glstm = self.get_initial_glstm(context_encode_flat) # [80, 256]
              lstm_preactive_glstm = tf.matmul(h_glstm, self.lstm_U_glstm) + x_t_glstm + tf.matmul(context_encode_flat, self.context_encode_W_glstm) + tf.matmul(tf.squeeze(rating, 2), self.guidance_W_glstm) # [80, 256 * 4]
              i_glstm, f_glstm, o_glstm, new_c_glstm = tf.split(lstm_preactive_glstm, 4, 1) # [80, 256]
              i_glstm = tf.nn.sigmoid(i_glstm)
              f_glstm = tf.nn.sigmoid(f_glstm)
              o_glstm = tf.nn.sigmoid(o_glstm)
              new_c_glstm = tf.nn.tanh(new_c_glstm)
              c_glstm = f_glstm * c_glstm + i_glstm * new_c_glstm
              h_glstm = o_glstm * tf.nn.tanh(new_c_glstm)
              logits_glstm = tf.matmul(h_glstm, self.decode_lstm_W_glstm) + self.decode_lstm_b_glstm # [80, 256]
              logits_glstm = tf.nn.relu(logits_glstm)
              logits_glstm = tf.nn.dropout(logits_glstm, 0.5)
              alpha = tf.matmul(logits_glstm, self.decode_alpha_W_glstm) + self.decode_alpha_b_glstm # [80, 5120]
              
              
                            
              
              
              # alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b # [80, 5120]
              alpha = tf.reshape(alpha, [-1, self.ctx_shape[0] + non_shape[0]])
              # alpha = tf.nn.softmax( alpha )
  
              # print(context, alpha)
              weighted_context = tf.multiply(all_context, tf.expand_dims(alpha, 2)) # [80, 5120, 1]
  
              lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(tf.squeeze(weighted_context), self.image_encode_W) # [80, 256 * 4]
              i, f, o, new_c = tf.split(lstm_preactive, 4, 1) # [80, 256]
  
              i = tf.nn.sigmoid(i)
              f = tf.nn.sigmoid(f)
              o = tf.nn.sigmoid(o)
              new_c = tf.nn.tanh(new_c)
  
              c = f * c + i * new_c # [80, 256]
              h = o * tf.nn.tanh(new_c) # [80, 256]
  
              logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b # [80, 256]
              logits = tf.nn.relu(logits)
              logits = tf.nn.dropout(logits, 0.5)
  
              logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b # [80, n_word]
              cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
              cross_entropy = cross_entropy * mask[:,ind]
  
              current_loss = tf.reduce_sum(cross_entropy)
              loss = loss + current_loss
  
          loss = loss / tf.reduce_sum(mask)
          return loss, context, rating, sentence, mask

    def build_valiator(self, maxlen):
        context = tf.placeholder("float32", [1, self.ctx_shape[0], self.ctx_shape[1]]) # [1, 4096, 1]
        rating = tf.placeholder("float32", [1, 1, 1])
        sentence = tf.placeholder("int32", [1, self.n_lstm_steps])
        mask = tf.placeholder("float32", [1, self.n_lstm_steps])
        #h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))\
        
        all_context = tf.concat([context, tf.expand_dims(tf.matmul(tf.squeeze(rating, 2), tf.transpose(self.non_image_vector)), 2)], 1)
        h, c = self.get_initial_lstm(all_context) # [1, 256]

        sqz = tf.squeeze(all_context, 2) # [1, 4096]
        context_encode = tf.matmul(sqz, self.image_att_W) # [1, 256]
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([1, self.dim_embed]) # [1, 256]
        loss = 0.0
        for ind in range(maxlen):
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # [1, 1024]
            context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b # [1, 256]
            context_encode = tf.nn.tanh(context_encode) # [1, 256]
        
            labels = tf.expand_dims(sentence[:,ind], 1)
            indices = tf.expand_dims(tf.range(0, 1, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense( concated, tf.stack([1, self.n_words]), 1.0, 0.0)
            
            x_t_glstm = tf.matmul(word_emb, self.lstm_W_glstm) + self.lstm_b_glstm 
            h_glstm, c_glstm = self.get_initial_glstm(context_encode)
            lstm_preactive_glstm = tf.matmul(h_glstm, self.lstm_U_glstm) + x_t_glstm + tf.matmul(context_encode, self.context_encode_W_glstm) + tf.matmul(tf.squeeze(rating, 2), self.guidance_W_glstm)
            i_glstm, f_glstm, o_glstm, new_c_glstm = tf.split(lstm_preactive_glstm, 4, 1) 
            i_glstm = tf.nn.sigmoid(i_glstm)
            f_glstm = tf.nn.sigmoid(f_glstm)
            o_glstm = tf.nn.sigmoid(o_glstm)
            new_c_glstm = tf.nn.tanh(new_c_glstm)
            c_glstm = f_glstm * c_glstm + i_glstm * new_c_glstm
            h_glstm = o_glstm * tf.nn.tanh(new_c_glstm)
            logits_glstm = tf.matmul(h_glstm, self.decode_lstm_W_glstm) + self.decode_lstm_b_glstm
            logits_glstm = tf.nn.relu(logits_glstm)
            logits_glstm = tf.nn.dropout(logits_glstm, 0.5)
            alpha = tf.matmul(logits_glstm, self.decode_alpha_W_glstm) + self.decode_alpha_b_glstm 

            #alpha = tf.matmul(context_encode, self.att_W) + self.att_b # [1, 5120]
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0] + non_shape[0]] ) 
            # alpha = tf.nn.softmax(alpha)

            alpha = tf.reshape(alpha, (self.ctx_shape[0] + non_shape[0], -1)) # [5120, 1]
            alpha_list.append(alpha)

            weighted_context = tf.multiply(sqz, tf.transpose(alpha)) # [1, 5120]
            # weighted_context = tf.expand_dims(weighted_context, 0)
            # weighted_context = tf.expand_dims(weighted_context, 1)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W) # [1, 256]

            i, f, o, new_c = tf.split(lstm_preactive, 4, 1)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f*c + i*new_c
            h = o*tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            max_prob_word = tf.argmax(logit_words, 1)
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * mask[:,ind]

            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

            with tf.device("/cpu:0"):
                word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logit_words)

        loss = loss / tf.reduce_sum(mask)
        return context, rating, generated_words, logit_list, alpha_list, sentence, loss, mask

    def build_generator(self, maxlen):
        context = tf.placeholder("float32", [1, self.ctx_shape[0], self.ctx_shape[1]]) # [1, 4096, 1]
        rating = tf.placeholder("float32", [1, 1, 1])
        #h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))\
        
        all_context = tf.concat([context, tf.expand_dims(tf.matmul(tf.squeeze(rating, 2), tf.transpose(self.non_image_vector)), 2)], 1)
        h, c = self.get_initial_lstm(all_context) # [1, 256]

        sqz = tf.squeeze(all_context, 2) # [1, 4096]
        context_encode = tf.matmul(sqz, self.image_att_W) # [1, 256]
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([1, self.dim_embed]) # [1, 256]
        for ind in range(maxlen):
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # [1, 1024]
            context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b # [1, 256]
            context_encode = tf.nn.tanh(context_encode) # [1, 256]

            x_t_glstm = tf.matmul(word_emb, self.lstm_W_glstm) + self.lstm_b_glstm # [1, 256 * 4]
            h_glstm, c_glstm = self.get_initial_glstm(context_encode) # [1, 256]
            lstm_preactive_glstm = tf.matmul(h_glstm, self.lstm_U_glstm) + x_t_glstm + tf.matmul(context_encode, self.context_encode_W_glstm) + tf.matmul(tf.squeeze(rating, 2), self.guidance_W_glstm) # [1, 256 * 4]
            i_glstm, f_glstm, o_glstm, new_c_glstm = tf.split(lstm_preactive_glstm, 4, 1) # [1, 256]
            i_glstm = tf.nn.sigmoid(i_glstm)
            f_glstm = tf.nn.sigmoid(f_glstm)
            o_glstm = tf.nn.sigmoid(o_glstm)
            new_c_glstm = tf.nn.tanh(new_c_glstm)
            c_glstm = f_glstm * c_glstm + i_glstm * new_c_glstm
            h_glstm = o_glstm * tf.nn.tanh(new_c_glstm)
            logits_glstm = tf.matmul(h_glstm, self.decode_lstm_W_glstm) + self.decode_lstm_b_glstm # [1, 256]
            logits_glstm = tf.nn.relu(logits_glstm)
            logits_glstm = tf.nn.dropout(logits_glstm, 0.5)
            alpha = tf.matmul(logits_glstm, self.decode_alpha_W_glstm) + self.decode_alpha_b_glstm # [1, 5120]
              
            # alpha = tf.matmul(context_encode, self.att_W) + self.att_b # [1, 5120]
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0] + non_shape[0]] ) 
            # alpha = tf.nn.softmax(alpha)

            alpha = tf.reshape(alpha, (self.ctx_shape[0] + non_shape[0], -1)) # [5120, 1]
            alpha_list.append(alpha)

            weighted_context = tf.multiply(sqz, tf.transpose(alpha)) # [1, 5120]
            # weighted_context = tf.expand_dims(weighted_context, 0)
            # weighted_context = tf.expand_dims(weighted_context, 1)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W) # [1, 256]

            i, f, o, new_c = tf.split(lstm_preactive, 4, 1)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f*c + i*new_c
            h = o*tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            logit_words = tf.nn.softmax(logit_words)

            max_prob_word = tf.argmax(logit_words, 1)

            with tf.device("/cpu:0"):
                word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logit_words)

        return context, rating, generated_words, logit_list, alpha_list


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


n_epochs=200
batch_size=80
dim_embed=256
dim_ctx=4096 + 1024 # 5120
dim_att=256
dim_hidden=256
ctx_shape=[4096, 1]
non_shape=[1024, 1]
pretrained_model_path = './model/model-199'
#pretrained_model_path = None
#############################
reviewText_path = './data/reviewTexts.b'
image_id_path = './data/image_ids.b'
feat_path = './data/now_feats.b'
model_path = './model/'
#############################


def train(pretrained_model_path=pretrained_model_path): 
    captions = pickle.load(open('./data/new_captions.b', "rb"))[60000:100000]
    feats = pickle.load(open('./data/new_now_feats.b', "rb"))[60000:100000]
    #feats = list(map(lambda x: x[:256], _feats))
    ratings = pickle.load(open('./data/new_ratings.b', "rb"))[60000:100000]
    #image_id = pickle.load(open(image_id_path, "rb"))
    
    maxlen = 100
    newcaptions = []
    newfeats = []
    newratings = []
    for caption, feat, rating in zip(captions, feats, ratings):
      if len(caption.split(' ')) <= maxlen:
        newcaptions.append(caption)
        newfeats.append(feat)
        newratings.append(rating)
    captions = newcaptions
    feats = newfeats
    ratings = newratings
        
    
    #wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    #pickle.dump(wordtoix, open("./wordtoix_6w_10w.b", "wb"))
    #pickle.dump(ixtoword, open("./ixtoword_6w_10w.b", "wb"))
    #pickle.dump(bias_init_vector, open("./bias_init_vector_6w_10w.b", "wb"))
    #print ("Word vocab done!")
    
    wordtoix = pickle.load(open("./wordtoix_6w_10w.b", "rb"))
    ixtoword = pickle.load(open("./ixtoword_6w_10w.b", "rb"))
    bias_init_vector = pickle.load(open("./bias_init_vector_6w_10w.b", "rb"))
    
    learning_rate=0.0002
    
    n_words = len(wordtoix)
    #feats = pickle.load(open(feat_path, "rb"))
    #captions = pickle.load(open(reviewText_path, "rb"))
    
    #deleted = [False] * len(feats)
    #newfeats = []
    #newcaptions = []
    #i = 0
    #for x in feats:
    #  if not x:
    #    deleted[i] = True
    #  i = i + 1
    #for i in range(len(feats)):
    #  if not deleted[i]:
    #    newfeats.append(feats[i])
    #    newcaptions.append(captions[i])
    #pickle.dump(newfeats, open('../data/new_now_feats.b', "wb"))
    #pickle.dump(newcaptions, open('../data/new_captions.b', "wb"))
    #feats = newfeats
    #captions = newcaptions
    
    print("Ready!")
    
    #maxlen = np.max( list(map(lambda x: len(x.split(' ')), captions) ))
    i = 0
    for x in captions:
      #if i % 10000 == 0:
      #  print(i)
      if len(x.split(' ')) > maxlen:
        spl = x.split(' ')[0:maxlen]
        last = maxlen - 1
        while last >= 0 and spl[last] != '' and spl[last][-1] != '.' :
          last = last - 1
        if last == -1:
          last = maxlen - 1
        spl = x.split(' ')[0:last + 1]
        captions[i] = ' '.join(spl)
        # print(captions[i])
      i = i + 1
      
    

    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximum alloc gpu50% of MEM
    sess = tf.InteractiveSession(config = config)

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1, 
            batch_size=batch_size,
            ctx_shape=ctx_shape,
            bias_init_vector=bias_init_vector)
            
    loss, context, rating, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    global_step = tf.Variable(0, trainable=False)
    #decayed_learning_rate = tf.train.exponential_decay(learning_rate, tf.Variable(0, trainable=False), n_epochs, 0.0001, staircase=False, name=None)  
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    tf.initialize_all_variables().run()
    if pretrained_model_path is not None:
        print ("Starting with pretrained model")
        saver.restore(sess, pretrained_model_path)

    # index = list(annotation_data.index)
    # np.random.shuffle(index)
    
    for epoch in range(n_epochs):
        for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            current_feats = np.array(feats[ start:end ])
            current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)

            current_captions = np.array(captions[start:end])
            current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)) 
            
            current_ratings = np.array(ratings[start:end]).reshape((batch_size, 1, 1))

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( list(map(lambda x: (x != 0).sum()+1, current_caption_matrix )))
 
            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, train_loss = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                rating:current_ratings,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})

            print ("Train Loss: ", train_loss, "Epoch: ", epoch, "/",  n_epochs, "Progress: ", start, "/", len(captions))
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


def test(test_feat='./guitar_player.npy', model_path='./model/model-198', maxlen=100):
    #f = open("data/reviewTexts.b", "rb")
    #captions = pickle.load(f)
    #wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    
    #test_feat = np.array(pickle.load(open("../data/new_now_feats.b", "rb"))[66666])
    test_feat = np.array(pickle.load(open("../data/test_image.b", "rb")))
    input_rating = np.array([1.0]).reshape((1, 1, 1))
    
    wordtoix = pickle.load(open("./wordtoix_6w_10w.b", "rb"))
    ixtoword = pickle.load(open("./ixtoword_6w_10w.b", "rb"))
    bias_init_vector = pickle.load(open("./bias_init_vector_6w_10w.b", "rb"))
    n_words = len(wordtoix)
    feat = test_feat.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)
    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen,
            batch_size=batch_size,
            ctx_shape=ctx_shape)

    context, rating, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=maxlen)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index = sess.run(generated_words, feed_dict={context:feat, rating:input_rating})
    alpha_list_val, logit_list_val = sess.run([alpha_list, logit_list], feed_dict={context:feat, rating:input_rating})
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    #generated_words.reverse()
    #alpha_list_val.reverse()
    #punctuation = np.argmax(np.array(generated_words) == '.')

    #generated_words = generated_words[punctuation:]
    #alpha_list_val = alpha_list_val[punctuation:]
    #generated_words.reverse()
    #alpha_list_val.reverse()
    
    #result_sentence = []
    #for word, logit_words in zip(generated_words, logit_list_val):
    #    max_prob = max(logit_words[0])
    #    print(max_prob)
    #    if max_prob >= 0.1:
    #        result_sentence.append(word)
    
    generated_sentence = ' '.join(generated_words)
    print(generated_sentence)
    return generated_words, alpha_list_val

#    ipdb.set_trace()

#train()
test()
