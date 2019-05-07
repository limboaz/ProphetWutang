#!/usr/bin/env python
# coding: utf-8

# In[28]:


import collections
from pprint import pprint
import json
import re
import os
import pickle
import numpy as np
import tensorflow as tf

space = re.compile(r'\s')
period = re.compile(r'(?<![A-Z])\.|(?<!\w)\'(?!\w)')
punct = re.compile(r'[^\'#@\.\w]')

def tokenize(sent):
    sent = space.split(sent)
    to = []
    tokens = []
    for t in sent:
        if t:
            to += period.split(t)
    for t in to:
        if t:
            tokens += punct.split(t)
    return [a for a in tokens if a is not '']

def get_lyrics_years(songs):
    lyrics = []
    years = []
    
    for song_id in list(songs.keys()):
        lyrics.append(songs[song_id]["lyrics"].lower())
        years.append(str(songs[song_id]["year"]))
        
    return lyrics, years

def dataset(lyrics, vocab_size):
    # Words that were uncommon get noted as Out of bounds
    count = [["OOB", 0]]
    count.extend(collections.
                 Counter([word for lyric in lyrics for word in lyric]).
                 most_common(vocab_size - 1))
    word_to_index = {}
    for word, _ in count:
        word_to_index[word] = len(word_to_index)
    encoded_lyrics = []
    for song in lyrics:
        encoded = []
        for word in song:
            index = word_to_index.get(word, 0)
            if index == 0:
                count[0][1] += 1
            encoded.append(index)
        encoded_lyrics.append(encoded)
        index_to_word = dict(zip(word_to_index.values(), word_to_index.keys()))
    return encoded_lyrics, count, word_to_index, index_to_word

def generate_batch(lyrics, batch_size, window_size):
    batch = []
    labels = []
    
    while len(batch) < batch_size:
        # select random song
        r_song_index = int(np.random.choice(len(lyrics), size=1))
        r_song = lyrics[r_song_index]
        # generate window
        window = [r_song[max(i - window_size, 0):(i + window_size + 1)] for i, _ in enumerate(r_song)]
        
        batch_labels = [(r_song[i:i + window_size], r_song[i + window_size]) for i in range(len(r_song) - window_size)]
        if len(batch_labels) <= 2:
            continue
        # extract batch and label for this iteration
        b, l = [list(x) for x in zip(*batch_labels)]
        b = [x + [r_song_index] for x in b]
        
        batch.extend(b[:batch_size])
        labels.extend(l[:batch_size])
        
    batch = batch[:batch_size]
    labels = labels[:batch_size]
    
    batch = np.array(batch)
    labels = np.transpose(np.array([labels]))
    return batch, labels


# In[29]:


batch_size = 500

# Number of unique words to consider in our model
vocabulary_size = 50000
generations = 100000
learning_rate = 0.001

# vector size
embedding_size = 200
song_embedding_size = 200
concatenated_size = embedding_size + song_embedding_size

# intervals to print out progress
save_interval = 500
print_loss_interval = 300

# negative examples to sample
num_sampled = 250

window_size = 5

data_folder = "model_out"
sess = tf.Session()


# In[ ]:


songs_filename = "songs/songs.json"
songs_file = open(songs_filename, "r+")
songs_dict = json.load(songs_file)


lyrics, years = get_lyrics_years(songs_dict)

tokenized_lyrics = []

print("[Tokenizing lyrics]")
for l in lyrics:
    tokenized_lyrics.append(tokenize(l))
print("[Done]")

# encoded_lyrics is the original list of lyrics but with tokens
# replaced with their corresponding dictionary index
print("[Encoding Lyrics]")
encoded_lyrics, count, word_to_index, index_to_word = dataset(
    tokenized_lyrics, vocabulary_size)
print("[Done]")

del lyrics
del tokenized_lyrics

print("Number of songs:", len(encoded_lyrics))


# In[ ]:


print("[Creating Model]")

with tf.name_scope('inputs'):
    x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1])
    y_target = tf.placeholder(tf.int32, shape=[None, 1])

with tf.name_scope('weights'):
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, concatenated_size],
                           stddev=1.0 / np.sqrt(concatenated_size)))
with tf.name_scope('biases'):
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

with tf.device('/gpu:0'):
    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        song_embeddings = tf.Variable(tf.random_uniform([len(encoded_lyrics), song_embedding_size], -1.0, 1.0))
        embed = tf.zeros([batch_size, embedding_size])

        # lookup word embeddings
        for element in range(window_size):
            embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

        song_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
        song_embed = tf.nn.embedding_lookup(song_embeddings, song_indices)

        # concatenate embeddings
        final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(song_embed)])

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights, biases=nce_biases,
            inputs=final_embed, labels=y_target,
            num_sampled=num_sampled, 
            num_classes=vocabulary_size))

# SGD optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

# Create model saving operation
saver = tf.train.Saver({"embeddings": embeddings, "song_embeddings": song_embeddings})

# Initialize global varialbles
init = tf.global_variables_initializer()


# In[ ]:


sess.run(init)
print('[Starting Training]')

loss_vec = []
loss_x_vec = []

for i in range(generations):
    batch_inputs, batch_labels = generate_batch(encoded_lyrics, batch_size,
                                               window_size)

    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    sess.run(optimizer, feed_dict=feed_dict)

    # Return the loss
    if (i + 1) % print_loss_interval == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))

    # Save dictionary + embeddings
    if (i + 1) % save_interval == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder, 'songs_vocab.pkl'), 'wb') as f:
            pickle.dump(word_to_index, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder, 'doc2vec_song_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
print("[Training doc2vec model Complete]")


# In[ ]:


# Start logistic model-------------------------
max_words = 100
logistic_batch_size = 500

# Split dataset into train and test sets
# Need to keep the indices sorted to keep track of document index
train_indices = np.sort(np.random.choice(len(years), round(0.8 * len(years)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(years))) - set(train_indices))))
song_train = [x for ix, x in enumerate(encoded_lyrics) if ix in train_indices]
song_test = [x for ix, x in enumerate(encoded_lyrics) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(years) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(years) if ix in test_indices])

# Pad/crop movie reviews to specific length
song_train = np.array([x[0:max_words] for x in [y + [0] * max_words for y in song_train]])
song_test = np.array([x[0:max_words] for x in [y + [0] * max_words for y in song_test]])


with tf.name_scope('inputs'):
    log_x_inputs = tf.placeholder(tf.int32, shape=[None, max_words + 1])  # plus 1 for doc index
    log_y_target = tf.placeholder(tf.int32, shape=[None, 1])

with tf.device('/gpu:0'):
    # Define logistic embedding lookup (needed if we have two different batch sizes)
    # Add together element embeddings in window:
    with tf.name_scope('embeddings'):
        log_embed = tf.zeros([logistic_batch_size, embedding_size])
        for element in range(max_words):
            log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:, element])

        log_doc_indices = tf.slice(log_x_inputs, [0, max_words], [logistic_batch_size, 1])
        log_doc_embed = tf.nn.embedding_lookup(doc_embeddings, log_doc_indices)

        # concatenate embeddings
        log_final_embed = tf.concat(1, [log_embed, tf.squeeze(log_doc_embed)])

with tf.name_scope('weights'):
    A = tf.Variable(tf.random_normal(shape=[concatenated_size, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(log_final_embed, A), b)

with tf.name_scope('loss'):
    logistic_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=model_output, logits=tf.cast(log_y_target, tf.float32)))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(log_y_target, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

with tf.name_scope('optimizer'):
    logistic_opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    logistic_train_step = logistic_opt.minimize(logistic_loss, var_list=[A, b])

saver = tf.train.Saver()

# Intitialize Variables
init = tf.global_variables_initializer()

sess.run(init)


# In[ ]:


# Start Logistic Regression
print('[Starting Logistic Doc2Vec Model Training]')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(song_train.shape[0], size=logistic_batch_size)
    rand_x = song_train[rand_index]
    # Append song index at the end of lyrics data
    rand_x_doc_indices = train_indices[rand_index]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])

    feed_dict = {log_x_inputs: rand_x, log_y_target: rand_y}
    sess.run(logistic_train_step, feed_dict=feed_dict)

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        rand_index_test = np.random.choice(song_test.shape[0], size=logistic_batch_size)
        rand_x_test = song_test[rand_index_test]
        # Append review index at the end of text data
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
        rand_y_test = np.transpose([target_test[rand_index_test]])

        test_feed_dict = {log_x_inputs: rand_x_test, log_y_target: rand_y_test}

        i_data.append(i + 1)

        train_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(logistic_loss, feed_dict=test_feed_dict)
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
        test_acc.append(test_acc_temp)
        print([(pred_y, y) for pred_y, y in zip(predictions, log_y_target)])
    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder, 'doc2vec_log_reg_model.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)


# In[ ]:


# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

