import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate
from keras.layers import Layer
import keras.backend as K
 
vocab_size = 10000
 
pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2
max_len = 200
gru_units = 32
batch_size = 200
embedding_size = 32


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment
        
        

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size, start_char=start_id,
                                                                        oov_char=oov_id, index_from=index_offset)
 
word2idx = tf.keras.datasets.imdb.get_word_index()
 
idx2word = {v + index_offset: k for k, v in word2idx.items()}
 
idx2word[pad_id] = '<PAD>'
idx2word[start_id] = '<START>'
idx2word[oov_id] = '<OOV>'
 
x_train = sequence.pad_sequences(x_train,
                                 maxlen=max_len,
                                 truncating='post',
                                 padding='post',
                                 value=pad_id)
x_test = sequence.pad_sequences(x_test, maxlen=max_len,
                                truncating='post',
                                padding='post',
                                value=pad_id)

sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_len)(sequence_input)

sample_hidden = tf.zeros((batch_size, gru_units))

GRU = tf.keras.layers.GRU(gru_units,
    return_sequences=True,
    return_state=True,
    recurrent_initializer='glorot_uniform')

sample_output, sample_hidden = GRU(embedded_sequences, initial_state=sample_hidden)

encoder = Encoder(vocab_size+1, embedding_size, gru_units)
initial_state = encoder.init_states(1)
encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)

attention = LuongAttention(gru_units)
context_vector, attention_weights = attention(sample_output, sample_hidden)

output = tf.keras.layers.Dense(1, activation='sigmoid')(context_vector)

model = tf.keras.Model(inputs=sequence_input, outputs=output)

print(model.summary())

#model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
#history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

