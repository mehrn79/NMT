---
jupyter:
  accelerator: GPU
  colab:
    name: Copy of NMT_model.ipynb
  gpuClass: standard
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

# **Attention Mechanisms in Recurrent Neural Networks (RNNs) With Keras**
``` {.python}
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import pandas as pd 
from tensorflow.keras import optimizers
```
خواندن دیتا
``` {.python}
! gdown --id 1M3r4dASBa5fUGA_Zreg071HlcllbXxbp
```

``` {.python}
with open('/content/language_data.csv') as f:
  language_data = f.read()
```
جدا سازی دیتا به دو دسته فارسی و انگلیسی
``` {.python}
language_data = pd.read_csv('language_data.csv')
english_text = language_data['English'].values
persian_text = language_data['Persian'].values
english_text[0]=english_text[0].replace('\ufeff','')
```
چون تعداد دیتا زیاد است و پارامترهای شبکه زیاد می شود تنها 200000 دیتا اولیه را مورد بررسی قرار می دهیم
``` {.python}
english_text = english_text[:200000]
persian_text = persian_text[:200000]
```
حدف عبارت های اضافی و علائم نگارشی از متن
``` {.python}
# Lowercase and remove punctuation in sentences
def clean_sentence(sentence):
    # Add a space ' ' befor the ? word
    sentence = sentence.replace('?', ' ?')
    # Lower case the sentence
    lower_case_sent = sentence.lower().strip()
    # Strip punctuation
    string_punctuation = string.punctuation  # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    string_punctuation = string_punctuation.replace('?','')
    # Clean the sentence
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    clean_sentence = '<start> ' + clean_sentence + ' <end>'
    return clean_sentence
```

``` {.python}
english_sent = [clean_sentence(pair) for pair in english_text]
persian_sent = [clean_sentence(pair) for pair in persian_text]
```
حالا دنباله ها را توکن می کنیم.در توکن‌سازی وازگان را  به اعداد صحیح تبدیل و آنها برای ایجاد طوبل یکسان به هم اضافه می شوند
``` {.python}
import tensorflow as tf

# Convert sequences to tokenizers
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  
  # Convert sequences into internal vocab
  lang_tokenizer.fit_on_texts(lang)

  # Convert internal vocab to numbers
  tensor = lang_tokenizer.texts_to_sequences(lang)

  # Pad the tensors to assign equal length to all the sequences
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer
```

``` {.python}
# Load the dataset
def load_dataset(targ_lang, inp_lang):

  # Tokenize the sequences
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
```

``` {.python}
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(persian_sent, english_sent)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
```
مجموعه داده‌های train و اعتبارسنجی را جدا میکنیم .
``` {.python}
from sklearn.model_selection import train_test_split

# Create training and validation sets using an 80/20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
```

نگاشت ایجاد شده بین کلمات دنباله و شاخص ها را اعتبار سنجی کنید.
``` {.python}
# Show the mapping b/w word index and language tokenizer
def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
      
print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])
```

    Input Language; index to word mapping
    1 ----> <start>
    9 ----> and
    100 ----> then
    17 ----> we
    247 ----> came
    77 ----> back
    37 ----> here
    2 ----> <end>

    Target Language; index to word mapping
    1 ----> <start>
    390 ----> بعدش
    6520 ----> برگشتيم
    27 ----> اينجا
    2 ----> <end>

## **مقداردهی اولیه پارامترهای مدل**

* BUFFER_SIZE: Total number of input/target samples.

* BATCH_SIZE: Length of the training batch.

* steps_per_epoch: The number of steps per epoch. Computed by dividing BUFFER_SIZE by BATCH_SIZE.

* embedding_dim: Number of nodes in the embedding layer.

* units: Hidden units in the network.

* vocab_inp_size: Length of the input (English) vocabulary.

* vocab_tar_size: Length of the output (persain) vocabulary

``` {.python}
# Essential model parameters
# Total number of input/target samples. In our model
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 256
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 150
units = 256
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
```
سپس، tf.data.Dataset API را فراخوانی می کنیم و یک مجموعه داده مناسب ایجاد می کنیم.
``` {.python}
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
```

متدهای __init__() و call() مدل را در یک کلاس Encoder قرار دهید.

در روش، __init__()، اندازه دسته و واحدهای رمزگذاری را مقداردهی اولیه می کند. یک لایه embedding اضافه کنید که vocab_size را به عنوان بعد ورودی و embedding_dim را به عنوان بعد خروجی بپذیرد. همچنین یک لایه GRU اضافه کنید که واحدها (بعد فضای خروجی) و اولین بعد پنهان را بپذیرد.

در متد call()، انتشار رو به جلو که باید از طریق شبکه رمزگذار اتفاق بیفتد را تعریف کنید.

علاوه بر این، یک متد () initialize_hidden_state برای مقداردهی اولیه حالت مخفی با ابعاد batch_size و units تعریف کنید.

``` {.python}
# Encoder class
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units

    # Embed the vocab to a dense embedding 
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # GRU Layer
    # glorot_uniform: Initializer for the recurrent_kernel weights matrix, 
    # used for the linear transformation of the recurrent state
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  # Encoder network comprises an Embedding layer followed by a GRU layer
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  # To initialize the hidden state
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
```
برای بررسی اشکال خروجی encoder و حالت پنهان، کلاس encoder را فراخوانی می کنیم.
``` {.python}
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
```

این کلاس باید متدهای __init__() و call() داشته باشد.

در متود __init__() سه لایه متراکم را مقداردهی اولیه می کند: یکی برای حالت decoder ("واحد" اندازه است)، دیگری برای خروجی های encoder ("واحدها" اندازه است) و دیگری برای شبکه کاملا متصل ( یک گره).

در متد call() حالت decoder را مقداردهی اولیه می کنیم  با گرفتن حالت پنهان encoder نهایی. حالت پنهان رسیور ایجاد شده را از یک لایه متراکم عبور می دهیم. همچنین، خروجی های encoder را از طریق لایه متراکم دیگر وصل می کنیم. هر دو خروجی را اضافه می کنیم، آنها را در یک tanh فعال می کنیم و آنها را به لایه کاملا متصل متصل می کنیم. این لایه کاملاً متصل دارای یک گره است. بنابراین، خروجی نهایی دارای ابعاد batch_size * max_length دنباله * 1 است.

بعداً، softmax را روی خروجی شبکه کاملاً متصل اعمال می کنیم تا وزن توجه attention شود.

با انجام مجموع وزنی وزن های توجه و خروجی های encoder، بردار متن را محاسبه می کنیم.
``` {.python}
# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```

``` {.python}
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
```

In the __init__() method, initialize the batch size, decoder units, embedding dimension, GRU layer, and a Dense layer. Also, create an instance of the BahdanauAttention class.

In the call() method:

Call the attention forward propagation and capture the context vector and attention weights.
Send the target token through an embedding layer.
Concatenate the embedded output and context vector.
Plug the output into the GRU layer and then into a fully-connected layer.

``` {.python}
# Decoder class
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # x shape == (batch_size, 1)
    # hidden shape == (batch_size, max_length)
    # enc_output shape == (batch_size, max_length, hidden_size)

    # context_vector shape == (batch_size, hidden_size)
    # attention_weights shape == (batch_size, max_length, 1)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
```

``` {.python}
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
```

``` {.python}
# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Loss function
def loss_function(real, pred):

  # Take care of the padding. Not all sequences are of equal length.
  # If there's a '0' in the sequence, the loss is being nullified
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
```
وزنه های مدل را در طول train بررسی می کند. این به بازیابی خودکار وزن ها در حین ارزیابی مدل کمک می کند.
``` {.python}
import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
```
بعد، فرآیند را تعریف می کنیم. ابتدا کلاس Encoder را فراخوانی می کنیم و خروجی های انکودر و hidden state را دریافت می کنیم. ورودی decoder را شروع می کنیم تا توکن start در تمام توالی های ورودی پخش شود (با استفاده از BATCH_SIZE نشان داده شده است). از تکنیک teacher forcing برای تکرار روی تمام حالت های decoder با جمله هدف به عنوان ورودی بعدی استفاده می کنیم. این حلقه تا زمانی ادامه می یابد که هر توکن در دنباله هدف (انگلیسی) بازدید شود.
``` {.python}
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  # tf.GradientTape() -- record operations for automatic differentiation
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # dec_hidden is used by attention, hence is the same enc_hidden
    dec_hidden = enc_hidden

    # <start> token is the initial decoder input
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):

      # Pass enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # Compute the loss
      loss += loss_function(targ[:, t], predictions)

      # Use teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  # As this function is called per batch, compute the batch_loss
  batch_loss = (loss / int(targ.shape[1]))

  # Get the model's variables
  variables = encoder.trainable_variables + decoder.trainable_variables

  # Compute the gradients
  gradients = tape.gradient(loss, variables)

  # Update the variables of the model/network
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
```
حال با کمک توابعی که تعریف کردیم عملیات یادگیری شبکه عصبی را شروع می کنیم
``` {.python}
import time

EPOCHS = 15

# Training loop
for epoch in range(EPOCHS):
  start = time.time()

  # Initialize the hidden state
  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  # Loop through the dataset
  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

    # Call the train method
    batch_loss = train_step(inp, targ, enc_hidden)

    # Compute the loss (per batch)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # Save (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  # Output the loss observed until that epoch
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

    Epoch 1 Batch 0 Loss 2.7630
    Epoch 1 Batch 100 Loss 1.6684
    Epoch 1 Batch 200 Loss 1.7350
    Epoch 1 Batch 300 Loss 1.6030
    Epoch 1 Batch 400 Loss 1.6787
    Epoch 1 Batch 500 Loss 1.6509
    Epoch 1 Batch 600 Loss 1.6336
    Epoch 1 Loss 1.7289
    Time taken for 1 epoch 298.90009474754333 sec

    Epoch 2 Batch 0 Loss 1.5861
    Epoch 2 Batch 100 Loss 1.6154
    Epoch 2 Batch 200 Loss 1.5725
    Epoch 2 Batch 300 Loss 1.5762
    Epoch 2 Batch 400 Loss 1.5064
    Epoch 2 Batch 500 Loss 1.5651
    Epoch 2 Batch 600 Loss 1.4641
    Epoch 2 Loss 1.5456
    Time taken for 1 epoch 263.264696598053 sec

    Epoch 3 Batch 0 Loss 1.3799
    Epoch 3 Batch 100 Loss 1.4678
    Epoch 3 Batch 200 Loss 1.4675
    Epoch 3 Batch 300 Loss 1.4294
    Epoch 3 Batch 400 Loss 1.4046
    Epoch 3 Batch 500 Loss 1.3885
    Epoch 3 Batch 600 Loss 1.3746
    Epoch 3 Loss 1.4233
    Time taken for 1 epoch 262.26785111427307 sec

    Epoch 4 Batch 0 Loss 1.3060
    Epoch 4 Batch 100 Loss 1.3583
    Epoch 4 Batch 200 Loss 1.2776
    Epoch 4 Batch 300 Loss 1.3386
    Epoch 4 Batch 400 Loss 1.3214
    Epoch 4 Batch 500 Loss 1.2472
    Epoch 4 Batch 600 Loss 1.3868
    Epoch 4 Loss 1.3059
    Time taken for 1 epoch 263.7838613986969 sec

    Epoch 5 Batch 0 Loss 1.2007
    Epoch 5 Batch 100 Loss 1.1707
    Epoch 5 Batch 200 Loss 1.2458
    Epoch 5 Batch 300 Loss 1.1927
    Epoch 5 Batch 400 Loss 1.2090
    Epoch 5 Batch 500 Loss 1.1831
    Epoch 5 Batch 600 Loss 1.1692
    Epoch 5 Loss 1.2010
    Time taken for 1 epoch 262.56413316726685 sec

    Epoch 6 Batch 0 Loss 1.0212
    Epoch 6 Batch 100 Loss 1.1379
    Epoch 6 Batch 200 Loss 1.1623
    Epoch 6 Batch 300 Loss 1.0853
    Epoch 6 Batch 400 Loss 1.1929
    Epoch 6 Batch 500 Loss 1.0925
    Epoch 6 Batch 600 Loss 1.0344
    Epoch 6 Loss 1.1087
    Time taken for 1 epoch 263.696368932724 sec

    Epoch 7 Batch 0 Loss 1.0271
    Epoch 7 Batch 100 Loss 1.0646
    Epoch 7 Batch 200 Loss 1.0462
    Epoch 7 Batch 300 Loss 0.9995
    Epoch 7 Batch 400 Loss 1.0297
    Epoch 7 Batch 500 Loss 1.0406
    Epoch 7 Batch 600 Loss 0.9847
    Epoch 7 Loss 1.0272
    Time taken for 1 epoch 263.0390362739563 sec

    Epoch 8 Batch 0 Loss 0.9265
    Epoch 8 Batch 100 Loss 0.9507
    Epoch 8 Batch 200 Loss 0.9366
    Epoch 8 Batch 300 Loss 0.9202
    Epoch 8 Batch 400 Loss 0.9176
    Epoch 8 Batch 500 Loss 1.0103
    Epoch 8 Batch 600 Loss 0.9096
    Epoch 8 Loss 0.9547
    Time taken for 1 epoch 263.8208107948303 sec

    Epoch 9 Batch 0 Loss 0.9503
    Epoch 9 Batch 100 Loss 0.8482
    Epoch 9 Batch 200 Loss 0.9408
    Epoch 9 Batch 300 Loss 0.8915
    Epoch 9 Batch 400 Loss 0.8691
    Epoch 9 Batch 500 Loss 0.9171
    Epoch 9 Batch 600 Loss 0.8744
    Epoch 9 Loss 0.8896
    Time taken for 1 epoch 263.05375146865845 sec

    Epoch 10 Batch 0 Loss 0.8122
    Epoch 10 Batch 100 Loss 0.8016
    Epoch 10 Batch 200 Loss 0.8313
    Epoch 10 Batch 300 Loss 0.7877
    Epoch 10 Batch 400 Loss 0.8288
    Epoch 10 Batch 500 Loss 0.8239
    Epoch 10 Batch 600 Loss 0.8755
    Epoch 10 Loss 0.8311
    Time taken for 1 epoch 263.84706926345825 sec

    Epoch 11 Batch 0 Loss 0.7728
    Epoch 11 Batch 100 Loss 0.7927
    Epoch 11 Batch 200 Loss 0.7673
    Epoch 11 Batch 300 Loss 0.8133
    Epoch 11 Batch 400 Loss 0.7325
    Epoch 11 Batch 500 Loss 0.8296
    Epoch 11 Batch 600 Loss 0.7859
    Epoch 11 Loss 0.7788
    Time taken for 1 epoch 263.0757508277893 sec

    Epoch 12 Batch 0 Loss 0.7100
    Epoch 12 Batch 100 Loss 0.6900
    Epoch 12 Batch 200 Loss 0.7450
    Epoch 12 Batch 300 Loss 0.6903
    Epoch 12 Batch 400 Loss 0.7128
    Epoch 12 Batch 500 Loss 0.6814
    Epoch 12 Batch 600 Loss 0.7730
    Epoch 12 Loss 0.7315
    Time taken for 1 epoch 263.9645149707794 sec

    Epoch 13 Batch 0 Loss 0.6290
    Epoch 13 Batch 100 Loss 0.6813
    Epoch 13 Batch 200 Loss 0.6544
    Epoch 13 Batch 300 Loss 0.6947
    Epoch 13 Batch 400 Loss 0.7711
    Epoch 13 Batch 500 Loss 0.6922
    Epoch 13 Batch 600 Loss 0.7112
    Epoch 13 Loss 0.6906
    Time taken for 1 epoch 263.22304677963257 sec

    Epoch 14 Batch 0 Loss 0.5882
    Epoch 14 Batch 100 Loss 0.6184
    Epoch 14 Batch 200 Loss 0.6548
    Epoch 14 Batch 300 Loss 0.6481
    Epoch 14 Batch 400 Loss 0.6914
    Epoch 14 Batch 500 Loss 0.6970
    Epoch 14 Batch 600 Loss 0.6823
    Epoch 14 Loss 0.6538
    Time taken for 1 epoch 263.85633516311646 sec

    Epoch 15 Batch 0 Loss 0.5936
    Epoch 15 Batch 100 Loss 0.5842
    Epoch 15 Batch 200 Loss 0.6288
    Epoch 15 Batch 300 Loss 0.6090
    Epoch 15 Batch 400 Loss 0.5889
    Epoch 15 Batch 500 Loss 0.6377
    Epoch 15 Batch 600 Loss 0.6460
    Epoch 15 Loss 0.6218
    Time taken for 1 epoch 263.0172333717346 sec

## **پیاده سازی فرآِند ترجمه**

این تابع ابتدا جمله وردی را تمیز می کند سپس توکنایز می کند و به مدل می دهد
``` {.python}
import numpy as np

# Evaluate function -- similar to the training loop
def evaluate(sentence):

  # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  # Preprocess the sentence given
  sentence = clean_sentence(sentence)

  # Fetch the indices concerning the words in the sentence and pad the sequence
  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  # Convert the inputs to tensors
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  # Loop until the max_length is reached for the target lang (ENGLISH)
  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # Store the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    # Get the prediction with the maximum attention
    predicted_id = tf.argmax(predictions[0]).numpy()

    # Append the token to the result
    result += targ_lang.index_word[predicted_id] + ' '

    # If <end> token is reached, return the result, input, and attention plot
    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # The predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
```

این تابع هم کار نمایش ورودی و خروجی را انجام میدهد 
``` {.python}
# Translate function (which internally calls the evaluate function)
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
 # plot_attention(attention_plot, sentence.split(' '), result.split(' '))
```

``` {.python}
# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

``` {.python}
translate(u"hello world")
translate(u"read useful book")
translate(u"I love you")
translate(u"you are my friend")
translate(u"I play football")
translate(u"I go fishing")
translate(u"where are you")
translate(u"where is my mom")
translate(u"what is your job")
translate(u"comming soon")
translate(u"bye my friend")


```
نمونه ای از عملکرد مترجم



    Input: <start> hello world <end>
    Predicted translation: آهاي ، دنيا <end> 

    Input: <start> read useful book <end>
    Predicted translation: کتاب بخون <end> 

    Input: <start> i love you <end>
    Predicted translation: من دوستت دارم <end> 

    Input: <start> you are my friend <end>
    Predicted translation: تو دوست مني <end> 

    Input: <start> i play football <end>
    Predicted translation: من بازي بازي ميكنم <end>

    Input: <start> i go fishing <end>
    Predicted translation: من ميرم ماهيگيري <end> 

    Input: <start> where are you <end>
    Predicted translation: کجايي <end> 

    Input: <start> where is my mom <end>
    Predicted translation: مادرم کجاست <end> 

    Input: <start> what is your job <end>
    Predicted translation: كارت چيه <end> 

    Input: <start> comming soon <end>
    Predicted translation: داره به زودي <end> 

    Input: <start> bye my friend <end>
    Predicted translation: خداحافظ ، دوست من <end> 

