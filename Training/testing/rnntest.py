from datetime import datetime
import pickle
import string
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences

file = open("data.txt", "r", encoding="utf8")
lines = []

for i in file:
    lines.append(i)

data = ""

for i in lines:
    data = ' '.join(lines)

data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
new_data = data.translate(translator)

z = []

for i in data.split():
    if i not in z:
        z.append(i)

data = ' '.join(z)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i - 1:i + 1]
    sequences.append(words)

sequences = np.array(sequences)
max_sequence_len = max([len(x) for x in sequences])

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
checkpoint = ModelCheckpoint("model.best.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

model.fit(X, y, validation_split=0.05, batch_size=128, epochs=1, shuffle=True)

text = "Hello World"
token_list = tokenizer.texts_to_sequences([text])[0]
token_list = pad_sequences([token_list], maxlen = max_sequence_len-1, padding="pre")

prediction = model.predict(token_list)

print(prediction)

for word,index in tokenizer.word_index.items():
    for predicted in prediction[0]:
        if index == predicted:
            output = word
            print(output)
            break
    



