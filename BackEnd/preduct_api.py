import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from flask import Flask
from flask_cors import CORS, cross_origin

maxWordBuffer = 3

try:
    tokenizer = Tokenizer()
    data = open('Training\data.txt').read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    modelname = 'saved_model\model171221_0.7211'
    model = tf.keras.models.load_model(modelname)
except Exception as e:
    print(e)

def next_word(pre_set):
    token_list = tokenizer.texts_to_sequences([pre_set])[0]
    token_list = pad_sequences([token_list], maxlen=16, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    classes=np.argmax(predicted,axis=1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == classes:
          output_word = word
          break
    return output_word


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/')
@cross_origin(supports_credentials=True)
def hello():
    return "Next word prediction API v1.0"

@app.route("/predict/", methods=['GET'])
@cross_origin(supports_credentials=True)
def retunEmpty():
    return ""

@app.route("/predict/<string:input>", methods=['GET'])
@cross_origin(supports_credentials=True)
def predict(input):
    if input != "":
        word_list = input.split()
        if(len(word_list) > maxWordBuffer):
            word_list = word_list[(maxWordBuffer*-1):]
        newString = ' '.join(word for word in word_list)
        return next_word(newString)
    else:
        return ""

if __name__ == '__main__':
    app.run(debug=False)
