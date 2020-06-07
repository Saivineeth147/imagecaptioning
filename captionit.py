import string
import sys
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


def extract_features(filename, model):
        try:
            imagee = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        imagee = imagee.resize((299,299))
        imagee = np.array(imagee)
        # for images that has 4 channels, we convert them into 3 channels
        if imagee.shape[2] == 4:
            imagee = imagee[..., :3]
        imagee = np.expand_dims(imagee, axis=0)
        imagee = imagee/127.5
        imagee = imagee - 1.0
        feature = model.predict(imagee)
        return feature
def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('model_9.h5')
model._make_predict_function()
xception_model = Xception(include_top=False, pooling="avg")
xception_model._make_predict_function()
def caption_this_image(input_img):
    photo = extract_features(input_img, xception_model)
    descriptionss = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    return descriptionss
