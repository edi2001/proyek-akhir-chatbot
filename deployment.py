#import library
from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
nltk.download('popular')
import json
import random

lemmatizer = WordNetLemmatizer()

#load data
model = load_model("chatbot_model.h5")
intents = json.loads(open("dataset.json").read())
kata = pickle.load(open("kata.pkl", "rb"))
kelas = pickle.load(open("kelas.pkl", "rb"))

def kalimat_bersih(kalimat):
    # tokenize pola - pisahkan kata-kata menjadi array
    token = nltk.word_tokenize(kalimat)
    # stem setiap kata - buat bentuk pendek untuk kata
    token = [lemmatizer.lemmatize(kata.lower()) for kata in token]
    return token

def vektorisasi_kata(kalimat, kata, detail=True):
    # tokenisasi  pattern
    token = kalimat_bersih(kalimat)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(kata)  
    for s in token:
        for i,w in enumerate(kata):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if detail:
                    print ("found in bag: %s" % w)
    return(np.array(bag))



def prediksi(kalimat, model):
    # filter out predictions below a threshold
    pred = vektorisasi_kata(kalimat, kata,detail=False)
    res = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25
    hasil = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    hasil.sort(key=lambda x: x[1], reverse=True)
    array_hasil = []
    for r in hasil:
        array_hasil.append({"intent": kelas[r[0]], "probability": str(r[1])})
    return array_hasil

def respon(x, dataset):
    tag = x[0]['intent']
    isi_dataset = dataset['intents']
    for i in isi_dataset:
        if(i['tag']== tag):
            hasil = random.choice(i['responses'])
            break
    return hasil

def responBot(msg):
    x = prediksi(msg, model)
    r = respon(x, intents)
    return r

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def ambil_data():
    user = request.args.get('pesan')
    return responBot(user)


if __name__ == "__main__":
    app.run(debug=True)


