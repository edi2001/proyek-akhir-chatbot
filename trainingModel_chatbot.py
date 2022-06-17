#import library
import json
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import numpy as np
import random
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras import callbacks 

Lm = WordNetLemmatizer()

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

#inisialisasi variabel
abai_kata=["?","!","."]
kata=[]
doc=[]
kelas=[]
file_dataset=open("dataset.json").read()
dataset=json.loads(file_dataset)

# menggunakan looping untuk mengakses data dalam dataset

for data in dataset["intents"]:
    for pertanyaan in data["patterns"]:

    #mengambil setiap kalimat dalam dataset untuk di tokenisasi menjadi kata-kata
     token= nltk.word_tokenize(pertanyaan)
     kata.extend(token)
     #menambahkan kalimat yang telah ditokenisasi ke dalam variabel array doc
     doc.append((token,data["tag"]))

     #menambahkan tag ke dalam array kelas
     if data["tag"] not in kelas:
         kelas.append(data["tag"])

#Lemmatisation
kata=[Lm.lemmatize(token.lower())for token in kata if token not in abai_kata]

#set kata yang telah di lemmatisasi
kata=sorted(list(set(kata)))
kelas=sorted(list(set(kelas)))

#menyimpan data
pickle.dump(kata,open("kata.pkl","wb"))
pickle.dump(kelas,open("kelas.pkl","wb"))

train=[]
output_kosong=[0]*len(kelas)

for d in doc:
    bag=[]
    #daftar kata-kata tokenized untuk pola
    pola_kata=d[0]
    # lemmatize setiap  kata dasar, dalam upaya untuk mewakili kata-kata terkait
    pola_kata = [Lm.lemmatize(word.lower()) for word in pola_kata]
    #buat kumpulan kata-kata  dengan angka 1, jika kecocokan kata ditemukan dalam pola saat ini
    for k in kata:
        bag.append(1) if k in pola_kata else bag.append(0)
    # output adalah '0' untuk setiap tag dan '1' untuk tag saat ini (untuk setiap pola)
    output_baris=list(output_kosong)
    output_baris[kelas.index(d[1])] = 1

    train.append([bag,output_baris])

# acak fitur  dan ubah menjadi np.array
random.shuffle(train)
training = np.array(train)
# buat list training dan tes. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Data pelatihan dibuat")

# Buat model - 3 lapisan. Lapisan pertama 128 neuron, 
# lapisan kedua 64 neuron dan lapisan keluaran ke-3  berisi jumlah neuron

model = Sequential()
model.add(Dense(1024, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Kompilasi model. Penurunan gradien stokastik dengan gradien akselerasi 
# Nesterov memberikan hasil yang baik untuk model ini

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
# callbacks =[earlystopping]
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("model telah telah terbuat")

         







