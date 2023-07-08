import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as file:
        words,labels, training, output = pickle.load(file)

except:
    ########################## liste koje cemo koristiti
    words = []
    labels = []
    docs_x = []
    docs_y = []

    ########################## punjenje predvidenih listi
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #ML part -> stemming; 
            # trazenje korijena rijeci; 
            # da to dobijemo moramo napraviti tokenizaciju
            # i dobit cemo listu svakog pojedinacnog pattern-a 
            wrds=nltk.word_tokenize(pattern)

            # sada to pretvaramo u jednu listu
            words.extend(wrds)

            # ovu distinkciju imamo zbog treniranja modela
            docs_x.append(wrds) # ovo su tokenizirani patterni; svaka pojedinacna rijec - token!
            docs_y.append(intent["tag"])  # ovo je lista svih "tagova" koji odgovaraju svakoj pojedinacnoj rijeci
            
        if intent["tag"] not in labels:
            # ovo su samo izdvojeni tagovi koji se ne ponavljaju
            labels.append(intent["tag"])


    ################################ printevi da vidim sto sam tocno dobila
    print("WORDS: sve pojedinacne rijeci iz patterna")
    print(words)

    print("DOC_X: svi tokenizirani patterns")
    print(docs_x)

    print("DOCY_Y: tag koji odgovara patternu")
    print(docs_y)

    print("labels: svaki pojedinacni tag")
    print(labels)



    ########################## uredivanje rijeci
    # potrebno je ukloniti duplikate rijeci 
    # i napraviti stemming (hvatanje korjena rijeci) da vidimo velicinu vokabulara naseg modela

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # set uklanja duplikate
    # list ih pretvara u listu (iz set vrste podataka)
    # sorted ih sortira
    words = sorted(list(set(words)))

    ########################## sortirani labelsi
    labels = sorted(labels)


    ##########################  ONE HOT ENCODING
    # one hot encoded: 
    # ML i neural network ne moze raditi s rijecima nego moze raditi s brojevima i zato trebamo vektore
    # koristimo OHE koji je u obliku brojeva u listi te oznacava prisustvo ili odsustvo rijeci
    # lista ce biti duljine koliko imamo rijeci, a svaka pozicija oznacava postoji li rijec ili ne

    ######## radimo bag of words odnosno liste trening i output rijeci u obliku 0 i 1
    training = []   
    output = []    # testing output - lista s 0 i 1 koja govori postoji li rijec ili ne!

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):  #doc_x su svi patterni, x - index
        bag = []

        # lista svakog pojedinacnog slova
        wrds = [stemmer.stem(w) for w in doc]


        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1    # labels - jedinstveni tagovi, docs_y - svi tagovi

        training.append(bag)
        output.append(output_row)

    ############################## pretvaranje listi u array
    # uzimamo liste i pretvaramo ih u array da ih mozemo koristiti u modelu
    training = np.array(training)
    output = np.array(output)   

    with open("data.pickle", "wb") as file:
        pickle.dump((words,labels, training, output), file)


############################### IZRADA MODELA; AI aspekt koda
# ovaj ce model u principu predviđati iz kojeg cemo "tag" uzeti odgovor useru:

# resetiranje prethodne sesije (za svaki slucaj)
tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()

# neuralnetwork: input layer koji ce biti duljine traning data
net = tflearn.input_data(shape=[None, len(training[0])]) # ovo je input shape koji ocekujemo za nas model
# hidden layer
# broj osam oznacava da imamo 8 neurona,
# a fully connected znaci da je svaki od input data povezan sa svakim neuronom
net = tflearn.fully_connected(net, 8) 
# hidden layer
# i ovdje 8 oznacava broj neurona i svaki neuron iz prethodnog layera je povezan sa svakim od ovih 8
net = tflearn.fully_connected(net, 8)
# output  layer (ali isto hidden)
# ovaj pak layer ima 6 neurona (duljina outputa) i ovdje je prethodni layer povezan sa svakim neuronom ovog layera
# poseban je zbog "softmaxa"; svi ce ovi neuroni proci kroz njega i dobit cemo vjerojatnost (probability za svaki od pojedinog neurona)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # dobit cemo vjerojatnost svakog neurona u tom layeru
net = tflearn.regression(net)

model = tflearn.DNN(net) # ovo je vrste neuronske mreze koju smo odabrali za ovaj model

try:
    model.load("model.ftlearn")
except:
    #treniranje modela
    model.fit(training, output, n_epoch=300, batch_size=8, show_metric=True)
    #spremanje modela
    model.save("model.tflearn")

############################ MAKE PREDICTION

# najprije radimo funkciju za bag of words od user inputa
def bag_of_words_from_user(recenica, words):
    bag = [0 for _ in range(len(words))]

    rijeci_u_recenici = nltk.word_tokenize(recenica)
    rijeci_u_recenici = [stemmer.stem(rijec.lower) for rijec in rijeci_u_recenici]

    for pojedina_rijec in rijeci_u_recenici:
        for index, word in enumerate(words):
            if word == pojedina_rijec:
                bag[index].append(1)

    return np.array(bag)