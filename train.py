import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

from string import punctuation
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')
import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random

words=[]
tags = []
words_tags = []

data_file = open('data.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        tag = intent['tag']
        tags.append(tag)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        words_tags.append((w, intent['tag']))

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in punctuation]
#words = [w for w in words if w not in stopwords]

words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

print (len(words_tags), "words_tags")
print (len(tags), "tags", tags)
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))

# initializing training data
training = []
output_empty = np.zeros(len(tags))
for wt in words_tags:
    bag = []
    #get just pattern 
    pattern_words = wt[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[tags.index(wt[1])] = 1

    training.append([bag, output_row])

training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tensorflow.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model_training.h5', hist)

print("model created")