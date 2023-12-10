import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)

# Tokenize words and build documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))
# Print information about the data
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to pickle files
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)  # Empty array for output

# Create bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]  # List of tokenized words for the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]  # Lemmatize each word
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)  # Create bag of words array
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1  # Set output row for current tag
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Convert training data to pandas DataFrame
df = pd.DataFrame(training, columns=['X', 'Y'])
train_x = df['X'].tolist()
train_y = df['Y'].tolist()
print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit and save the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("Model created")