#spam Classfier-imports
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Data-Handling
mails = pd.read_csv('spam.csv', encoding = 'latin-1')
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
mails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
mails.drop(['labels'], axis = 1, inplace = True)

#train-test Split
totalMails = 4825 + 747
trainIndex, testIndex = list(), list()
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]
trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)


#visualization of Common words
spam_words = ' '.join(list(mails[mails['label'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

#visualization of Common words not spam
ham_words = ' '.join(list(mails[mails['label'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# Converting x_train to integer token, token is just an index number that can uniquely identify a particular word
max_len = 2000 
max_feature = 50000
tokenizer = Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(trainData.iloc[:,0])
x_train_features = np.array(tokenizer.texts_to_sequences(trainData.iloc[:,0]))
x_test_features = np.array(tokenizer.texts_to_sequences(testData.iloc[:,0]))
x_train_features = pad_sequences(x_train_features,maxlen=max_len)
x_test_features = pad_sequences(x_test_features,maxlen=max_len)

embed_size = 100  # how many unique words to use (i.e num rows in embedding vector)
#Model
inp = Input(shape=(max_len,))
x = Embedding(max_feature, embed_size)(inp)
x = Bidirectional(LSTM(10, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(8, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train_features, trainData.iloc[:,1], batch_size=512, epochs=10, validation_data=(x_test_features, testData.iloc[:,1]))


## Neural Network
ytest=testData.iloc[:,1].tolist()
from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score
y_predict  = [1 if o>0.5 else 0 for o in model.predict(x_test_features)]
confusion_matrix(ytest,y_predict)
tn, fp, fn, tp = confusion_matrix(ytest,y_predict).ravel()
print("Precision: {:.2f}%".format(100 * precision_score(ytest, y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(ytest, y_predict)))
f1_score(ytest,y_predict)

