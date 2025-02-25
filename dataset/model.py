import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


import nltk
nltk.download('stopwords')


# Function for stemming
def stemming(content):
    if not isinstance(content, str):  
        content = str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# Data loading and preprocessing
column_names = ['text', 'category', 'target']
sm_data = pd.read_csv(r'f:/Project/new/dataset/tanglish.csv', names=column_names, encoding='ISO-8859-1')


port_stem = PorterStemmer()
sm_data['stemmed_content'] = sm_data['text'].apply(stemming)


# Separating the data and labels
X = sm_data['stemmed_content'].values
Y = sm_data['target'].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)



# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Model training
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


# Accuracy scores
training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))

print("Training accuracy:", training_data_accuracy)
print("Test accuracy:", test_data_accuracy)



# Save the model and vectorizer
pickle.dump(model, open('trained_model.sav', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

import joblib
joblib.dump(model, "trained_model.sav")
