import json
import os
from tensorflow.keras.models import load_model
from google.cloud import storage
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unicodedata import normalize
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 

# Set the GCS bucket and model file name
bucket_name = 'model-eda-bucket'
model_file = 'model.h5'

# Initialize NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Create a GCS client
client = storage.Client()
#encode
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Cliente Desconoce Como Actualizar Datos', 'Información Requisitos Y Documentos', 'Orientación En El Uso De La App', 'Usuario No Pasa Filtro Se Seguridad App Billetera Móvil']

# Load the model
def load_model_from_gcs():
    # Get the GCS bucket
    bucket = client.bucket(bucket_name)

    # Download the model file to a temporary location
    temp_location = '/tmp/model.h5'
    blob = bucket.blob(model_file)
    blob.download_to_filename(temp_location)

    # Load the model from the temporary location
    model = load_model(temp_location)

    # Delete the temporary file
    os.remove(temp_location)

    return model

# Preprocess the input text


def text_preprocessor(df, colname_nlp='', remove_stopwords=True, remove_accents=True, remove_salutations=True):
    stop_words = set(stopwords.words('spanish'))
    salutations = ['señores compensar', 'quedo atenta', 'porfavor',
                   'recibir respuesta', 'pronta respuesta', 'cordial saludo',
                   'cordiales saludos', 'reciba un cordial saludo', 'afectuosos saludos',
                   'cordialmente', 'atentamente', 'Estimado',
                   'Estimada', 'Distinguido', 'buenos dias', 'buen dia',
                   'buenas noches', 'buena noche', 'buenas tardes', 'buena tarde', 'muchas gracias de antemano',
                   'muchas gracias', 'muchisimas gracias', 'gracias',
                   'por favor']
    
    if isinstance(df, pd.DataFrame):
        X = df[colname_nlp]
        X = X.copy()
        X['text_processed'] = X[colname_nlp]
        
        if remove_stopwords:
            X['text_processed'] = X['text_processed'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))
        
        if remove_accents:
            X['text_processed'] = X['text_processed'].apply(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8'))
        
        if remove_salutations:
            for salutation in salutations:
                X['text_processed'] = X['text_processed'].str.replace(salutation, '')
        
        X['text_processed'] = X['text_processed'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        return X['text_processed']
    
    elif isinstance(df, str):
        text = df
        
        if remove_stopwords:
            text = ' '.join([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
        
        if remove_accents:
            text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        if remove_salutations:
            for salutation in salutations:
                text = text.replace(salutation, '')
        
        text = re.sub(r'[^\w\s]', '', text)
        
        return text    

# Define the Cloud Function
def causal_salida(request):
    # Load the model
    model = load_model_from_gcs()

    # Parse the request JSON
    request_json = request.get_json(silent=True)
    calls = request_json['calls']
    replies = []

    # Perform predictions
    for call in calls:
        input_text = call[0]
        input_sequence = tokenizer.texts_to_sequences([text_preprocessor(df=input_text)])
        input_padded = pad_sequences(input_sequence, maxlen=648)


        # Perform prediction
        predicted_probs = model.predict(input_padded)
        predicted_label_index = predicted_probs.argmax()
        predicted_label = label_encoder.classes_[predicted_label_index]

        replies.append(predicted_label)

    return json.dumps({'replies': replies})

################################################################





import json
from datetime import datetime
import calendar
from tensorflow.keras.models import load_model
from google.cloud import storage
from google.auth import compute_engine
import sys 
import subprocess
from google.colab import auth
from google.cloud import bigquery
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unicodedata import normalize
import re
import string
import nltk
from nltk.corpus import stopwords
from google.cloud import bigquery


label_encoder = LabelEncoder()
label_encoder.classes_ = ['Cliente Desconoce Como Actualizar Datos', 'Información Requisitos Y Documentos', 'Orientación En El Uso De La App', 'Usuario No Pasa Filtro Se Seguridad App Billetera Móvil']
client = storage.Client(project)
bucket = client.bucket('model-eda-bucket')
# Download the model file to a temporary location
temp_location = '/tmp/model.h5'
blob = bucket.blob(model_file)
blob.download_to_filename(temp_location)
model = load_model(temp_location)
os.remove(temp_location)




def text_preprocessor(df, colname_nlp='', remove_stopwords=True, remove_accents=True, remove_salutations=True):
    stop_words = set(stopwords.words('spanish'))
    salutations = ['señores compensar', 'quedo atenta', 'porfavor',
                   'recibir respuesta', 'pronta respuesta', 'cordial saludo',
                   'cordiales saludos', 'reciba un cordial saludo', 'afectuosos saludos',
                   'cordialmente', 'atentamente', 'Estimado',
                   'Estimada', 'Distinguido', 'buenos dias', 'buen dia',
                   'buenas noches', 'buena noche', 'buenas tardes', 'buena tarde', 'muchas gracias de antemano',
                   'muchas gracias', 'muchisimas gracias', 'gracias',
                   'por favor']
    
    if isinstance(df, pd.DataFrame):
        X = df[colname_nlp]
        X = X.copy()
        X['text_processed'] = X[colname_nlp]
        
        if remove_stopwords:
            X['text_processed'] = X['text_processed'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))
        
        if remove_accents:
            X['text_processed'] = X['text_processed'].apply(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8'))
        
        if remove_salutations:
            for salutation in salutations:
                X['text_processed'] = X['text_processed'].str.replace(salutation, '')
        
        X['text_processed'] = X['text_processed'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        return X['text_processed']
    
    elif isinstance(df, str):
        text = df
        
        if remove_stopwords:
            text = ' '.join([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
        
        if remove_accents:
            text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        if remove_salutations:
            for salutation in salutations:
                text = text.replace(salutation, '')
        
        text = re.sub(r'[^\w\s]', '', text)
        
        return text    

def causal_salida(request, model):
    request_json = request.get_json(silent=True)
    print('Req Json',type(request_json))
    replies = []    
    calls = request_json['calls']
    for call in calls:
        input_text = call[0]
        # farm_year = call[1]
        input_sequence = tokenizer.texts_to_sequences([text_preprocessor(df=input_text)])
        input_padded = pad_sequences(input_sequence, maxlen=648)

        # Perform prediction
        predicted_probs = model.predict(input_padded)
        predicted_label_index = predicted_probs.argmax()
        predicted_label = label_encoder.classes_[predicted_label_index]
        replies.append(predicted_label)

    return json.dumps({'replies': [str(x) for x in replies]})       

 