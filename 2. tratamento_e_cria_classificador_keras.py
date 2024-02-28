########################################
#
# Script para a realização do tratamento final ao texto e para a criação do modelo classificador KERAS
#
#######################################
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import os
from pathlib import Path
from sklearn import metrics
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import warnings
#warnings.filterwarnings('error')
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.regularizers import L1L2
from sklearn.feature_extraction.text import CountVectorizer

##########
#
# VARIÁVEIS CONTROLÁVEIS
#
##########
passar_tratamento = True
random_state = 10
inc_rand_state = 1
versao_classificador = 1
qnt_modelos_criar = 2
dados_teste = 0.15
num_epoch = 2
batch_size = 4500



##########
#
# TRATAMENTO DADOS
#
##########
dados = pd.read_csv("dados_forma_aleatoria.csv")

if not passar_tratamento:
    #verifica a existencia de colunas com valores nullos. Se existir, remover
    print("\n\n=== A verificar existência de valores null ===")
    if dados.isnull().values.any():
        print("\t> Valores nulos encontrados. A remover linhas...")
        dados = dados.dropna()
    else:
        print("\t> Sem valores nulos encontrados.")

    print("================== Concluído =================")

    # distribuição de valores
    print("\n\n=== Distribuição de valores ===")
    print("\t0 = real")
    print("\t1 = spam/fake")
    print(dados["spam"].value_counts())
    print("Total Linhas: " + str(dados["spam"].count()))
    print("===============================")

    # limpeza do texto
    print("\n\n=== Limpeza do texto ===")
    # # converter texto para minusculas
    print("\t> Converter para minusculas")
    dados["texto"] = dados["texto"].str.lower()

    # # remover pontuação e carateres especiais
    #print("\t> A remover carateres especiais e pontuação")
    #dados["texto"] = dados["texto"].str.replace(r"\W", " ", regex=True)
    # comentado, pois muito spam tem links aleatorios

    print("\t> A Remover multiplos espaços")
    dados["texto"] = dados["texto"].str.replace(r"\s+", " ", regex=True)

    # # remover "stopwords"
    print("\t> A Remover stopwords")
    lista_stopwords = stopwords.words('english')
    dados['texto'] = dados['texto'].apply(lambda frase: " ".join([palavra for palavra in frase.split() if palavra not in (lista_stopwords)]))

    # # lematizar palavras
    print("\t> A lematizar")
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = WordNetLemmatizer()
    dados['texto'] = dados['texto'].apply(lambda frase: " ".join([stemmer.lemmatize(palavra) for palavra in w_tokenizer.tokenize(frase)]))

    print("================== Concluído =================")

    # guardar e carregar - para evitar fazer sempre o tratamento aos dados, se já tivermos esses dados tratados
    # guardar dados tratados
    print("\t> A guardar dados tratados")
    dados.to_csv("dados_tratados.csv", index=False)
    print("\t> Concluido")
else:
    # carregar dados tratados
    print("A passar tratamento de dados... A carregar dados tratados previamente")
    dados = pd.read_csv("dados_tratados.csv")



##########
#
# CRIAÇÃO CLASSIFICADORES
#
##########
print("\n\n======= A inciar criação de " + str(qnt_modelos_criar) + " classificadores com dados aleatórios diferentes, com random_state a inciar em " + str(random_state) + " e a incrementar por " + str(inc_rand_state) + " em cada iteração =====")
print("% Treino: " + str(100 - dados_teste * 100))
print("% Teste: " + str(dados_teste * 100))


qnt_criados = 0
while qnt_modelos_criar > qnt_criados:
    qnt_criados += 1

    print("===== A inciar iteração número " + str(qnt_criados) + " de " + str(qnt_modelos_criar) + " =====")
    caminho_base = "./modelos_keras/interacao_" + str(qnt_criados) + "/"
    Path(caminho_base).mkdir(parents=True, exist_ok=True)

    # DIVISÃO DOS DADOS
    dados.dropna(inplace=True)
    print("\t=== Divisão de dados para treino e teste ===")

    sentences_train, sentences_test, y_train, y_test = train_test_split(dados["texto"], dados["spam"], test_size=dados_teste, random_state=random_state)

    print("\t\t> A guardar dados de treino e teste")
    with open(caminho_base + 'sentences_train.dados', 'wb') as f:
        pickle.dump(sentences_train, f)
    with open(caminho_base + 'sentences_test.dados', 'wb') as f:
        pickle.dump(sentences_test, f)
    with open(caminho_base + 'Y_train.dados', 'wb') as f:
        pickle.dump(y_train, f)
    with open(caminho_base + 'Y_test.dados', 'wb') as f:
        pickle.dump(y_test, f)
    print("\t======== Divisão de dados concluida! "+ str(100-dados_teste*100) +"% treino/"+ str(dados_teste*100) +"% teste ========")

    #Classificadores
    print("\t=== A criar classificador ===")
    # # # Vetorização e TF-IDF (term frequency-inverse document frequency)
    print("\t> Classsificador:")
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=num_epoch, verbose=True, validation_data=(X_test, y_test), batch_size=batch_size)

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(test_accuracy))

    model.save(caminho_base + 'modeloKeras_epoch'+ str(num_epoch)+'_batchsize'+str(batch_size)+'_TrLoss'+ str(round(train_loss,2)) +'_TrAcc' + str(round(train_accuracy,2)) + '_TeLoss' + str(round(test_loss,2)) + '_TeAcc' + str(round(test_accuracy,2)) + '.keras')
        
    random_state += inc_rand_state
    versao_classificador += 1