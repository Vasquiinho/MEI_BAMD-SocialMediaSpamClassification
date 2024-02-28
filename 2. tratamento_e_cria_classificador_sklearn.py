########################################
#
# Script para a realização do tratamento final ao texto e para a criação do modelo classificador SKLEARN
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
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
import pickle
import warnings
warnings.filterwarnings('error')


##########
#
# VARIÁVEIS CONTROLÁVEIS
#
##########
passar_tratamento = True
random_state = 10
inc_rand_state = 1
mostrar_plt = False
guardar_plt = True
versao_classificador = 1
criar_RegLogistica = False
criar_DecisionTree = False
criar_RandomForest = False
criar_RegLinear = True
qnt_modelos_criar = 5
dados_teste = 0.15


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
print("Modelo Regressão Logistica: " + str(criar_RegLogistica))
print("Modelo Árvore de Desisão: " + str(criar_DecisionTree))
print("Modelo Regressão Linear: " + str(criar_RegLinear))
print("Modelo Random Forest: " + str(criar_RandomForest))

qnt_criados = 0
while qnt_modelos_criar > qnt_criados:
    qnt_criados += 1

    print("===== A inciar iteração número " + str(qnt_criados) + " de " + str(qnt_modelos_criar) + " =====")
    caminho_base = "./modelos_sklearn/interacao_" + str(qnt_criados) + "/"
    Path(caminho_base).mkdir(parents=True, exist_ok=True)

    # DIVISÃO DOS DADOS
    dados.dropna(inplace=True)
    print("\t=== Divisão de dados para treino e teste ===")

    X_train, X_test, y_train, y_test = train_test_split(dados["texto"], dados["spam"], test_size=dados_teste, random_state=random_state)

    print("\t\t> A guardar dados de treino e teste")
    with open(caminho_base + 'X_train.dados', 'wb') as f:
        pickle.dump(X_train, f)
    with open(caminho_base + 'X_test.dados', 'wb') as f:
        pickle.dump(X_train, f)
    with open(caminho_base + 'Y_train.dados', 'wb') as f:
        pickle.dump(X_train, f)
    with open(caminho_base + 'Y_test.dados', 'wb') as f:
        pickle.dump(X_train, f)
    print("\t======== Divisão de dados concluida! "+ str(100-dados_teste*100) +"% treino/"+ str(dados_teste*100) +"% teste ========")


    #Classificadores
    print("\t=== A criar classificadores ===")
    if criar_RegLogistica:
        # # Regressão logistica
        # # # Vetorização e TF-IDF (term frequency-inverse document frequency)
        print("\t> Classsificador Regressão Logistica:")
        pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression(max_iter=200, random_state=random_state))])

        nao_converge = False
        try:
            model = pipe.fit(X_train, y_train) # treinar
        except ConvergenceWarning:
            nao_converge = True

        prediction = model.predict(X_test)
        print("\t\t>Precisão: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

        print("\t\tA guardar classificador...")
        if nao_converge:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_NaoConverge_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RLogistica"
        else:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RLogistica"

        
        with open(nome_ficheiro_modelo, 'wb') as f:
            pickle.dump(model, f)
        #cria matriz confusao
        disp = plot_confusion_matrix(model, X_test, y_test, display_labels=['Fake', 'Real'])
        disp.ax_.set_title("Regressão logistica - versão " + str(versao_classificador) + " - " + str(round(accuracy_score(y_test, prediction)*100,2)) + "%")
        if mostrar_plt:
            print("\t\t\tA mostrar Matriz da Confusão...")
            plt.show()
        if guardar_plt:
            plt.savefig(caminho_base + "MatrizConfusao_RegLogistica_v" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.png")
    plt.close()

    if criar_RegLinear:
        # # Regressão linear
        # # # Vetorização e TF-IDF (term frequency-inverse document frequency)
        print("\t> Classsificador Regressão Linear:")
        pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LinearRegression())])

        nao_converge = False
        try:
            model = pipe.fit(X_train, y_train) # treinar
        except ConvergenceWarning:
            nao_converge = True
        
        prediction = model.predict(X_test)
        print("\t\t>Precisão: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

        print("\t\tA guardar classificador...")
        if nao_converge:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_NaoConverge_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RLinear"
        else:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RLinear"

        with open(nome_ficheiro_modelo, 'wb') as f:
            pickle.dump(model, f)
        disp = plot_confusion_matrix(model, X_test, y_test, display_labels=['Fake', 'Real'])
        disp.ax_.set_title("Regressão Linear - versão " + str(versao_classificador) + " - " + str(round(accuracy_score(y_test, prediction)*100,2)) + "%")
        if mostrar_plt: 
            print("\t\t\tA mostrar Matriz da Confusão...")
            plt.show()
        if guardar_plt:
            plt.savefig(caminho_base + "MatrizConfusao_RegLinear_v" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.png")
    plt.close()

    if criar_DecisionTree:
        # # Arvore Decisao
        print("\t> Classsificador Árvore de Decisão:")
        pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', DecisionTreeClassifier(criterion= 'entropy', max_depth = 20,  splitter='best', random_state=random_state))])
        
        nao_converge = False
        try:
            model = pipe.fit(X_train, y_train) # treinar
        except ConvergenceWarning:
            nao_converge = True
        
        prediction = model.predict(X_test)
        print("\t\t>Precisão: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

        print("\t\tA guardar classificador...")
        if nao_converge:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_NaoConverge_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.DTree"
        else:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.DTree"

        with open(nome_ficheiro_modelo, 'wb') as f:
            pickle.dump(model, f)
        disp = plot_confusion_matrix(model, X_test, y_test, display_labels=['Fake', 'Real'])
        disp.ax_.set_title("Árvore de Decisão - versão " + str(versao_classificador) + " - " + str(round(accuracy_score(y_test, prediction)*100,2)) + "%")
        if mostrar_plt: 
            print("\t\t\tA mostrar Matriz da Confusão...")
            plt.show()
        if guardar_plt:
            plt.savefig(caminho_base + "MatrizConfusao_ArvDecisao_v" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.png")
    plt.close()

    if criar_RandomForest:
        # # Arvore Decisao
        print("\t> Classsificador RandomForest:")
        pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=random_state))])
        
        nao_converge = False
        try:
            model = pipe.fit(X_train, y_train) # treinar
        except ConvergenceWarning:
            nao_converge = True
        
        prediction = model.predict(X_test)
        print("\t\t>Precisão: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

        print("\t\tA guardar classificador...")
        if nao_converge:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_NaoConverge_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RForest"
        else:
            nome_ficheiro_modelo = caminho_base + "cl_v_" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.RForest"

        with open(nome_ficheiro_modelo, 'wb') as f:
            pickle.dump(model, f)
        disp = plot_confusion_matrix(model, X_test, y_test, display_labels=['Fake', 'Real'])
        disp.ax_.set_title("RandomForest - versão " + str(versao_classificador) + " - " + str(round(accuracy_score(y_test, prediction)*100,2)) + "%")
        if mostrar_plt: 
            print("\t\t\tA mostrar Matriz da Confusão...")
            plt.show()
        if guardar_plt:
            plt.savefig(caminho_base + "MatrizConfusao_RandForest_v" + str(versao_classificador) + "_" + str(round(accuracy_score(y_test, prediction)*100,2)) + "%.png")
        
    plt.close()
    random_state += inc_rand_state
    #versao_classificador += 1