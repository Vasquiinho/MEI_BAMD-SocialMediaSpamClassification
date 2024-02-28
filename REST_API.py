from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_jsonpify import jsonify
from funcoes_chama_classificadores import *
import urllib.request
import bs4
import re
from flask_cors import CORS
import requests
from deep_translator import GoogleTranslator
from googleapiclient.discovery import build
import scrapy
from scrapy.crawler import CrawlerProcess, CrawlerRunner 
import os
import subprocess
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

#
#
# CARREGAR VARIÁVEIS (pré-carregar classificadores em memória para tornar a execução de pedidos mais rápida)
#
#
import pickle
import torch
import torch.nn as nn
modelo_arvore_decisao = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_93.22%.DTree", 'rb'))
modelo_reg_logistica = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_95.29%.RLogistica", 'rb'))
modelo_random_forest = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_96.62%.RForest", 'rb'))

class classifier(nn.Module):
    #definir camadas a usar no modelo
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        #construtor
        super().__init__()          
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        #activation function
        self.act = nn.Sigmoid()
        

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #concatenar estados escondidos
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        #funções ativação finais
        outputs=self.act(dense_outputs)
        
        return outputs

#número de parametros "treinaveis"
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#definição da metrica
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
# função para treino
def train(model, iterator, optimizer, criterion):
    #inicializar epoch
    epoch_loss = 0
    epoch_acc = 0
    
    #colocar modelo em modo treino
    model.train()  
    for batch in iterator:
        #resets the gradients after every batch
        optimizer.zero_grad()   
        #retrieve text and no. of words
        text, text_lengths = batch.texto   
        #convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()  
        #compute the loss
        loss = criterion(predictions, batch.spam)        
        #compute the binary accuracy
        acc = binary_accuracy(predictions, batch.spam)   
        #backpropage the loss and compute the gradients
        loss.backward()       
        #update the weights
        optimizer.step()      
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# função de avaliação do modelo
def evaluate(model, iterator, criterion):
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    #deactivating dropout layers
    model.eval()
    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            #retrieve text and no. of words
            text, text_lengths = batch.texto
            #convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()
            #compute loss and accuracy
            loss = criterion(predictions, batch.spam)
            acc = binary_accuracy(predictions, batch.spam)
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2


import random
from torchtext.legacy import data
TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

NOME_FICHEIRO_DADOS = "dados_forma_aleatoria.csv"
# campos a carregar e tratamento a realizar
fields = [('texto',TEXT),('spam', LABEL)]
# importar dados
FORMATO = "csv"
dados = data.TabularDataset(path = NOME_FICHEIRO_DADOS, format = FORMATO, fields = fields, skip_header = True)
train_data, valid_data = dados.split(split_ratio=0.8, random_state = random.seed(2021))
TEXT.build_vocab(train_data, min_freq=3, vectors = "glove.6B.100d")  
LABEL.build_vocab(train_data)
size_of_vocab = len(TEXT.vocab)

modelo_pytoch = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, bidirectional = True, dropout = dropout)
modelo_pytoch.load_state_dict(torch.load("modelos_pytorch/iteracao_1/classificador_trainLoss_0.07_trainAcc_0.97_validLoss_0.09_validAcc_0.96.pytoch"))
modelo_pytoch.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_pytoch = modelo_pytoch.to(device)

import spacy
nlp = spacy.load('en_core_web_sm')
def predict_pytorch(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()




#
#
# DEFINIÇÃO DE ROTAS DA API
#
#

@app.route('/', methods=['GET'])  
def index():
    return jsonify({"msg": "API TP Business Analytics e Mineração de Dados - 20766"})


@app.route('/geral', methods=['POST'])  
def geral():
    dados = request.form.to_dict()
    if len(dados) == 0 or "texto" not in dados:
        return jsonify({"status": False, "msg": "Dados em falta!"})
    #se chegou aqui, fazer a classificação
    texto = dados["texto"]
    #obtem classificações
    res_pytorch = predict_pytorch(modelo_pytoch, texto)
    res_sklearn_dtree = modelo_arvore_decisao.predict_proba([texto])
    res_sklearn_rlogistica = modelo_reg_logistica.predict_proba([texto])
    res_sklearn_rforst = modelo_random_forest.predict_proba([texto])
    # prepara objeto a devolver
    res_classificadores = {
        "sklearn_dtree": {
            "prob": res_sklearn_dtree[0][1] if res_sklearn_dtree[0][0] < 0.5 else res_sklearn_dtree[0][0],
            "label": "Spam" if res_sklearn_dtree[0][0] < 0.5 else "Não é spam"
        },
        "sklearn_rlogistica": {
            "prob": res_sklearn_rlogistica[0][1] if res_sklearn_rlogistica[0][0] < 0.5 else res_sklearn_rlogistica[0][0],
            "label": "Spam" if res_sklearn_rlogistica[0][0] < 0.5 else "Não é spam"
        },
        "sklearn_rforest": {
            "prob": res_sklearn_rforst[0][1] if res_sklearn_rforst[0][0] < 0.5 else res_sklearn_rforst[0][0],
            "label": "Spam" if res_sklearn_rforst[0][0] < 0.5 else "Não é spam"
        },
        "pytorch": {
            "prob": res_pytorch if res_pytorch > 0.5 else 1 - res_pytorch,
            "label": "Spam" if res_pytorch > 0.5 else "Não é spam"
        }
    }
    # devolve resultado
    return jsonify({"status": True, "resultado": res_classificadores})


@app.route('/fb', methods=['POST'])  
def fb():
    dados = request.form.to_dict()
    if len(dados) != 3 or "idpub" not in dados or "chave" not in dados or "limite" not in dados:
        return jsonify({"status": False, "msg": "Dados em falta!"})
    
    #se chegou aqui, obter dados FB
    post_id = dados["idpub"]
    access_token = dados["chave"]
    quantidade_comentarios = dados["limite"]
    requestfb = requests.get('https://graph.facebook.com/' + str(post_id) + "/comments?access_token=" + access_token + "&limit=" + str(quantidade_comentarios))
    comentarios = []
    # obter comentários
    if(requestfb.status_code == 200):
        respostaJson = requestfb.json()
        for valor in respostaJson["data"]:
            comentarios.append(GoogleTranslator(source='auto', target='en').translate(str(valor["message"])))
    else:
        return jsonify({"status": False, "msg": "Ocorreu um problema a obter os dados do Facebook!", "err": requestfb})

    if len(comentarios) == 0:
        return jsonify({"status": False, "msg": "Publicação sem comentários"})

    resultado = []
    # para cada comentário
    for comentario in comentarios:
        #obtem classificações
        res_pytorch = predict_pytorch(modelo_pytoch, comentario)
        res_sklearn_dtree = modelo_arvore_decisao.predict_proba([comentario])
        res_sklearn_rlogistica = modelo_reg_logistica.predict_proba([comentario])
        res_sklearn_rforst = modelo_random_forest.predict_proba([comentario])
        resultado.append({
            "comentario": comentario,
            "res_classificadores": {
                "sklearn_dtree": {
                    "prob": res_sklearn_dtree[0][1] if res_sklearn_dtree[0][0] < 0.5 else res_sklearn_dtree[0][0],
                    "label": "Spam" if res_sklearn_dtree[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rlogistica": {
                    "prob": res_sklearn_rlogistica[0][1] if res_sklearn_rlogistica[0][0] < 0.5 else res_sklearn_rlogistica[0][0],
                    "label": "Spam" if res_sklearn_rlogistica[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rforest": {
                    "prob": res_sklearn_rforst[0][1] if res_sklearn_rforst[0][0] < 0.5 else res_sklearn_rforst[0][0],
                    "label": "Spam" if res_sklearn_rforst[0][0] < 0.5 else "Não é spam"
                },
                "pytorch": {
                    "prob": res_pytorch if res_pytorch > 0.5 else 1 - res_pytorch,
                    "label": "Spam" if res_pytorch > 0.5 else "Não é spam"
                }
            }
        })
    
    # devolve resultado
    return jsonify({"status": True, "resultado": resultado})


@app.route('/yt', methods=['POST'])  
def yt():
    dados = request.form.to_dict()
    if len(dados) != 3 or "idvideo" not in dados or "chave" not in dados or "limite" not in dados:
        return jsonify({"status": False, "msg": "Dados em falta!"})
    
    #se chegou aqui, obter dados YT
    video_id = dados["idvideo"]
    api_key = dados["chave"]
    quantidade = dados["limite"]

    # obter comentários
    comentarios = []
    # criar acesso aos dados do youtube
    youtube = build('youtube', 'v3', developerKey=api_key)
    # obter dados video
    video_response = youtube.commentThreads().list(part='snippet,replies',videoId=video_id, maxResults=quantidade).execute()
    # para cada item no resultado 
    for item in video_response['items']:
        # extrair comentario
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comentarios.append(comment)

    if len(comentarios) == 0:
        return jsonify({"status": False, "msg": "Publicação sem comentários"})

    resultado = []
    # para cada comentário
    for comentario in comentarios:
        #obtem classificações
        res_pytorch = predict_pytorch(modelo_pytoch, comentario)
        res_sklearn_dtree = modelo_arvore_decisao.predict_proba([comentario])
        res_sklearn_rlogistica = modelo_reg_logistica.predict_proba([comentario])
        res_sklearn_rforst = modelo_random_forest.predict_proba([comentario])
        resultado.append({
            "comentario": comentario,
            "res_classificadores": {
                "sklearn_dtree": {
                    "prob": res_sklearn_dtree[0][1] if res_sklearn_dtree[0][0] < 0.5 else res_sklearn_dtree[0][0],
                    "label": "Spam" if res_sklearn_dtree[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rlogistica": {
                    "prob": res_sklearn_rlogistica[0][1] if res_sklearn_rlogistica[0][0] < 0.5 else res_sklearn_rlogistica[0][0],
                    "label": "Spam" if res_sklearn_rlogistica[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rforest": {
                    "prob": res_sklearn_rforst[0][1] if res_sklearn_rforst[0][0] < 0.5 else res_sklearn_rforst[0][0],
                    "label": "Spam" if res_sklearn_rforst[0][0] < 0.5 else "Não é spam"
                },
                "pytorch": {
                    "prob": res_pytorch if res_pytorch > 0.5 else 1 - res_pytorch,
                    "label": "Spam" if res_pytorch > 0.5 else "Não é spam"
                }
            }
        })
    
    # devolve resultado
    return jsonify({"status": True, "resultado": resultado})


@app.route('/reddit', methods=['POST'])  
def reddit():
    dados = request.form.to_dict()
    if len(dados) != 1 or "reddit_link" not in dados:
        return jsonify({"status": False, "msg": "Dados em falta!"})
    
    #se chegou aqui, obter dados Reddit
    reddit_link = dados["reddit_link"]

   #obter página
    pagina=urllib.request.urlopen(reddit_link).read().decode('utf-8','ignore')
    
    #parse aos dados obtidos com o request
    soup=bs4.BeautifulSoup(pagina, features="html.parser")

    #extrair "àreas" onde estão os comentários
    comentarios=soup.find_all("div", {"data-test-id": "comment"})

    array_comentarios = []
    for comentario in comentarios:
        #comentário está no primeiro "div", e depois no primero "p"
        if comentario.find_all("div")[0].find_all("p")[0].string:
            comentario_txt = comentario.find_all("div")[0].find_all("p")[0].string.strip()
        else:
            continue
        
        # remover pontuação
        comentario_txt = re.sub(r'[^\w\s\\\/]',' ',comentario_txt)
        comentario_txt = re.sub(r'  +',' ',comentario_txt)

        array_comentarios.append(comentario_txt)

    resultado = []
    # para cada comentário
    for comentario in array_comentarios:
        #obtem classificações
        res_pytorch = predict_pytorch(modelo_pytoch, comentario)
        res_sklearn_dtree = modelo_arvore_decisao.predict_proba([comentario])
        res_sklearn_rlogistica = modelo_reg_logistica.predict_proba([comentario])
        res_sklearn_rforst = modelo_random_forest.predict_proba([comentario])
        resultado.append({
            "comentario": comentario,
            "res_classificadores": {
                "sklearn_dtree": {
                    "prob": res_sklearn_dtree[0][1] if res_sklearn_dtree[0][0] < 0.5 else res_sklearn_dtree[0][0],
                    "label": "Spam" if res_sklearn_dtree[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rlogistica": {
                    "prob": res_sklearn_rlogistica[0][1] if res_sklearn_rlogistica[0][0] < 0.5 else res_sklearn_rlogistica[0][0],
                    "label": "Spam" if res_sklearn_rlogistica[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rforest": {
                    "prob": res_sklearn_rforst[0][1] if res_sklearn_rforst[0][0] < 0.5 else res_sklearn_rforst[0][0],
                    "label": "Spam" if res_sklearn_rforst[0][0] < 0.5 else "Não é spam"
                },
                "pytorch": {
                    "prob": res_pytorch if res_pytorch > 0.5 else 1 - res_pytorch,
                    "label": "Spam" if res_pytorch > 0.5 else "Não é spam"
                }
            }
        })
    
    # devolve resultado
    return jsonify({"status": True, "resultado": resultado})




@app.route('/imdb', methods=['POST'])  
def imdb():
    dados = request.form.to_dict()
    if len(dados) != 1 or "imdb_link" not in dados:
        return jsonify({"status": False, "msg": "Dados em falta!"})
    
    #se chegou aqui, obter dados IMDb
    imdb_link = dados["imdb_link"]

    # comando subprocess para correr o scrapy
    cmd = f'scrapy runspider ImdbSpider_api.py -a start_url="{imdb_link}" --nolog'
    # executa o comando
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # espera até terminar
    process.wait()
    # obter resultados
    data, err = process.communicate()

    if process.returncode is 0:
        data.decode('utf-8')
    else:
        return jsonify({"status": False, "msg": "Ocorreu um problema a obter os comentários"})

    json_obj = json.loads(data)
    if json_obj["status"] != "True" and json_obj["status"] != True:
        return jsonify({"status": False, "msg": "Ocorreu um problema a obter os comentários"})

    # se sucesso, ler ficheiro com comentários
    # (cria ficheiro porque demora muito a obter o output inteiro pelo subprocess)
    with open('data.json') as json_file:
        json_coments = json.load(json_file)

    # remover ficheiro após ler comentários
    os.remove("data.json")

    comentarios_imdb = json_coments

    if len(comentarios_imdb) == 0:
        return jsonify({"status": False, "msg": "Não foram obtidos comentários"})

    resultado = []
    for comentario in comentarios_imdb:
        #obtem classificações
        res_pytorch = predict_pytorch(modelo_pytoch, comentario)
        res_sklearn_dtree = modelo_arvore_decisao.predict_proba([comentario])
        res_sklearn_rlogistica = modelo_reg_logistica.predict_proba([comentario])
        res_sklearn_rforst = modelo_random_forest.predict_proba([comentario])
        resultado.append({
            "comentario": comentario,
            "res_classificadores": {
                "sklearn_dtree": {
                    "prob": res_sklearn_dtree[0][1] if res_sklearn_dtree[0][0] < 0.5 else res_sklearn_dtree[0][0],
                    "label": "Spam" if res_sklearn_dtree[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rlogistica": {
                    "prob": res_sklearn_rlogistica[0][1] if res_sklearn_rlogistica[0][0] < 0.5 else res_sklearn_rlogistica[0][0],
                    "label": "Spam" if res_sklearn_rlogistica[0][0] < 0.5 else "Não é spam"
                },
                "sklearn_rforest": {
                    "prob": res_sklearn_rforst[0][1] if res_sklearn_rforst[0][0] < 0.5 else res_sklearn_rforst[0][0],
                    "label": "Spam" if res_sklearn_rforst[0][0] < 0.5 else "Não é spam"
                },
                "pytorch": {
                    "prob": res_pytorch if res_pytorch > 0.5 else 1 - res_pytorch,
                    "label": "Spam" if res_pytorch > 0.5 else "Não é spam"
                }
            }
        })
    
    # devolve resultado
    return jsonify({"status": True, "resultado": resultado})

if __name__ == '__main__':
     app.run(port='5002')