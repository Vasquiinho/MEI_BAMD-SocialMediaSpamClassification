import pickle

# SKLEARN

modelo_arvore_decisao = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_93.22%.DTree", 'rb'))
modelo_reg_logistica = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_95.29%.RLogistica", 'rb'))
modelo_random_forest = pickle.load(open("modelos_sklearn/interacao_1/cl_v_1_96.62%.RForest", 'rb'))

texto = "Download as many ringtones as u like no restrictions, 1000s 2 choose. U can even send 2 yr buddys. Txt Sir to 80082".lower()



SKLEARN = True
PYTORCH = True
KERAS = False


print("=== Texto Input ===")
print(texto)

if(SKLEARN):
    print("================")
    print("Árvore de Decisão:")
    print("\t Classe (0 - não spam, 1 - spam): spam" if modelo_arvore_decisao.predict([texto]) == 1 else "\t Classe (0 - não spam, 1 - spam): não spam")
    print("\tProbabilidade cada classe [0, 1]: " + str(modelo_arvore_decisao.predict_proba([texto])))
    print("Regressão Logistica:")
    print("\t Classe (0 - não spam, 1 - spam): spam" if modelo_reg_logistica.predict([texto]) == 1 else "\t Classe (0 - não spam, 1 - spam): não spam")
    print("\tProbabilidade cada classe [0, 1]: " + str(modelo_reg_logistica.predict_proba([texto])))
    print("Random Forest:")
    print("\t Classe (0 - não spam, 1 - spam): spam" if modelo_random_forest.predict([texto]) == 1 else "\t Classe (0 - não spam, 1 - spam): não spam")
    print("\tProbabilidade cada classe [0, 1]: " + str(modelo_random_forest.predict_proba([texto])))
    #print("Regressão Linear:")
    #print("\t Classe (0 - não spam, 1 - spam):" + modelo_arvore_decisao.predict([texto]))
    #print("\tProbabilidade cada classe [0, 1]: " + modelo_arvore_decisao.predict_proba([texto]))

if(PYTORCH):
    import torch
    import torch.nn as nn

    ##########
    #
    # DECLARAÇÃO DE FUNÇÕES
    #
    ##########
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

    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, bidirectional = True, dropout = dropout)
    model.load_state_dict(torch.load("modelos_pytorch/iteracao_1/classificador_trainLoss_0.07_trainAcc_0.97_validLoss_0.09_validAcc_0.96.pytoch"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    import spacy
    nlp = spacy.load('en_core_web_sm')
    def predict(model, sentence):
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
        length = [len(indexed)]                                    #compute no. of words
        tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
        tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
        length_tensor = torch.LongTensor(length)                   #convert to tensor
        prediction = model(tensor, length_tensor)                  #prediction 
        return prediction.item()

    print(predict(model, texto))


if(KERAS):
    from tensorflow import keras
    modelKeras = keras.models.load_model('modelos_keras/interacao_1/modeloKeras_epoch2_batchsize4500_TrLoss0.26_TrAcc0.94_TeLoss0.28_TeAcc0.92.keras')
    modelKeras.predict([texto])
