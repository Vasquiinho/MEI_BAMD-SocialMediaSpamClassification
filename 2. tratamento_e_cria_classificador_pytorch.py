########################################
#
# Script para a realização do tratamento final ao texto e para a criação do modelo classificador PYTORCH
#
#######################################
from pathlib import Path
import torch
from torchtext.legacy import data
from sklearn.model_selection import train_test_split
import torch.optim as optim
import random
import torch.nn as nn
import pickle


##########
#
# VARIÁVEIS CONTROLÁVEIS
#
##########
# seed para se obter sempre os mesmos resultados
SEED = 2021
QNT_DADOS_TREINO = 0.8
FORMATO = "csv"
BATCH_SIZE = 64
N_EPOCHS = 5
qnt_a_criar = 2



torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# descobrir se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA DISPONÍVEL?: " + str(torch.cuda.is_available()))

# definir o preprocessamento que será realizado pelo pytorchtext ao importar dados
TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)


NOME_FICHEIRO_DADOS = "dados_forma_aleatoria.csv"
# campos a carregar e tratamento a realizar
fields = [('texto',TEXT),('spam', LABEL)]
# importar dados
dados = data.TabularDataset(path = NOME_FICHEIRO_DADOS, format = FORMATO, fields = fields, skip_header = True)


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

##########
#
# CRIAÇÃO CLASSIFICADORES
#
##########
print("\n\n======= A inciar criação de " + str(qnt_a_criar) + " classificadores com dados aleatórios diferentes, com SEED a inciar em " + str(SEED) + " e a incrementar por 1 em cada iteração =====")
print("% Treino: " + str(QNT_DADOS_TREINO * 100))
print("% Teste: " + str(100 - QNT_DADOS_TREINO * 100))
qnt_criados = 0
while qnt_criados < qnt_a_criar:
    qnt_criados = qnt_criados+1
    print("===== A inciar iteração número " + str(qnt_criados) + " de " + str(qnt_a_criar) + " =====")
    caminho_base='modelos_pytorch/iteracao_' + str(qnt_criados) + "/"
    Path(caminho_base).mkdir(parents=True, exist_ok=True)

    # dividir dados
    print("\t=== Divisão de dados para treino e teste ===")
    train_data, valid_data = dados.split(split_ratio=QNT_DADOS_TREINO, random_state = random.seed(SEED))

    TEXT.build_vocab(train_data, min_freq=3, vectors = "glove.6B.100d")  
    LABEL.build_vocab(train_data)
    size_of_vocab = len(TEXT.vocab)

    #Load an iterator
    train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), batch_size = BATCH_SIZE, sort_key = lambda x: len(x.texto), sort_within_batch=True, device = device)
    print("\t=== A criar modelo ===")
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, bidirectional = True, dropout = dropout)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    #push to cuda if available
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        #train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        #evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), caminho_base + "classificador_trainLoss_" + str(round(train_loss,2)) + "_trainAcc_" + str(round(train_acc,2)) + "_validLoss_" + str(round(valid_loss,2)) + "_validAcc_" + str(round(valid_acc,2)) + ".pytoch")
        print(f'\tEPOCH' + str(epoch) + f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    SEED = SEED + 1



#load weights
#model.load_state_dict(torch.load(path))
#model.eval()


#inference 
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


#make predictions
#print(predict(model, "A billionaire GOP donor is fed up with Donald Trump and he s fed up with Republicans in Congress who refuse to do anything about Trump. Now he has a very strongly worded message for Congress   he wants them to do something about the monster in the White House or he ll stop writing them checks. In fact, it might be too late.Florida Cuban-American billionaire Mike Fernandez is disgusted by Trump and in an interview posted on Thursday by Politico, he expressed (shall we say) frustration at Republicans who  balls  to impeach Trump. All the Republicans who hide behind the flag and hide behind the church, they don t have the f  balls to do what it takes,  Fernandez told POLITICO Florida in a telephone interview on Thursday.Fernandez has long been a huge supporter of Republicans, especially at the state level in Florida. He supported Mitt Romney, Florida Governor Rick Scott, Jeb Bush and several others, but he s no longer a Republican because of Trump. In 2016, he spent $3.5 million in an effort to defeat Trump. I am out of the political process. Too disgusted, too expensive, too supportive of ego maniacs whose words have the value of quicksand,  he wrote in an email to a Republican fundraiser seeking political contributions. It is demoralizing to me to see adults worshipping a false idol. I can t continue to write checks for anyone,  he said.  I know what it s like to lose a country. It gets worse. Fernandez called Trump one of the worst words in the English language   for forced-birth Republicans. He called Trump an  abortion of a human being. If I was the doctor and knew what that baby would do, I d have made sure it never would have seen the light of day,  he said of the president.The most interesting thing about all of this is that this might be what it takes for Republicans to wake up. Face it, they aren t going to grow a set of balls (to paraphrase Fernandez) any time in the near future, but they will be terrified of losing their donor base.Billionaires, more than anyone, care about how our country is perceived on the world stage. Chaos means market uncertainty, which is never a good thing for the moneyed class. If Trump continues down his current path, he will soon start losing the confidence of the GOP donor base. If Congress doesn t do something about him, look for all of the Trump sycophant members of Congress to be primaried, by the people who have all the money.Featured image via Alex Wong/Getty Images,"))

#insincere question
#print(predict(model, "Anyone else notice that Megan Fox is in this video?"))