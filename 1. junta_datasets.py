########################################
#
# Script para juntar os datasets num só ficheiro
# - cria colunas "texto" e "spam"
#
#######################################
import pandas as pd

# identificação das colunas do ficheiro final
colunas_finais = ["texto", "spam"]
# identificação das colunas dos ficheiros iniciais correspondentes às colunas do ficheiro final
mapeia_colunas = [["Tweet", "CONTENT", "v2", "text"], ["Type", "CLASS", "v1", "label_num", "spam"]] # grupo 1 fica na coluna 1 das colunas finais, grupo N fica na coluna N das colunas finais
# datasets a importar
datasets_importar = ["./Datasets/sms_spam.csv", "./Datasets/Fake News/Combinados.csv", "./Datasets/email_spam_ham_dataset.csv", "./Datasets/Twitter/dados.csv", "./Datasets/Youtube/Youtube01.csv", "./Datasets/Youtube/Youtube02.csv", "./Datasets/Youtube/Youtube03.csv", "./Datasets/Youtube/Youtube04.csv", "./Datasets/Youtube/Youtube05.csv"]


dict_final = {}
# para cada dataset, procurar a existência de colunas com os nomes em "mapeia_colunas". Se existir, os valores dessa coluna do dataset que está a ser tratado será colocado na coluna correspondente no dataset final
for dataset in datasets_importar:
    dados_atuais = pd.read_csv(dataset)
    for index, nome_coluna_final in enumerate(colunas_finais):
        for nome_coluna in mapeia_colunas[index]:
            if nome_coluna in dados_atuais.columns:
                if nome_coluna_final in dict_final:
                    dict_final[nome_coluna_final].extend(dados_atuais[nome_coluna].tolist())
                else:
                    dict_final[nome_coluna_final] = dados_atuais[nome_coluna].tolist()


dataset_final = pd.DataFrame.from_dict(dict_final)

dataset_final = dataset_final.dropna() #remove linhas com campos vazios

dataset_final = dataset_final.replace(r"\r\n", " ", regex=True) # remove newlines do texto
dataset_final = dataset_final.replace(r"\n", " ", regex=True) # remove newlines do texto
dataset_final = dataset_final.replace(r"\r", " ", regex=True) # remove newlines do texto

#alguns datasets têm "ham" / "Quality" / "spam" escrito na coluna "spam". Trocar por 0 e 1
dataset_final["spam"] = dataset_final["spam"].replace(r"ham", 0, regex=True)
dataset_final["spam"] = dataset_final["spam"].replace(r"Quality", 0, regex=True)
dataset_final["spam"] = dataset_final["spam"].replace(r"spam", 1, regex=True)
dataset_final["spam"] = dataset_final["spam"].replace(r"Spam", 1, regex=True)
 
#guarda dataset final
dataset_final.to_csv("dados.csv", index=False)

#guarda dataset final linhas aleatórias
dataset_final.sample(frac=1).to_csv("dados_forma_aleatoria.csv", index=False)