import requests
import sys
from funcoes_chama_classificadores import *
from deep_translator import GoogleTranslator


#token expira 1 em 1 hora
access_token = ""
#post_id = "1458059541211795"

num_argumentos = len(sys.argv)
if(num_argumentos != 4):
    print("\n!! Erro - São necessários 3 argumentos, o id da publicação e a quantidade de comentários a obter e a chave da api! !! \n")
    exit()

# funcção para dizer se é spam ou não. Recebe array com as probabilidades de cada label
def e_spam_sklearn(arr):
    if(arr[0][0] >= 0.5):
        return "Não é Spam"
    else:
        return "É spam"

# funcção para dizer se é spam ou não
def e_spam_valor(valor):
    if(valor <= 0.5):
        return "Não é Spam"
    else:
        return "É spam"  

post_id = sys.argv[1]
quantidade_comentarios = sys.argv[2]
access_token = sys.argv[3]

request = requests.get('https://graph.facebook.com/' + str(post_id) + "/comments?access_token=" + access_token + "&limit=" + str(quantidade_comentarios))

comentarios = []
# obter comentários
if(request.status_code == 200):
    respostaJson = request.json()
    for valor in respostaJson["data"]:
        comentarios.append(GoogleTranslator(source='auto', target='en').translate(str(valor["message"])))


try:
    #obter classificações sklearn - árvore decisão
    classificacoes_dtree = classifica_sklearn_dtree(comentarios)
    classificacoes_rlogistica = classifica_sklearn_rlogistica(comentarios)
    classificacoes_randomforest = classifica_sklearn_randomforest(comentarios)
    classificacoes_pytorch = classifica_pytorch(comentarios)

    spam_por_modelo = [0,0,0,0] # idx 0 = dtree, 1 = reglogistica, 2 = random forest, 3 = pytorch
    media_cometario = []
    qnt_media_spam = 0

    # mostra resultado na consola
    for idx, comentario in enumerate(comentarios):
        contagem_spam_coment = 0
        if(e_spam_sklearn(classificacoes_dtree[idx]) == "É spam"):
            spam_por_modelo[0] += 1
            contagem_spam_coment += 1
        if(e_spam_sklearn(classificacoes_rlogistica[idx]) == "É spam"):
            spam_por_modelo[1] += 1
            contagem_spam_coment += 1
        if(e_spam_sklearn(classificacoes_randomforest[idx]) == "É spam"):
            spam_por_modelo[2] += 1
            contagem_spam_coment += 1
        if(e_spam_valor(classificacoes_pytorch[idx]) == "É spam"):
            spam_por_modelo[3] += 1
            contagem_spam_coment += 1
        
        if(contagem_spam_coment > 2):
            media_cometario.append("SPAM")
            qnt_media_spam += 1
        else:
            media_cometario.append("NÃO É SPAM")

        print("\n\n" + comentario)
        print("\t SKLEARN ÁRVORE DECISÃO: " + e_spam_sklearn(classificacoes_dtree[idx]))
        print("\t SKLEARN REGRESSÃO LOGISTICA: " + e_spam_sklearn(classificacoes_rlogistica[idx]))
        print("\t SKLEARN RANDOM FOREST: " + e_spam_sklearn(classificacoes_randomforest[idx]))
        print("\t SKLEARN PYTORCH: " + e_spam_valor(classificacoes_pytorch[idx]))
        print("\t MÉDIA: " + media_cometario[idx])


    print("\n\nQuantidade de classificados como spam por classificador:")
    print("\tÁrvore decisão: " + str(spam_por_modelo[0]))
    print("\tRegressão logistica: " + str(spam_por_modelo[1]))
    print("\tRandom forest: " + str(spam_por_modelo[2]))
    print("\tPytorch: " + str(spam_por_modelo[3]))
    print("\tQuantidade de publicações com média SPAM: " + str(qnt_media_spam))
        
except Exception as e:
    print("\n!! Erro - Ocorreu um erro a obter dados do facebook! !!\n")
    print(e)