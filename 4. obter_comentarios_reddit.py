import urllib.request
from funcoes_chama_classificadores import *
import bs4
import sys
import re

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

def obter_comentarios(url):
    #obter página
    pagina=urllib.request.urlopen(url).read().decode('utf-8','ignore')
    
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
    
    return array_comentarios

     
def main():
    num_argumentos = len(sys.argv)
    if(num_argumentos != 2):
        print("\n!! Erro - É necessário indicar a URL por argumento! !! \n")
        exit()

    url = sys.argv[1]
    comentarios = obter_comentarios(url)
        
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
        print("\tQuantidade de vídeos com média SPAM: " + str(qnt_media_spam))
        print("\tTotal comentários: " + str(len(comentarios)))
            
    except Exception as e:
        print("\n!! Erro - Ocorreu um erro a obter dados do reddit! !!\n")
        print(e)


if __name__=="__main__": 
    main()