import scrapy
from funcoes_chama_classificadores import *

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

######
#
# MAIN
#
######
print("======================================")
print("A iniciar a obtenção de comentários...")
print("======================================")
# subclasse da "Spider" para a criação de um spider específico para o IMDb

class ImdbSpider(scrapy.Spider):
    name = "spiderimdb"

    # construtor
    def __init__(self, *args, **kwargs):
        super(ImdbSpider, self).__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]  # obter url da consola

    def parse(self, response):
        # seletor CSS para cada conteiner de comentário
        SET_SELECTOR = '.review-container'
        comentarios = []
        # para cada area de comentário
        for areaComentario in response.css(SET_SELECTOR):
            if(areaComentario.css(".text")):
                comentarios.append(areaComentario.css(".text").extract()[0].replace(
                    '<div class="text show-more__control">', "").replace('</div>', "").replace('<br>', ""))  # extrair o comentario

        if not comentarios:
            print("======================================")
            print("Não existem comentários a obter na página indicada")
            print("======================================")
            return

        # já se possui os comentários, classificar...
        print("======================================")
        print("Comentários encontrados! A classificar...")
        print("======================================")
        try:
            # obter comentários
            classificacoes_dtree = classifica_sklearn_dtree(comentarios)
            classificacoes_rlogistica = classifica_sklearn_rlogistica(comentarios)
            classificacoes_randomforest = classifica_sklearn_randomforest(comentarios)
            classificacoes_pytorch = classifica_pytorch(comentarios)

            # idx 0 = dtree, 1 = reglogistica, 2 = random forest, 3 = pytorch
            spam_por_modelo = [0, 0, 0, 0]
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
                print(("\n\n" + comentario).encode('ascii', 'xmlcharrefreplace'))
                print(("\t SKLEARN ÁRVORE DECISÃO: " + e_spam_sklearn(classificacoes_dtree[idx])))
                print(("\t SKLEARN REGRESSÃO LOGISTICA: " + e_spam_sklearn(classificacoes_rlogistica[idx])))
                print(("\t SKLEARN RANDOM FOREST: " + e_spam_sklearn(classificacoes_randomforest[idx])))
                print(("\t SKLEARN PYTORCH: " + e_spam_valor(classificacoes_pytorch[idx])))
                print(("\t MÉDIA: " + media_cometario[idx]))

            print(("\n\nQuantidade de classificados como spam por classificador:"))
            print(("\tÁrvore decisão: " + str(spam_por_modelo[0])))
            print(("\tRegressão logistica: " + str(spam_por_modelo[1])))
            print(("\tRandom forest: " + str(spam_por_modelo[2])))
            print(("\tPytorch: " + str(spam_por_modelo[3])))
            print(("\tQuantidade de publicações com média SPAM: " + str(qnt_media_spam)))
            print(("\tTotal de comentários: " + str(len(comentarios))))


        except Exception as e:
            print("\n!! Erro - Ocorreu um erro a obter dados do imdb! !!\n")
            print(e)
