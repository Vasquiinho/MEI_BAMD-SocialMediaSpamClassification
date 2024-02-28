import scrapy
import json
from funcoes_chama_classificadores import *

######
#
# MAIN
#
######

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
        x = 0

        # para cada area de comentário
        for areaComentario in response.css(SET_SELECTOR):
            if(areaComentario.css(".text")):
                x += 1
                comentarios.append(areaComentario.css(".text").extract()[0].replace(
                    '<div class="text show-more__control">', "").replace('</div>', "").replace('<br>', ""))  # extrair o comentario

        if not comentarios:
            print('{"status":"False", "msg":"Erro a obter comentarios"}')
            return

        # já se possui os comentários, classificar...
        print('{"status":"True", "comentarios": "no_ficheiro" }')
        with open('data.json', 'w') as outfile:
            json.dump(comentarios, outfile)
        return