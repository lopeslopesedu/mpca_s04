import os
from chatbot import Chatbot
from sys import platform
if platform == "linux":
    barra_sistema = "//"
else:
    barra_sistema = "\\"


def configuracao_treino():
    arquivo = open("."+barra_sistema+"config_treino"+barra_sistema+"configuracao_treinamento.txt","r")
    lista_treinos = []
    treino = []
    for linha in arquivo:
        treino = []
        treino = linha.split()
        lista_treinos.append(treino)

    arquivo.close()
    del(arquivo,linha,treino)
    return lista_treinos

def configuracao_base():
    arquivo = open("."+barra_sistema+"config_treino"+barra_sistema+"configuracao_base.txt","r")
    linhas = int(arquivo.readline())
    return linhas

def treinar_lista_modelos(lista_treinos):
    for treino in lista_treinos:
        chatbot = Chatbot()
        chatbot.tamanho_perguntas = int(treino[0])
        chatbot.celulas = treino[1]
        chatbot.epocas = treino[2]
        chatbot.rede = treino[3]
        if(treino[4]=="GERAL"):
            chatbot.base_dados.usar_base_geral = True
        else:
            chatbot.base_dados.usar_base_geral = False
        if (treino[5] == "ESPECIALISTA"):
            chatbot.base_dados.usar_base_especialista = True
        else:
            chatbot.base_dados.usar_base_especialista = False
        #treinar somente uma rede
        chatbot.treinar()
        #salva o chatbot
        chatbot.salvar_modelo()
        chatbot.salvar_dados()
        print("Rodada de Treino Concluída com Sucesso")
        print("----------------------------------------------------------------------------")
        del (chatbot)

if __name__ == '__main__':

    while(True):
        print("1 - Treinar chatbots")
        print("2 - Iterar chatbots")
        print("3 - Calcular Bleu")
        print("4 - Calcular PP")
        print("0 - sair")

        menu = int(input("Informe a Opção desejada: "))

        if(menu == 1):
            lista_treinos = configuracao_treino()
            treinar_lista_modelos(lista_treinos)
        if(menu == 2):
            chatbot = Chatbot.carregar_chatbots_treinados()
            chatbot.interar()
            #limpa o ambiente
            del(chatbot)
        if (menu == 3):
            chatbot = Chatbot.carregar_chatbots_treinados()
            chatbot.calcular_Bleu()
            # limpa o ambiente
            del (chatbot)
        if (menu == 4):
            chatbot = Chatbot.carregar_chatbots_treinados()
            chatbot.calcular_PP()
            # limpa o ambiente
            del (chatbot)
        if(menu == 0):
            exit()