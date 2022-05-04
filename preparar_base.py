import sys,os, csv
from sys import platform
from os import close
from common_functions import clean_text
import pickle

if platform == "linux":
    barra_sistema = "//"
else:
    barra_sistema = "\\"

def carregar_arquivos():
        # Bloco responsavel pela leitura e pre-processamento do dataset
        # carrego todas as linhas do arquivo de conversas
        # exemplo de linha: L867 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ What good stuff?

    linhas = open("."+barra_sistema+"data_set"+barra_sistema+'geral'+barra_sistema+"movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
        # carrego todas as linhas do arquivo de estrutura de conversas
        # exemplo de linha: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L866', 'L867', 'L868', 'L869']

    conversas = open('.'+barra_sistema+'data_set'+barra_sistema+'geral'+barra_sistema+'movie_conversations.txt', encoding='utf-8',
                         errors='ignore').read().split('\n')
    return linhas,conversas

def carregar_arquivos_data_set_especialistas():
    lista_arquivos_pastas =[]
    pasta = '.'+barra_sistema+'data_set'+barra_sistema+'especialista'
    for diretorio, subpastas, arquivos in os.walk(pasta):
        for arquivo in arquivos:
            if(arquivo[-3:]=="csv"):
                lista_arquivos_pastas.append(os.path.join(diretorio, arquivo))

    questoes = []
    respostas = []

    # Bloco responsavel pela leitura e pre-processamento do dataset
    # carrego todas as linhas do arquivo de conversas
    # exemplo de linha: L867 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ What good stuff?

    for arquivo in lista_arquivos_pastas:
        with open(arquivo) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                questoes.append(clean_text(row[1]))
                respostas.append(clean_text(row[2]))
    return questoes,respostas

def pre_processar_texto(linhas,conversas):
    #global tamanho_perguntas_respostas_padrao
    # repare que a linha de exemplo de conversa L867 faz parte de um contexto de dialogo, junto com a L866,L868,L869
    # faço a separação da estrutura de conversa
    estrutura_conversa = []
    for conversa in conversas:
        estrutura_conversa.append(conversa.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",", "").split())
    # exemplo de linha: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L866', 'L867', 'L868', 'L869']
    # pós split: ['L866', 'L867', 'L868', 'L869']
    # pós replace: L866 L867 L868 L869

    # faço a separação de cada uma das linhas de conversa, armazenando o conteúdo pós key
    dialogo = {}
    for linha in linhas:
        dialogo[linha.split(' +++$+++ ')[0]] = linha.split(' +++$+++ ')[-1]

    # limpo as variáveis
    del (linhas, conversas, conversa, linha)

    # monto duas listas, para armazenar as perguntas e respostas
    perguntas = []
    respostas = []
    # nesse data set toda pergunta vem acompanhada de uma resposta
    for conversa in estrutura_conversa:
        for i in range(len(conversa) - 1):
            perguntas.append(dialogo[conversa[i]])
            respostas.append(dialogo[conversa[i + 1]])

    # limpo as variáveis
    del (dialogo, estrutura_conversa, conversa, i)

    # Trabalho apenas com questões menores que 13 caracteres
    perguntas_classificadas = []
    respostas_classificadas = []
    for i in range(len(perguntas)):
        #if len(perguntas[i]) < tamanho_perguntas_respostas_padrao:
        perguntas_classificadas.append(perguntas[i])
        respostas_classificadas.append(respostas[i])

    # listas com os textos tanto das perguntas como das respostas limpas
    perguntas_limpas = []
    respostas_limpas = []

    for linha in perguntas_classificadas:
        perguntas_limpas.append(clean_text(linha))

    for linha in respostas_classificadas:
        respostas_limpas.append(clean_text(linha))

    # limpo as variáveis utilizadas
    del (respostas, perguntas, linha)

    for i in range(len(respostas_limpas)):
        respostas_limpas[i] = ' '.join(respostas_limpas[i].split()[:11])

    # limpo as variáveis utilizadas
    del (respostas_classificadas, perguntas_classificadas)
    return respostas_limpas,perguntas_limpas

def salva_data_sets(data_set_geral,data_set_especialista):
    arquivo = open("."+barra_sistema+"data_set"+barra_sistema+"data_set_geral.obj", 'wb')
    pickle.dump(data_set_geral, arquivo)
    arquivo.close()
    arquivo = open("."+barra_sistema+"data_set"+barra_sistema+"data_set_especialista.obj", 'wb')
    pickle.dump(data_set_especialista, arquivo)
    arquivo.close()

def prepara_bases():
    # carrega arquivos data set GERAL
    linhas, conversas = carregar_arquivos()
    linhas, conversas = pre_processar_texto(linhas, conversas)
    # carrega arquivos data set ESPECIALISTA
    linhas_especialista, respostas_especialistas = carregar_arquivos_data_set_especialistas()
    return [linhas,conversas], [linhas_especialista, respostas_especialistas]

if __name__ == '__main__':
    print("----------------------------------------------------------------------------")
    print("Technical Intelligence Assistant")
    print("Módulo de Preparação de Base de Dados")
    print("----------------------------------------------------------------------------")
    print("Carregando os Arquivos do data set Geral")
    # carrega arquivos data set GERAL
    linhas, conversas = carregar_arquivos()
    linhas, conversas = pre_processar_texto(linhas, conversas)
    print("Carregamento Concluído com sucesso!")
    print("Carregando os Arquivos do data set Especialista")
    # carrega arquivos data set ESPECIALISTA
    linhas_especialista, respostas_especialistas = carregar_arquivos_data_set_especialistas()
    print("Carregamento Concluído com sucesso!")
    print("----------------------------------------------------------------------------")
    print("Foram carregadas:")
    print("Perguntas e Respostas base geral: " + str(len(linhas)))
    print("Perguntas e Respostas base especialistas: " + str(len(linhas_especialista)))
    print("----------------------------------------------------------------------------")
    print("Salvando data_set's!")
    salva_data_sets([linhas,conversas], [linhas_especialista, respostas_especialistas])
    print("Os data_set's foram salvos com sucesso!")
    print("----------------------------------------------------------------------------")