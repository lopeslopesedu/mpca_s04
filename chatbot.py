# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model,load_model
from keras.callbacks import History
from nltk.translate.bleu_score import sentence_bleu
from AttentionLayer import AttentionLayer

from common_functions import clean_text
from prettytable import PrettyTable
from texttable import Texttable
import preparar_base
from preparar_base import prepara_bases
from data_set import Data_set
from datetime import datetime
import numpy as np
import pickle
import sys,os
from tqdm import tqdm
import time
from sys import platform
if platform == "linux":
    barra_sistema = "//"
else:
    barra_sistema = "\\"




class Chatbot:
    def __init__(self):
        self.caminho = ""
        self.data = []
        self.data_final_treinamento = []
        self.rede = "LSTM"
        self.epocas = 20
        self.celulas_LSTM = 400
        self.modelo = ""
        self.tamanho_perguntas=20
        self.tamanho_base = 20000
        self.base_dados = Data_set()
        self.history = []
        self.embed = []
        self.encoder_embed = []
        self.modelo = []
        self.enc_model = []
        self.dec_model = []
        self.dec_dense = []
        self.attn_layer= []
        self.usar_base_geral = True

    def carregar_dados_treino(self):
        self.base_dados.carregar_data_sets()
        self.base_dados.limitar_caracteres(self.tamanho_perguntas)

    def word_embedding(self):
        # contagem de palavras
        # dicionario montado com a contagem de cada uma das palavras, somando perguntas e respostas
        word2count = {}

        for linha in self.base_dados.treino[0]:
            for palavra in linha.split():
                if palavra not in word2count:
                    word2count[palavra] = 1
                else:
                    word2count[palavra] += 1
        for linha in self.base_dados.treino[1]:
            for palavra in linha.split():
                if palavra not in word2count:
                    word2count[palavra] = 1
                else:
                    word2count[palavra] += 1

        # limpo as variaveis de ambiente
        del (palavra, linha)

        # variável utilizada para retirar as palavras menos utilizadas
        lixo = 5

        # monto um dicionário com o vocabulário desse data set
        self.base_dados.vocabulario = {}
        numero_palavras = 0
        for palavra, contador in word2count.items():
            if contador >= lixo:
                self.base_dados.vocabulario[palavra] = numero_palavras
                numero_palavras += 1

        ## limpo as variáveis utilizadas
        del (word2count, palavra, contador, lixo, numero_palavras)

        # adiciono os marcadores de inicio e final de sentenças
        for i in range(len(self.base_dados.treino[1])):
            self.base_dados.treino[1][i] = '<SOS> ' + self.base_dados.treino[1][i] + ' <EOS>'
        # adiciono os tokens ao vocabulário
        tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
        x = len(self.base_dados.vocabulario)
        for token in tokens:
            self.base_dados.vocabulario[token] = x
            x += 1

        # ?
        self.base_dados.vocabulario['cameron'] = self.base_dados.vocabulario['<PAD>']
        self.base_dados.vocabulario['<PAD>'] = 0

        # limpo as váriaveis de ambiente
        del (token, tokens)
        del (x)

        ### inv answers dict ###
        self.base_dados.chaves_vocabulario = {w: v for v, w in self.base_dados.vocabulario.items()}

        # limpo as variaveis utilizadas
        del (i)

        # Passo por todas as linhas de questões feitas
        encoder_input = []
        for linha in self.base_dados.treino[0]:
            lista = []
            # para cada palavra
            for palavra in linha.split():
                # verifico se ela está no vocab
                if palavra not in self.base_dados.vocabulario:
                    # caso nao esteja(count <5) adiciono o token de "saída"
                    lista.append(self.base_dados.vocabulario['<OUT>'])
                else:
                    # caso esteja adiciono a quantidade de aparições de cada palavra
                    lista.append(self.base_dados.vocabulario[palavra])
            # adiciono a lista de encoder input.
            # cada linha da lista tera um lista contendo o número de apareciçoes da cada palavra que compõe aquela linha de pergunta
            encoder_input.append(lista)

        # realizo o mesmo processo para decoder input, porém na lista de respostas
        decoder_input = []
        for linha in self.base_dados.treino[1]:
            lista = []
            for palavra in linha.split():
                if palavra not in self.base_dados.vocabulario:
                    lista.append(self.base_dados.vocabulario['<OUT>'])
                else:
                    lista.append(self.base_dados.vocabulario[palavra])
            decoder_input.append(lista)

        # imprime informações
        print("Quantidade de palavras no vocabulário: " + str(len(self.base_dados.vocabulario)))
        # limpo as variáveis utilizadas
        del (linha, lista, palavra)

        #self.base_dados.vocabulario = vocabulario
        #self.base_dados.chaves_vocabulario = chaves_vocabulario
        self.base_dados.encoder_input = encoder_input
        self.base_dados.decoder_input = decoder_input

    def preparacao_treinament(self):
        """
                Realizo o pad_sequences das duas listas de encoder
                Esta função transforma uma lista (de comprimento num_samples) de sequências (listas de inteiros) em uma matriz Numpy 2D de forma (num_samples, num_timesteps)
                o parâmetro padding='post' = faz com que todos tenho o mesmo número de componentes. isso é:
                sequence = [[1], [2, 3], [4, 5, 6]]
                array =(
                    [1,0,0],
                    [2,3,0],
                    [4,5,6]
                )
                """
        # global tamanho_perguntas_respostas_padrao
        self.base_dados.encoder_input = pad_sequences(self.base_dados.encoder_input, self.tamanho_perguntas, padding='post',
                                      truncating='post')
        self.base_dados.decoder_input = pad_sequences(self.base_dados.decoder_input, self.tamanho_perguntas, padding='post',
                                      truncating='post')
        # faço uma lista com cada uma das respostas, retirando a posicao 0

        for i in self.base_dados.decoder_input:
            self.base_dados.decoder_final_output.append(i[1:])
        # faço o processo de pad
        self.base_dados.decoder_final_output = pad_sequences(self.base_dados.decoder_final_output, self.tamanho_perguntas, padding='post',
                                             truncating='post')

        # limpo variavel de ambiente
        del (i)
        # converto o vetor de inteiros para uma binário8
        self.base_dados.decoder_final_output = to_categorical(self.base_dados.decoder_final_output, len(self.base_dados.vocabulario), dtype=np.int8)

        # bloco 2 - Preparando e Treinando o Modelo
        # inicio o keras com shape = 13 (tamanho da pergunta e resposta esperada)
        self.base_dados.encoder_input_shape = Input(shape=(self.tamanho_perguntas,))
        self.base_dados.decoder_input_shape = Input(shape=(self.tamanho_perguntas,))

        """
            A Embedding layer nos permite converter cada palavra em um vetor de comprimento fixo de tamanho definido. O vetor resultante é denso com valores reais em vez de apenas 0 e 1. O comprimento fixo dos vetores de palavras nos ajuda a representar as palavras de uma maneira melhor com dimensões reduzidas.
            """
        self.embed = Embedding(len(self.base_dados.vocabulario) + 1, output_dim=50,
                          input_length=self.tamanho_perguntas,
                          trainable=True
                          )

    def treinar_modelo(self):
        # chamo o método que prepara as variáveis para o treinamento
        print("preparando o treinamento")
        self.preparacao_treinament()
        print("preparação concluída")
        # configuro o encoder
        self.encoder_embed = self.embed(self.base_dados.encoder_input_shape)

        if (self.rede == "LSTM"):
            encoder_lstm = LSTM(int(self.celulas_LSTM), return_sequences=True, return_state=True)
            # output, return sequences, return state
            encoder_outputs, h, c = encoder_lstm(self.encoder_embed)
            encoder_states = [h, c]
            # faço o mesmo para o decoder
            decoder_embed = self.embed(self.base_dados.decoder_input_shape)
            decoder_lstm = LSTM(int(self.celulas_LSTM), return_state=True, return_sequences=True, dropout=0.05)
            decoder_output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
        elif (self.rede == "BLSTM"):
            encoder_blstm = Bidirectional(
                LSTM(int(self.celulas_LSTM), return_state=True, dropout=0.05, return_sequences=True))
            # output, return sequences, return state
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_blstm(self.encoder_embed)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            encoder_states = [state_h, state_c]

            decoder_embed = self.embed(self.base_dados.decoder_input_shape)
            decoder_lstm = LSTM(int(self.celulas_LSTM) * 2, return_state=True, return_sequences=True, dropout=0.05)
            decoder_output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
        elif (self.rede == "BLSTMMA"):
            encoder_blstm = Bidirectional(
                LSTM(int(self.celulas_LSTM), return_state=True, dropout=0.05, return_sequences=True))
            # output, return sequences, return state
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_blstm(self.encoder_embed)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            encoder_states = [state_h, state_c]

            decoder_embed = self.embed(self.base_dados.decoder_input_shape)
            decoder_lstm = LSTM(int(self.celulas_LSTM) * 2, return_state=True, return_sequences=True, dropout=0.05)
            output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)

            # attention
            attn_layer = AttentionLayer()
            attn_op, attn_state = attn_layer([encoder_outputs, output])
            decoder_output = Concatenate(axis=-1)([output, attn_op])

        tamanho_vocabulario = len(self.base_dados.vocabulario)
        decoder_dense = Dense(tamanho_vocabulario, activation='softmax')
        final_output = decoder_dense(decoder_output)

        self.modelo = Model([self.base_dados.encoder_input_shape, self.base_dados.decoder_input_shape], final_output)
        self.modelo.summary()
        self.modelo.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

        self.history = History()
        # treino o modelo com o encoder e decoder, numa estrutura de perguntas e respostas com o número de épocas informadas
        self.modelo.fit([self.base_dados.encoder_input, self.base_dados.decoder_input], self.base_dados.decoder_final_output,
                        epochs=int(self.epocas), batch_size=8,
                   validation_split=0.15, callbacks=[self.history])
        print("A rede " + self.rede + " foi treinada com sucesso!")

    def prepara_pasta(self):
        self.data_final_treinamento = datetime.now()
        data_em_texto = "{}_{}_{}_{}_{}_{}".format(self.data_final_treinamento.day, self.data_final_treinamento.month,
                                                   self.data_final_treinamento.year, self.data_final_treinamento.hour,
                                                   self.data_final_treinamento.minute, self.data_final_treinamento.second)

        try:
            os.makedirs("." + barra_sistema + "modelos_treinados" + barra_sistema + self.rede + "_" + data_em_texto)
            self.caminho= "." + barra_sistema + "modelos_treinados" + barra_sistema + self.rede + "_" + data_em_texto
        except OSError:
            # faz o que acha que deve se não for possível criar
            # salva no diretorio padrao
            print("Não foi possível criar a pasta")

    def salvar_modelo(self):
        #prepara dados para salvar
        self.prepara_pasta()

        #salva modelo
        self.modelo.save(self.caminho+barra_sistema+"model.h5")
        self.modelo.save_weights(self.caminho + barra_sistema + "model_w.h5")
        print("Pesos do Modelo Salvos com sucesso")

        #limpa informações inuteis
        del(self.base_dados.base,self.modelo)
        self.base_dados.limpar_dados_salvar()
        self.modelo = ""

    def salvar_dados(self):
        #salva informações do modelo
        arquivo = open(self.caminho + barra_sistema + "base_dados.obj", 'wb')
        pickle.dump(self.base_dados, arquivo)
        arquivo.close()
        arquivo = open(self.caminho + barra_sistema + "acc_loss.txt", "w")
        arquivo.write(str(self.history.history["acc"]) + '\n')
        arquivo.write(str(self.history.history["loss"]) + '\n')
        arquivo.close()

        #salva resumo em txt.
        arquivo = open(self.caminho + barra_sistema + "config_modelo.txt", "w")
        arquivo.write(str(self.rede) + '\n')
        arquivo.write(str(self.epocas) + '\n')
        arquivo.write(str(self.celulas_LSTM) + '\n')
        arquivo.write(str(self.tamanho_perguntas) + '\n')
        arquivo.write(str(self.data) + '\n')
        arquivo.write(str(self.data_final_treinamento)+ '\n')
        arquivo.write(str(self.tamanho_base))
        arquivo.close()
        print("Chatbot salvo com sucesso")


    def treinar(self):
        self.data = datetime.now()
        print("----------------------------------------------------------------------------")
        print("Configuração de Treino")
        print(
            "rede: " + self.rede + "--" + str(self.tamanho_perguntas) + "--" + str(self.celulas) + "--" + str(self.epocas))
        print("----------------------------------------------------------------------------")
        self.carregar_dados_treino()
        #limitar tamanho data_set ao limite da memória do pc
        self.base_dados.limitar_tamanho_base(self.tamanho_base)


        print("Recorte para treino e Teste")
        self.base_dados.recortar_base_treino_teste()

        print("wordembedding")
        self.word_embedding()

        print("Executando Treinamento")
        self.treinar_modelo()
        print("Chatbot Treinado com sucesso")

    def carregar_dados_chatbot(self,caminho):
        #arquivo = open(caminho + barra_sistema + "history.obj", 'rb')
        #self.history = pickle.load(arquivo)
        #arquivo.close()
        arquivo = open(caminho + barra_sistema + "base_dados.obj", 'rb')
        self.base_dados = pickle.load(arquivo)
        arquivo.close()

        arquivo = open(caminho + barra_sistema + "config_modelo.txt", 'rb')
        self.rede = arquivo.readline()[:-1]
        #print(len(self.rede))
        if len(self.rede)==5:
            self.rede="LSTM"
        elif len(self.rede)==6:
            self.rede="BLSTM"
        else:
            self.rede="BLSTMMA"
        self.epocas = int(arquivo.readline()[:-1])
        self.celulas_LSTM = int(arquivo.readline()[:-1])
        self.tamanho_perguntas = int(arquivo.readline()[:-1])
        self.data = arquivo.readline()[:-1]
        self.data_final_treinamento = arquivo.readline()[:-1]
        self.tamanho_base = int(arquivo.readline())
        self.caminho = caminho
        arquivo.close()

        arquivo = open(caminho + barra_sistema + "acc_loss.txt", 'rb')
        acc = arquivo.readline()[:-1]
        loss = arquivo.readline()[:-1]
        self.history = [acc,loss]
        arquivo.close()

    def carregar_modelo_salvo(self):
        self.modelo = load_model(self.caminho + barra_sistema + "model.h5")
        self.modelo.load_weights(self.caminho+barra_sistema+"model_w.h5")
        self.modelo.summary()

    def recuperar_estado_rede(self):
        encoder_inputs = self.modelo.layers[1].input
        decoder_inputs = self.modelo.layers[0].input
        self.embed = self.modelo.layers[2]
        enc_embed = self.embed(encoder_inputs)
        encoder_lstm = self.modelo.layers[3]
        dec_embed = self.embed(decoder_inputs)
        latent_dim = self.celulas_LSTM
        decoder_state_input_h=[]
        decoder_state_input_c=[]
        print(self.rede)
        if (self.rede == "LSTM"):
            decoder_state_input_h = Input(shape=(latent_dim,), name='input_5_')
            decoder_state_input_c = Input(shape=(latent_dim,), name='input_6_')
            encoder_outputs, h, c = encoder_lstm(enc_embed)
            encoder_states = [h, c]
            self.enc_model = Model([encoder_inputs], encoder_states)
            decoder_lstm = self.modelo.layers[4]
            self.attn_layer = None
        #elif (self.rede == "BLSTM"):
        else:
            decoder_state_input_h = Input(shape=(latent_dim * 2,), name='input_5_')
            decoder_state_input_c = Input(shape=(latent_dim * 2,), name='input_6_')
            encoder_outputs, fstate_h, fstate_c, bstate_h, bstate_c = encoder_lstm(enc_embed)
            h = Concatenate()([fstate_h, bstate_h])
            c = Concatenate()([fstate_c, bstate_c])
            encoder_states = [h, c]
            self.enc_model = Model(encoder_inputs, [encoder_outputs, encoder_states])
            decoder_lstm = self.modelo.layers[6]
            self.attn_layer = None
            if (self.rede == "BLSTMMA"):
                self.attn_layer = self.modelo.layers[7]

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(dec_embed, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        if (self.rede == "LSTM"):
            self.dec_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        else:
            self.dec_model = Model([decoder_inputs, decoder_states_inputs], [decoder_outputs] + decoder_states)

        self.dec_dense = self.modelo.layers[-1]

    def interacao(self):
        # Bloco 4 - Utilizando o chatbot
        print("------------------------------------------------------------")
        print("Bem Vindo ao Módulo de Interação da TIA - Technical Intelligence Assistant")
        print("Versão: 0.4")
        print("Rede utilizada: " + self.rede)
        print("------------------------------------------------------------")
        # basicamente faço a mesma sequencia da preparação do dataset
        pergunta = ""
        while True:
            # recebo a pergunta e pré processo ela
            pergunta = input("you : ")
            if (pergunta == "q"):
                break
            # faço a limpeza do texto
            pergunta = clean_text(pergunta)
            # pergunta = "hello"

            pergunta_formatada = [pergunta]
            # pergunta_formatada = ["hello"]

            txt = []
            for x in pergunta_formatada:
                # x = "hello"
                lst = []
                for y in x.split():
                    # y = "hello"
                    try:
                        lst.append(self.base_dados.vocabulario[y])
                        # vocabulario['hello'] = 454
                    except:
                        lst.append(self.base_dados.vocabulario['<OUT>'])
                txt.append(lst)

            # txt = [[454]]
            txt = pad_sequences(txt, self.tamanho_perguntas, padding='post')
            # txt = [[454,0,0,0,.........]
            if (self.rede == "LSTM"):
                stat = self.enc_model.predict(txt)
            else:
                enc_op, stat = self.enc_model.predict(txt)
            empty_target_seq = np.zeros((1, 1))
            #   empty_target_seq = [0]
            empty_target_seq[0, 0] = self.base_dados.vocabulario['<SOS>']
            #    empty_target_seq = [255]
            # empty_target_seq, stat = prepara_texto(prepro1)
            stop_condition = False
            decoded_translation = ''
            while not stop_condition:
                dec_outputs, h, c = self.dec_model.predict([empty_target_seq] + stat)
                if (self.rede == "LSTM" or self.rede == "BLSTM"):
                    decoder_concat_input = self.dec_dense(dec_outputs)
                # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]
                else:
                    # utiliza mecanismo de atenção
                    attn_op, attn_state = self.attn_layer([enc_op, dec_outputs])
                    decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
                    decoder_concat_input = self.dec_dense(decoder_concat_input)

                sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])

                sampled_word = self.base_dados.chaves_vocabulario[sampled_word_index] + ' '

                if sampled_word != '<EOS> ':
                    decoded_translation += sampled_word

                if sampled_word == '<EOS> ' or len(decoded_translation.split()) > self.tamanho_perguntas:
                    stop_condition = True

                empty_target_seq = np.zeros((1, 1))
                empty_target_seq[0, 0] = sampled_word_index
                stat = [h, c]
            print("T.I.A. : ", decoded_translation)
            print("==============================================")

    def interacao_lista(self,lista_perguntas):
        # Bloco 4 - Utilizando o chatbot
        #print("------------------------------------------------------------")
        #print("Bem Vindo ao Módulo de Interação da TIA - Technical Intelligence Assistant")
        #print("Versão: 0.4")
        #print("Rede utilizada: " + self.rede)
        #print("------------------------------------------------------------")
        # basicamente faço a mesma sequencia da preparação do dataset
        pergunta = ""
        resultado = []
        for i in tqdm(range(len(lista_perguntas[0]))):
            pergunta = lista_perguntas[0][i]
        #for pergunta in lista_perguntas[0]:
            # recebo a pergunta e pré processo ela
            #pergunta = input("you : ")
            if (pergunta == "q"):
                break
            # faço a limpeza do texto
            pergunta = clean_text(pergunta)
            # pergunta = "hello"

            pergunta_formatada = [pergunta]
            # pergunta_formatada = ["hello"]

            txt = []
            for x in pergunta_formatada:
                # x = "hello"
                lst = []
                for y in x.split():
                    # y = "hello"
                    try:
                        lst.append(self.base_dados.vocabulario[y])
                        # vocabulario['hello'] = 454
                    except:
                        lst.append(self.base_dados.vocabulario['<OUT>'])
                txt.append(lst)

            # txt = [[454]]
            txt = pad_sequences(txt, self.tamanho_perguntas, padding='post')
            # txt = [[454,0,0,0,.........]
            if (self.rede == "LSTM"):
                stat = self.enc_model.predict(txt)
            else:
                enc_op, stat = self.enc_model.predict(txt)
            empty_target_seq = np.zeros((1, 1))
            #   empty_target_seq = [0]
            empty_target_seq[0, 0] = self.base_dados.vocabulario['<SOS>']
            #    empty_target_seq = [255]
            # empty_target_seq, stat = prepara_texto(prepro1)
            stop_condition = False
            decoded_translation = ''
            while not stop_condition:
                dec_outputs, h, c = self.dec_model.predict([empty_target_seq] + stat)
                if (self.rede == "LSTM" or self.rede == "BLSTM"):
                    decoder_concat_input = self.dec_dense(dec_outputs)
                # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]
                else:
                    # utiliza mecanismo de atenção
                    attn_op, attn_state = self.attn_layer([enc_op, dec_outputs])
                    decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
                    decoder_concat_input = self.dec_dense(decoder_concat_input)

                sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])

                sampled_word = self.base_dados.chaves_vocabulario[sampled_word_index] + ' '

                if sampled_word != '<EOS> ':
                    decoded_translation += sampled_word

                if sampled_word == '<EOS> ' or len(decoded_translation.split()) > self.tamanho_perguntas:
                    stop_condition = True

                empty_target_seq = np.zeros((1, 1))
                empty_target_seq[0, 0] = sampled_word_index
                stat = [h, c]
            #print("T.I.A. : ", decoded_translation)
            resultado.append(decoded_translation)
            #print("==============================================")
        self.base_dados.lista_respostas = resultado

    def interar(self):
        self.recuperar_estado_rede()
        self.interacao()

    def interar_lista(self):
        self.recuperar_estado_rede()
        self.interacao_lista(self.base_dados.teste)

    def salvar_consultas(self):
        # salva informações do modelo
        arquivo = open(self.caminho + barra_sistema + "lista_respostas.obj", 'wb')
        pickle.dump(self.base_dados.lista_respostas, arquivo)
        arquivo.close()

    def calcula_media_ngram(self):
        quantidade = len(self.base_dados.scores_bleu[0])
        media_1 = sum(self.base_dados.scores_bleu[0]) / quantidade
        media_2 = sum(self.base_dados.scores_bleu[1]) / quantidade
        media_3 = sum(self.base_dados.scores_bleu[2]) / quantidade
        media_4 = sum(self.base_dados.scores_bleu[3]) / quantidade
        self.base_dados.scores_bleu = [media_1, media_2, media_3, media_4]

    def calcula_score_bleu_completa(self):
        self.base_dados.scores_bleu=[]
        for i in tqdm(range(len(self.base_dados.teste[0]))):
            referencia = self.base_dados.teste[1][i]
            candidata = self.base_dados.lista_respostas[i]
            referencia = [referencia.split()]
            #scores = []
            score1 = sentence_bleu(referencia, candidata.split(), weights=(1, 0, 0, 0))
            score2 = sentence_bleu(referencia, candidata.split(), weights=(0.5, 0.5, 0, 0))
            score3 = sentence_bleu(referencia, candidata.split(), weights=(0.33, 0.33, 0.33, 0))
            score4 = sentence_bleu(referencia, candidata.split(), weights=(0.25, 0.25, 0.25, 0.25))
            self.base_dados.scores_bleu.append([round(score1, 7), round(score2, 7), round(score3, 7), round(score4, 7)])

        self.calcula_media_ngram()
        print(self.base_dados.scores_bleu)

    def calcular_score_PP(self):
        return ""

    def carregar_consultas(self):
        arquivo = open(self.caminho + barra_sistema + "lista_respostas.obj", 'rb')
        self.base_dados.lista_respostas = pickle.load(arquivo)
        arquivo.close()

    def verificar_lista_consultas(self):
        lista_arquivos_pastas = []
        #pasta = '.' + barra_sistema + "modelos_treinados"
        for diretorio, subpastas, arquivos in os.walk(self.caminho):
            # print(diretorio)
            for arquivo in arquivos:
                if (arquivo == "lista_respostas.obj"):
                    return True
        return False

    def calcular_Bleu(self):
        if(self.verificar_lista_consultas()):
            #carrega consultas
            self.carregar_consultas()
        else:
            resultado = self.interar_lista()
            self.salvar_consultas()
        self.calcula_score_bleu_completa()

    def calcular_PP(self):
        if (self.verificar_lista_consultas()):
            # carrega consultas
            self.carregar_consultas()
        else:
            resultado = self.interar_lista()
            self.salvar_consultas()
            print("teste")
        self.calcular_score_PP()



    @staticmethod
    def lista_chatbots_salvos():
        lista_arquivos_pastas = []
        pasta = '.' + barra_sistema + "modelos_treinados"
        for diretorio, subpastas, arquivos in os.walk(pasta):
            # print(diretorio)
            for arquivo in arquivos:
                if (arquivo == "config_modelo.txt"):
                    lista_arquivos_pastas.append(os.path.join(diretorio, arquivo))

        print("Modelos treinados encontrados :" + str(len(lista_arquivos_pastas)))
        i = 1
        t = PrettyTable(['ID', 'Rede', 'Epocas', 'Celulas', 'Caracteres', 'Ini', 'Fim'])
        lista = []
        for arquivo in lista_arquivos_pastas:
            with open(arquivo, 'r') as modelo_treinado:
                i += 1

                rede = modelo_treinado.readline()[:-1]
                epocas = modelo_treinado.readline()[:-1]
                #if (int(epocas) < 5):
                #    continue

                celulas = modelo_treinado.readline()[:-1]
                tamanho_perguntas_respostas_padrao = modelo_treinado.readline()[:-1]
                data_inicio = modelo_treinado.readline()[:-1]
                data_fim = modelo_treinado.readline()

                t.add_row([i - 1, rede, epocas, celulas, tamanho_perguntas_respostas_padrao, data_inicio, data_fim])

        # print(lista)
        print(t)
        print("Caso deseje encerrar digite 0")
        num_modelo = input("informe o número referente ao modelo encontrado: ")
        print(num_modelo)
        if (num_modelo == "0"):
            exit()
        return lista_arquivos_pastas[int(num_modelo) - 1][:-17]

    @staticmethod
    def carregar_chatbots_treinados():
        caminho = Chatbot.lista_chatbots_salvos()
        print(caminho)
        chatbot = Chatbot()
        chatbot.carregar_dados_chatbot(caminho)
        chatbot.carregar_modelo_salvo()
        return chatbot











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
        #treinar somente uma rede
        chatbot.treinar()
        #salva o chatbot
        chatbot.salvar()
        print("Rodada de Treino Concluída com Sucesso")
        print("----------------------------------------------------------------------------")
        del (chatbot)

if __name__ == '__main__':
    print("----------------------------------------------------------------------------")
    print("Teste")
    print("----------------------------------------------------------------------------")
    #lista_treinos = configuracao_treino()
    #treinar_lista_modelos(lista_treinos)

    chatbot = Chatbot.carregar_chatbots_treinados()


