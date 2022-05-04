from sys import platform
import preparar_base
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys,os
if platform == "linux":
    barra_sistema = "//"
else:
    barra_sistema = "\\"

class Data_set:
    def __init__(self):
        self.geral = []
        self.especialista = []
        self.vocabulario = {}
        self.invocabulario = []
        self.base = []
        self.chaves_vocabulario = []
        self.lista_respostas = []
        self.scores_bleu = []
        self.usar_base_geral = True
        self.usar_base_especialista = False
        self.encoder_input = []
        self.decoder_input = []
        self.encoder_input_shape = []
        self.decoder_input_shape = []
        self.decoder_final_output = []

        self.treino = []
        self.teste = []

    def limpar_dados_salvar(self):
        del(self.encoder_input,self.decoder_input,self.encoder_input_shape,self.decoder_input_shape,self.decoder_final_output)


    def carregar_data_sets(self):
        print("Carregando Base de dados")
        self.geral,self.especialista = preparar_base.prepara_bases()
        print("Dados carregadores com sucesso")

    #fazer filtro por numero de caracteres
    def limitar_caracteres(self,numero_caracteres):
        print("Filtrando base")
        tamanho_base = len(self.geral[0])
        perguntas = []
        respostas = []
        for i in tqdm(range(tamanho_base)):
            if ((len(self.geral[0][i]) <= numero_caracteres) and (len(self.geral[1][i]) <= numero_caracteres)):
                perguntas.append(self.geral[0][i])
                respostas.append(self.geral[1][i])


        del(self.geral)
        self.geral = [perguntas,respostas]
        del(perguntas,respostas)
        perguntas = []
        respostas = []
        tamanho_base = len(self.especialista[0])
        for i in tqdm(range(tamanho_base)):
            if ((len(self.especialista[0][i]) <= numero_caracteres) and (len(self.especialista[1][i]) <= numero_caracteres)):
                perguntas.append(self.especialista[0][i])
                respostas.append(self.especialista[1][i])

        del(self.especialista)
        self.especialista = [perguntas,respostas]
        del(perguntas,respostas)

        if(self.usar_base_geral):
            self.base = [self.geral[0]+self.especialista[0],self.geral[1]+self.especialista[1]]
        else:
            self.base = [self.especialista[0], self.especialista[1]]

        if (self.usar_base_especialista):
            self.base = [self.geral[0] + self.especialista[0], self.geral[1] + self.especialista[1]]
        else:
            self.base = [self.geral[0], self.geral[1]]

        print("Base filtrada com sucesso")

    def limitar_tamanho_base(self,numero_linhas):
        if len(self.geral) > numero_linhas:
            self.geral = [self.geral[:][:numero_linhas_treinamento],
                              self.geral[1][:numero_linhas_treinamento]]

            self.base = [self.geral[0]+self.especialista[0],self.geral[1]+self.especialista[1]]

    def recortar_base_treino_teste(self):
        X = self.base[0]
        y = self.base[1]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=42)
        self.treino = [X_train,y_train]
        self.teste = [X_test,y_test]


