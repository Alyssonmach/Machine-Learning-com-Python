# Aprendizagem por Regras - Base de Dados de Crédito

# importando a biblioteca orange do python
import Orange
# com a biblioteca orange, não é necessário fazer toda a parte de 
# pré-processamento, pois ele já consegue lidar melhor com os dados 
# em relação ao scikit-learn

# criando o objeto "dataframe" para receber todos os registros da tabela de
# dados
dataframe = Orange.data.Table('Dados de Credito.csv')

# visualizando os atributos previsores e classe na saída de dados
print(dataframe.domain)

# colocar um identificador "c#nameclass" para identificar a classe
# colocar um identificador "i#id" para ignorar o id do cliente, esse
# que só prejudicaria o algoritmo

# separando a tabela de dados em base de treinamento e base de teste
base_dividida = Orange.evaluation.testing.sample(dataframe, n = 0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# visualizando a quantidade de registros em cada uma das bases
print(len(base_treinamento))
print(len(base_teste))

# criando o objeto "cn2_learner" para realização do treinamento com a base
# de dados
cn2_learner = Orange.classification.rules.CN2Learner()

#treinando o algoritmo com o objeto "cn2_learner" criado
classificador = cn2_learner(base_treinamento)

# observando as regras geradas para avaliar o algoritmo
for regras in classificador.rule_list:
    print(regras)

# fazendo o treinamento com os dados e a análise da capacidade de predição
# do algoritmo
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, 
                                                     [lambda testdata: classificador])
print(Orange.evaluation.CA(resultado))

# resultado: 98% de acertos
# o algoritmo está muito bom para essa base de dados
