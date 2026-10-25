# Aprendizagem por regras — base de crédito

# Importa a biblioteca Orange.
import Orange
# Com Orange, não é necessário fazer todo o pré-processamento,
# pois a biblioteca oferece recursos para lidar com esses dados.

# Cria o DataFrame para armazenar todos os registros da tabela.
dataframe = Orange.data.Table('Dados de Credito.csv')

# Exibe os atributos preditores e a classe.
print(dataframe.domain)

# Use o identificador "c#nameclass" para indicar a classe.
# Use o identificador "i#id" para ignorar o ID do cliente,
# que não deve ser usado pelo algoritmo.

# Divide os dados em conjuntos de treinamento e teste.
base_dividida = Orange.evaluation.testing.sample(dataframe, n = 0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# Exibe a quantidade de registros em cada conjunto.
print(len(base_treinamento))
print(len(base_teste))

# Cria o objeto "cn2_learner" para treinar o modelo com a base
# de dados.
cn2_learner = Orange.classification.rules.CN2Learner()

# Treina o algoritmo com o objeto "cn2_learner".
classificador = cn2_learner(base_treinamento)

# Exibe as regras geradas para avaliar o algoritmo.
for regras in classificador.rule_list:
    print(regras)

# Treina o modelo e avalia sua capacidade preditiva.
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, 
                                                     [lambda testdata: classificador])
print(Orange.evaluation.CA(resultado))

# Resultado: 98% de acurácia.
# O algoritmo apresentou bom desempenho nesta base de dados.
