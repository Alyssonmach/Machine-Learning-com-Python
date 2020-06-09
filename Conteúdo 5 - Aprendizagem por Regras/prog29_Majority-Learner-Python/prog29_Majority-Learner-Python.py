# Aplicação do método Majority Learner usando Aprendizagem de Regras

# importando a biblioteca orange do python
import Orange

# criando o objeto "dataframe" e passando todos os registros da tabela de 
# dados
dataframe = Orange.data.Table('Dados de Credito.csv')

# visualizando os atributos previsores e classe
print(dataframe.domain)

# lemandro que a tabela de dados recebeu "i#" no id para ignorar esse campo
# e "c#" em default para identificar a classe

# separando a tabela de dados em uma tabela para treinamento e outra para 
# teste
base_dividida = Orange.evaluation.testing.sample(dataframe, n = 0.25)

# separando em diferentes variáveis a base de dados para teste e outra 
# treinamento
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# visualizando a quantidade de registros para cada uma das bases criadas
print(len(base_treinamento))
print(len(base_teste))

# criando um objeto "classificador" para ser utilizando no treinamento com
# os dados
classificador = Orange.classification.MajorityLearner()

# fazendo o treinamento com os dados e a análise da capacidade de predição
# do algoritmo
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, 
                                                     [classificador])

# exibindo na saída de dados a capacidade de predição do algoritmo por aprendizagem
# indutiva de regras
print(Orange.evaluation.CA(resultado))

# capacidade de predição: 86.4%

# importando o Majority Learn (zeroR) para fazer a contagem de acertos do
# algoritmo usado em aprendizagem por regras
from collections import Counter

# exibindo na saída de dados os elementos de acerto em relação a todos 
# elementos da tabela de teste 
print(Counter(str(d.get_class()) for d in base_teste))
# aproximandamente 437 acertos de 500 registros





