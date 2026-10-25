# Aplicação do Majority Learner à aprendizagem por regras

# Importa a biblioteca Orange.
import Orange

# Cria o DataFrame e carrega os registros da tabela.
dataframe = Orange.data.Table('Dados de Credito.csv')

# Exibe os atributos preditores e a classe.
print(dataframe.domain)

# Lembre-se: o identificador "i#" marca o ID para ser ignorado,
# e "c#" indica a classe.

# Divide os dados em conjuntos de treinamento e teste.
base_dividida = Orange.evaluation.testing.sample(dataframe, n = 0.25)

# Separa os conjuntos de treinamento e de teste em variáveis distintas.
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# Exibe a quantidade de registros em cada conjunto.
print(len(base_treinamento))
print(len(base_teste))

# Cria o classificador que será usado no treinamento com os dados.
classificador = Orange.classification.MajorityLearner()

# Treina o modelo e avalia sua capacidade preditiva.
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, 
                                                     [classificador])

# Exibe a capacidade preditiva do algoritmo de aprendizagem
# indutiva de regras.
print(Orange.evaluation.CA(resultado))

# Acurácia: 86,4%.

# Importa o Majority Learner (ZeroR) para comparar os acertos do
# algoritmo de aprendizagem por regras.
from collections import Counter

# Exibe a quantidade de acertos em relação ao total de
# registros da base de teste.
print(Counter(str(d.get_class()) for d in base_teste))
# Aproximadamente 437 acertos em 500 registros.





