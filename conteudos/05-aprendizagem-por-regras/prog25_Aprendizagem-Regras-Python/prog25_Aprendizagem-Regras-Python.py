# Aprendizagem por regras — base de risco de crédito

# Importa a biblioteca Orange.
import Orange

# Cria o DataFrame para armazenar os registros da tabela.
dataframe  = Orange.data.Table('Risco_credito.csv')

# Exibe os atributos preditores e a classe.
print(dataframe.domain)

# Cria o objeto "cn2_learner", um algoritmo de indução por
# regras.
cn2_learner = Orange.classification.rules.CN2Learner()

# Treina o classificador com a base de dados.
# A base de dados deve identificar a classe da seguinte forma:
# "c#nameclass". Caso contrário, o Orange não identificará a classe.
classificador = cn2_learner(dataframe)

# Exibe as regras geradas pelo algoritmo de indução.
for regras in classificador.rule_list:
    print(regras)
    
# Insere dados para classificação usando o algoritmo treinado.
# 1ª entrada: histórico bom, dívida alta, sem garantias e renda > 35.
# 2ª entrada: histórico ruim, dívida alta, garantia adequada e renda < 15.
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'],
                           ['ruim', 'alta', 'adequada', '0_15']])

# Exibe o resultado.
for i in resultado:
    print(dataframe.domain.class_var.values[i])


