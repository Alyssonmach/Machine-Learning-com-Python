# Aprendizagem por Regras - Usando a base de dados de Risco de Crédito

# importando a biblioteca Orange do Python
import Orange

# criando o objeto "dataframe" para receber os registros da tabela de dados
dataframe  = Orange.data.Table('Risco_credito.csv')

# observando na saída de dados os atributos previsores e classe
print(dataframe.domain)

# criando o objeto "cn2_learner", esse que é um algoritmo de indução por
# regras
cn2_learner = Orange.classification.rules.CN2Learner()

# treinando com a base de dados no objeto "classificador"
# A base de dados deve possuir, no nome da classe, o seguinte identificador:
# "c#nameclass". Caso contrário, o orange não poderá indentificar a classe
classificador = cn2_learner(dataframe)

# Observando as regras criadas nesse algoritmo de indução
for regras in classificador.rule_list:
    print(regras)
    
# entrada de dados usando o algoritmo treinado
# 1º entrada: história boa, dívida alta, garantias nenhuma, renda > 35
# 2º entrada: história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'],
                           ['ruim', 'alta', 'adequada', '0_15']])

# visualizando a saída de dados para "resultado"
for i in resultado:
    print(dataframe.domain.class_var.values[i])


