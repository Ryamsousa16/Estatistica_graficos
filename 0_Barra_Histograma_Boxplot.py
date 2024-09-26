import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Função para ler o arquivo.csv
ceagfgv = pd.read_csv("ceagfgv.csv")

x = int(input("Digite um número \n"
              "1 para ver o gráfico de barras\n"
              "2 para ver o gráfico de histograma\n"
              "3 para ver o gráfico de boxplot \n"
              "4 Para ver as Medidas de posição: "))

if x == 1:
    # Construir e apresentar as tabelas das variáveis qualitativas (strings)

    tabela_ingles = ceagfgv['ingles'].value_counts()
    tabela_ingles_porcentagem = ceagfgv['ingles'].value_counts(normalize=True)
    tabela_ingles_acumulada = tabela_ingles_porcentagem.cumsum()
    Tabela_completa = pd.DataFrame({
        "Contagens": tabela_ingles,
        "%": tabela_ingles_porcentagem,
        "% Acumulada": tabela_ingles_acumulada
    })

    # Construir tabela de contingência para duas variáveis categóricas

    tabela_contingencia = pd.crosstab(ceagfgv['estcivil'], ceagfgv['bebida'])
    print(tabela_contingencia)

    # Construção do gráfico de barras das variáveis categóricas
    # Gráfico de barras em pé

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(tabela_ingles.index, tabela_ingles.values)  # primeiro valor são os rótulos e o segundo são os valores
    plt.title("Fluência")
    plt.ylabel("Frequência")

    # Gráfico de barras deitado
    plt.subplot(1, 2, 2)  # se quiser deitado, só usar esse subplot com esses valores
    plt.barh(tabela_ingles.index, tabela_ingles.values)
    plt.title("Fluência")
    plt.xlabel("Frequência")

    plt.show()

elif x == 2:
    # Construir o histograma das variáveis quantitativas

    # Tabela para histograma
    plt.hist(ceagfgv['salario'], bins=10, edgecolor='black')
    plt.title("Salários")
    plt.xlabel("Reais")
    plt.ylabel("Frequência")
    plt.show()

    # 2º exemplo de gráfico com histograma
    plt.bar(ceagfgv['anosformado'].value_counts().index, ceagfgv['anosformado'].value_counts().values)
    plt.title("Anos formado")
    plt.ylabel("Frequência")
    plt.show()

elif x == 3:
    # Obter as medidas de posição das variáveis quantitativas
    # Gráfico de boxplot com apenas uma "barra"
    Salarios = ceagfgv['salario'].describe()
    Anos_formado = ceagfgv['anosformado'].describe()
    Numero_de_filhos = ceagfgv['filhos'].describe()
    Tabela_quantitativas = pd.DataFrame({
        "Salarios": Salarios,
        "Anos_formado": Anos_formado,
        "Numero_de_filhos": Numero_de_filhos
    })
    print(Tabela_quantitativas)

    # Segundo exemplo de gráfico de boxplot com mais de uma "barra"
    # Obter o boxplot das variáveis quantitativas
    plt.boxplot(ceagfgv['salario'])
    plt.title("Salários")
    plt.ylabel("Reais")
    plt.show()

    # Quando precisar colocar mais uma "barra" no boxplot ou mais uma coluna, insira os valores aqui abaixo
    plt.boxplot([ceagfgv['anosformado'], ceagfgv['filhos']], labels=['Anos Formado', 'Número de Filhos'])
    plt.show()

elif x == 4:
    # Medidas de posição
    salario_summary = ceagfgv['salario'].describe()
    anosformado_summary = ceagfgv['anosformado'].describe()
    print(salario_summary)
    print(anosformado_summary)

    mediana_salario = ceagfgv['salario'].median()
    print("Mediana do Salário:", mediana_salario)

    # Medidas de dispersão (variabilidade)
    variancia_salario = ceagfgv['salario'].var()
    desvio_padrao_salario = ceagfgv['salario'].std()
    Amplitude = 3425 - 1800  # 1800 é a mediana, 3425 é o terceiro quartil do boxplot
    print("Amplitude do Salário:", Amplitude)

    variancia_anosformado = ceagfgv['anosformado'].var()
    desvio_padrao_anosformado = ceagfgv['anosformado'].std()
    print("Variância do Salário:", variancia_salario)
    print("Desvio Padrão do Salário:", desvio_padrao_salario)
    print("Variância dos Anos Formado:", variancia_anosformado)
    print("Desvio Padrão dos Anos Formado:", desvio_padrao_anosformado)

    # Medidas de forma (assimetria e curtose)
    coeficiente_assimetria = skew(ceagfgv['salario'])
    coeficiente_curtose = kurtosis(ceagfgv['salario'])
    print("Coeficiente de Assimetria:", coeficiente_assimetria)
    print("Coeficiente de Curtose:", coeficiente_curtose)

    # Histograma
    plt.hist(ceagfgv['salario'], bins=10, edgecolor='black')
    plt.title("Histograma de Salários")
    plt.xlabel("Salários")
    plt.ylabel("Frequência")
    plt.show()
