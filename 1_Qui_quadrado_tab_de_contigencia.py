import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols

# Carregar os dados (ajuste o caminho conforme necessário)
# Gráfico de barras
# Correlação
# Gráfico de dispersão
# boxplot com 3 dados


milsa = pd.read_csv("milsa.csv")

# Construir a tabela de dupla entrada (tabela de contingência) com frequência absoluta
tabela_contingencia = pd.crosstab(milsa['Est.civil'], milsa['Inst'])
print(tabela_contingencia)

# Adicionar as somas das colunas e linhas
tabela_contingencia_margins = tabela_contingencia.copy()
tabela_contingencia_margins.loc['Total'] = tabela_contingencia.sum()
tabela_contingencia_margins['Total'] = tabela_contingencia.sum(axis=1)
print(tabela_contingencia_margins)

# Construir a tabela de contingência com frequência relativa
tabela_contingencia_relativa = tabela_contingencia / tabela_contingencia.sum().sum()
print(tabela_contingencia_relativa)

# Frequência relativa com margens
tabela_contingencia_relativa_linha = tabela_contingencia.div(tabela_contingencia.sum(axis=1), axis=0)
print(tabela_contingencia_relativa_linha)

tabela_contingencia_relativa_coluna = tabela_contingencia.div(tabela_contingencia.sum(axis=0), axis=1)
print(tabela_contingencia_relativa_coluna)

# Construção dos gráficos de barras
plt.figure(figsize=(10, 6))

# Gráfico de barras para Estado Civil x Nível de Instrução
tabela_contingencia.plot(kind='bar', figsize=(10, 6), stacked=False, legend=True)
plt.title("Estado Civil por Nível de Instrução")
plt.xlabel("Estado Civil")
plt.ylabel("Frequência")
plt.legend(title="Nível de Instrução")
plt.show()

# Gráfico de barras transposto para Nível de Instrução x Estado Civil
tabela_contingencia.T.plot(kind='bar', figsize=(10, 6), stacked=False, legend=True)
plt.title("Nível de Instrução por Estado Civil")
plt.xlabel("Nível de Instrução")
plt.ylabel("Frequência")
plt.legend(title="Estado Civil")
plt.show()

# Cálculo do teste de Qui-quadrado para a tabela de contingência, retornando quatro valores
chi2, p, dof, expected = chi2_contingency(tabela_contingencia)

# Valor do Qui-quadrado
X2 = chi2
# Número total de observações
n = tabela_contingencia.to_numpy().sum()
# Fórmula do coeficiente de contingência
C = np.sqrt(X2 / (X2 + n))
print("Coeficiente de Contingência:", C)
# Calcular o coeficiente de contingência diretamente (sem uma função específica, mas o cálculo acima já faz isso)


# Supondo que o arquivo "milsa" seja um DataFrame
# Carregar os dados (ajuste o caminho conforme necessário)
milsa = pd.read_csv("milsa.csv")

# Quartis da variável Salario
quartis_salario = milsa['Salario'].quantile([0, 0.25, 0.5, 0.75, 1.0])
print(quartis_salario)

# Classes para variável Salario conforme os quantis
salario_cut = pd.cut(milsa['Salario'], bins=quartis_salario, include_lowest=True)
print(salario_cut)

# Tabela das frequências absolutas das variáveis níveis de Instrução e Salario
tabela_salario = pd.crosstab(milsa['Inst'], salario_cut)
print(tabela_salario)

# Frequência relativa
tabela_salario_relativa = tabela_salario / tabela_salario.sum().sum()
print(tabela_salario_relativa)

# Gráfico Boxplot para Salario nas categorias de Inst com 3 valores
milsa.boxplot(column='Salario', by='Inst')
plt.title("Boxplot de Salário por Nível de Instrução")
plt.suptitle("")  # Remove o título adicional
plt.show()

# Medidas de resumo (média, mediana, desvio padrão, mínimo e máximo) para cada categoria de Inst
media_por_inst = milsa.groupby('Inst')['Salario'].mean()
mediana_por_inst = milsa.groupby('Inst')['Salario'].median()
dp_por_inst = milsa.groupby('Inst')['Salario'].std()
min_por_inst = milsa.groupby('Inst')['Salario'].min()
quantis_por_inst = milsa.groupby('Inst')['Salario'].quantile([0.25, 0.5, 0.75])
max_por_inst = milsa.groupby('Inst')['Salario'].max()

print("Média:", media_por_inst)
print("Mediana:", mediana_por_inst)
print("Desvio Padrão:", dp_por_inst)
print("Mínimo:", min_por_inst)
print("Quantis:", quantis_por_inst)
print("Máximo:", max_por_inst)

# Modelo ANOVA para o cálculo do coeficiente de determinação (R²)
modelo_anova = ols('Salario ~ Inst', data=milsa).fit()
anova_sum = sm.stats.anova_lm(modelo_anova, typ=2)

# Cálculo do R² (coeficiente de determinação)
ss_total = anova_sum['sum_sq'].sum()
ss_modelo = anova_sum.loc['Inst', 'sum_sq']
r_squared = ss_modelo / ss_total
print("R² (coeficiente de determinação):", r_squared)

# Estudo de duas variáveis quantitativas
# Classes para a variável Anos conforme os quartis
quartis_anos = milsa['Anos'].quantile([0, 0.25, 0.5, 0.75, 1.0])
anos_cut = pd.cut(milsa['Anos'], bins=quartis_anos, include_lowest=True)

# Classes para a variável salário conforme os quartis
salario_cut = pd.cut(milsa['Salario'], bins=quartis_salario, include_lowest=True)

# Tabela de contingência entre Salario e Anos
tabela_sal_anos = pd.crosstab(salario_cut, anos_cut)
print(tabela_sal_anos)

# Frequência relativa para a tabela de Salario e Anos
tabela_sal_anos_relativa = tabela_sal_anos / tabela_sal_anos.sum().sum()
print(tabela_sal_anos_relativa)

# Gráfico de dispersão entre Anos e Salario
plt.scatter(milsa['Anos'], milsa['Salario'])
plt.title("Gráfico de Dispersão: Anos vs Salário")
plt.xlabel("Anos")
plt.ylabel("Salário")
plt.show()

# Coeficiente de correlação entre Anos e Salario (Pearson, Spearman, Kendall)
correlacao_pearson = milsa[['Anos', 'Salario']].corr(method='pearson')
correlacao_spearman = milsa[['Anos', 'Salario']].corr(method='spearman')
correlacao_kendall = milsa[['Anos', 'Salario']].corr(method='kendall')

print("Correlação Pearson:\n", correlacao_pearson)
print("Correlação Spearman:\n", correlacao_spearman)
print("Correlação Kendall:\n", correlacao_kendall)
