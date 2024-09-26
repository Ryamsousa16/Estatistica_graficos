import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prince

# boxplot com multipas variáveis (salario x Inst e Est.Civil)
# Matriz de correlação
# Gráfico biplot

# Carregar os dados (ajuste o caminho conforme necessário)
milsa = pd.read_csv("milsa.csv")

# Tabela para as duas variáveis qualitativas e uma quantitativa
tabela = pd.crosstab([milsa['Regiao'], milsa['Est.civil']], milsa['Inst'])
tabela_freq_relativa = tabela / tabela.sum().sum()
print(tabela_freq_relativa)

# Medidas descritivas para múltiplas variáveis
# Função equivalente ao aggregate em R
resumo_salario = milsa.groupby(['Est.civil', 'Inst'])['Salario'].describe()
print(resumo_salario)

# Boxplot para comparar as estatísticas da variável Salario nas categorias de Inst e Est.civil
plt.figure(figsize=(10, 6))
sns.boxplot(x="Inst", y="Salario", hue="Est.civil", data=milsa)
plt.title("Salários por Nível de Instrução e Estado Civil")
plt.xlabel("Nível de Instrução")
plt.ylabel("Salário")
plt.show()

# Substituir valores NA em "Filhos" pela média
valor_substituto = milsa['Filhos'].mean(skipna=True)
milsa['Filhos'] = milsa['Filhos'].fillna(valor_substituto)
print("Valores NA em 'Filhos' substituídos pela média:", valor_substituto)

# Visualizar a matriz de correlação entre salário, Anos, e Filhos
matriz_correlacao = milsa[['Salario', 'Anos', 'Filhos']].corr()
print(matriz_correlacao)

# Gráfico da matriz de correlação usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', square=True)
plt.title("Matriz de Correlação")
plt.show()

# Análise de Correspondência Múltipla (MCA) com Prince
mca_variaveis = milsa[['Regiao', 'Est.civil', 'Inst']].astype('category')
mca = prince.MCA(n_components=2)
resultado_mca = mca.fit(mca_variaveis)

# Coordenadas das linhas e colunas
row_coords = resultado_mca.row_coordinates(mca_variaveis)
col_coords = resultado_mca.column_coordinates(mca_variaveis)

# Plotando o biplot do MCA
plt.figure(figsize=(10, 8))
plt.scatter(row_coords[0], row_coords[1], label="Observações", color='blue', alpha=0.5)
plt.scatter(col_coords[0], col_coords[1], label="Variáveis", color='red', alpha=0.7)
plt.title("MCA Biplot: Região, Estado Civil, Nível de Instrução")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.legend()
plt.show()
