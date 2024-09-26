import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Grafico de dispersão com regressão linear
# Gráfico de resíduos (simples)
# Gráfico de resíduos (múltiplo)
# Carregar os dados (ajuste o caminho conforme necessário)
distancia = pd.read_csv("distancia.csv")  # DataFrame com as colunas 'distancia' e 'idade'
esteira = pd.read_excel("esteira.xlsx")  # DataFrame com as colunas 'VO2', 'IMC' e 'carga'

# Correlação das variáveis distância e idade
corr_simples = distancia['distancia'].corr(distancia['idade'])
print("Correlação simples entre distância e idade:", corr_simples)

# Correlação das variáveis VO2, IMC e carga
multiplo = esteira[['VO2', 'IMC', 'carga']]
correlacoes_multiplo = multiplo.corr()
print("Matriz de correlação múltipla:\n", correlacoes_multiplo)

# Regressão linear simples: distância em função da idade
X_simples = distancia[['idade']]  # Variável independente
y_simples = distancia['distancia']  # Variável dependente

modelo_simples = LinearRegression()
modelo_simples.fit(X_simples, y_simples)

# Coeficientes do modelo simples
print("Coeficiente do modelo simples:", modelo_simples.coef_)
print("Intercepto do modelo simples:", modelo_simples.intercept_)

# Gráfico de dispersão e linha de regressão
plt.scatter(distancia['idade'], distancia['distancia'], label="Dados")
plt.plot(distancia['idade'], modelo_simples.predict(X_simples), color="blue", label="Regressão Linear")
plt.xlabel("Idade (em anos)")
plt.ylabel("Distância (m)")
plt.title("Gráfico de Dispersão")
plt.legend()
plt.show()

# Regressão linear múltipla: VO2 em função de IMC e carga
X_multiplo = esteira[['IMC', 'carga']]
y_multiplo = esteira['VO2']

modelo_multiplo = sm.OLS(y_multiplo, sm.add_constant(X_multiplo)).fit()

# Resumo dos modelos
print("Resumo do modelo simples:\n", sm.OLS(y_simples, sm.add_constant(X_simples)).fit().summary())
print("Resumo do modelo múltiplo:\n", modelo_multiplo.summary())

# Gráfico de resíduos do modelo simples
residuos_simples = y_simples - modelo_simples.predict(X_simples)
plt.plot(residuos_simples, label="Resíduos Simples")
plt.title("Gráfico de Resíduos (Modelo Simples)")
plt.show()

# Gráfico de resíduos do modelo múltiplo
residuos_multiplos = modelo_multiplo.resid
plt.plot(residuos_multiplos, label="Resíduos Múltiplos")
plt.title("Gráfico de Resíduos (Modelo Múltiplo)")
plt.show()

import pandas as pd
import statsmodels.api as sm

# Supondo que o DataFrame 'inibina_1' já tenha sido carregado (substitua pelo caminho correto, se necessário)
inibina_1 = pd.read_csv('inibina_1.csv')

# Definir as variáveis dependente (resposta) e independente (difinib)
X = inibina_1[['difinib']]  # Variável independente
y = inibina_1['resposta']    # Variável dependente (resposta binária)

# Adicionar uma constante (intercepto) ao modelo
X = sm.add_constant(X)

# Ajustar o modelo de regressão logística
modelo_simples = sm.Logit(y, X).fit()

# Resumo estatístico do modelo
resumo = modelo_simples.summary()
print(resumo)
