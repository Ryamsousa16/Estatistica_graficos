import pandas as pd
import statsmodels.api as sm

# substitua pelo caminho correto, se necessário
inibina_1 = pd.read_excel('inibina_1.xlsx')

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
