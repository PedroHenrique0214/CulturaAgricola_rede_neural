# Cultura Agricola usando rede neural
Predição de 'crops' (culturas agrícolas) para encontrar a melhor opção para determinado solo.

### 01. Introdução

Vamos subor que tenhamos uma série de informações coletadas sobre um solo, e queremos descobrir qual seria a plantação adequada para se plantar nesse solo com base em informações de outros produtores. Podemos então utilizar um modelo de machine learning para nos dizer qual seria a melhor opção de plantação. 

Este é um problema de classificação multiclasse, com mais de duas saídas possíveis. Vamos criar uma rede neural para tentar predizer qual é a melhor opção de plantação com base em nosso data set coletado na Kagle. A construção da rede neural se dará através do TensorFlow, e poderemos observar como utilizar essa poderosa ferramenta de rede profunda de maneira rápida para solucionar um problema que pode ocorrer na decisão do pequeno agropecuário ao se decidir o que plantar no solo.

dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/8a6ff858-d004-4b16-b32b-5376d88544d0)


### 02. Conhecendo o dataset

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nitrogen</th>
      <th>Phosphorus</th>
      <th>Potassium</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>pH_Value</th>
      <th>Rainfall</th>
      <th>Crop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
      <td>Rice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
      <td>Rice</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
      <td>Rice</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
      <td>Rice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
      <td>Rice</td>
    </tr>
  </tbody>
</table>
</div>

**Temos algumas informações sobre o solo**
* Nitrogênio;
* Fósforo;
* Potássio;
* Temperatura;
* Humidade;
* pH;
* Pluviosidade.

  ### 03. Visualizando nossos dados
  
![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/f0cd4edf-2d34-4a3e-bbac-179cf3786e0e)

A análise dos boxplots revela a presença de alguns outliers em nosso conjunto de dados. No entanto, é crucial considerar o contexto do nosso problema e examinar mais detalhadamente os dados coletados. Esses valores são consistentes com padrões observados em solos destinados à produção agrícola. Por exemplo, valores de pH abaixo de 4 ou acima de 10, embora raros, não são impossíveis e, portanto, não devem ser descartados automaticamente como outliers.

Para solucionar problemas relacionados a dados, é fundamental compreender o contexto em que esses dados foram coletados. Esse entendimento é essencial para a continuidade das nossas análises. Sem esse conhecimento contextual, uma pessoa pode interpretar erroneamente valores atípicos, como concentrações de potássio incomuns no solo, podendo descartá-los indevidamente ou presumir que houve algum erro na coleta dos dados.
  
![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/4a1a0b96-7814-42eb-94fe-465477057117)

Analisando a distribuição dos nosso dados podemos observar algumas coisas:
* A distribuição normal na temperatura e no ph do solo já eram características esperadas para essea atributos. 
* Muitos atributos aparentam ter uma bimodalidade na distribuição, o que indica duas condições distintas para produção agrícola. Por exemplo, indicações de potássio entre 15~50 e uma contagem em ~200 indica que tem produtos que podem se beneficiar de valores extremos de concentração de potássio no solo. Isso pode ser observado também em umidade e fósforo.
* Vemos claramento a presença de outliers, são bastantes visuais. Mas novamento, isso não significa que são valores mal coletados, é necessário saber a realidade da produção agrícula e como nossos dados foram coletados.


**Vamos a média de nossas features para cada saída (plantação) possível.**
![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/e47a9815-b14d-42ee-9fb5-2d6cc666e196)

Podemos visualizar quais são os valores ideais para cada tipo de produção com base em nossos dados. Com esses valores, nosso modelo será capaz de recomendar a melhor cultura a ser plantada, dependendo dos valores coletados do solo.

Observamos que diferentes culturas podem ter características recomendadas semelhantes, apresentando valores indicados similares para alguns atributos do solo. No entanto, vamos desenvolver um modelo capaz de distinguir esses padrões similares e encontrar soluções específicas para cada caso.

Um produtor agrícola poderia analisar essas tabelas por dias para determinar o melhor curso de ação para sua produção. Contudo, com o modelo que iremos construir, nossa rede neural fornecerá essas respostas de forma automática, rápida e eficiente. Nosso modelo poderia obter muitas outras saídas e informações para aprimorar nosso modelo e aplicá-lo na prática em uma grande escala, mas, esta nossa abordagem atual já permitirá realizar testes e auxiliar pequenos empreendedores que não dispõem de todos os recursos necessários para escolher a melhor opção. 

### 04. Fazendo nosso modelo de Rede Neural

```python
# Criando o modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')
])

# Compilando o modelo
modelo.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])

# 'Fit' o modelo
history = modelo.fit(X_train,
           tf.one_hot(y_train, depth=22),
           epochs=50,
           validation_data=(X_test, tf.one_hot(y_test, depth=22)))
```
![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/d3c6dd59-d5b9-481f-b1bb-4672ce8a581d)

* *Fazendo nosso modelo normalizado*
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Ajustando nosso modelo de treino para normalizar
X_train_norm = scaler.fit_transform(X_train)
# Normalizando também nosso teste
X_test_norm = scaler.fit_transform(X_test)

# Criando o modelo
modelo_norm = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')
])

# Compilando o modelo
modelo_norm.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])

# 'Fit' o modelo
history_norm = modelo_norm.fit(X_train_norm,
           tf.one_hot(y_train, depth=22),
           epochs=50,
           validation_data=(X_test_norm, tf.one_hot(y_test, depth=22)))
```

Nosso modelo normalizado apresentou melhor acurácia do que nosso modelo sem nenhuma transformação dos dados. Vamos prosseguir com nossos dados normalizados. Modelos de rede neural tendem a operar melhor com os dados tratados.

![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/549a3624-3ac3-4dbf-9420-4a42a1ff69be)

### 05. Validação do nosso modelo

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       1.00      1.00      1.00        33
           2       1.00      1.00      1.00        27
           3       1.00      1.00      1.00        36
           4       0.97      1.00      0.99        36
           5       0.97      0.85      0.90        33
           6       0.86      1.00      0.93        19
           7       1.00      1.00      1.00        28
           8       0.95      0.66      0.78        32
           9       1.00      1.00      1.00        41
          10       0.96      1.00      0.98        25
          11       0.84      0.90      0.87        29
          12       1.00      1.00      1.00        23
          13       1.00      0.97      0.98        29
          14       1.00      1.00      1.00        25
          15       1.00      0.89      0.94        36
          16       1.00      0.96      0.98        26
          17       0.97      1.00      0.99        37
          18       1.00      1.00      1.00        27
          19       1.00      1.00      1.00        30
          20       0.75      0.96      0.84        28
          21       0.87      1.00      0.93        27


![image](https://github.com/PedroHenrique0214/CulturaAgricola_rede_neural/assets/155765414/8e222d8b-839a-4eac-b1c5-da064dcb6671)
