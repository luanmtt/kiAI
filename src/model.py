import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


'''
    
    train_test_split    → faz o split dos dados em teste, treino
    LabelEncoder        → faz o processo de codificação "label" → índice
    StandardScaler      → normaliza cada feature tal que mean=0 e std=1

'''


# --------------------------------------------------------------------------------------------------
# features:
# o modelo aprende, a partir dessas features, quais são as combinações entre elas que predizem cada label.


FEATURES = [

    "bpm", "ar", "cs", "od", "difficulty_rating",
    "stream_density", "mean_distance", "std_distance",
    "mean_velocity", "slider_ratio", "interval_variance_log",
    "kiai_section_count", "kiai_note_ratio", "mean_kiai_dist",
    "notes_per_second", "base_bpm"
]


# --------------------------------------------------------------------------------------------------
# importar dataset.csv

'''
    
    Importando o .csv normalmente. Aqui, retiramos todos os linhas que possuem NaN.
    Além disso, tiramos o campo "_maps" para ficar apenas o nome da feature:
        
        stream_maps → stream
        jump_maps   → jump
            ...

'''

df = pd.read_csv("data/processed/dataset.csv")
df["label"] = df["label"].str.replace("_maps", "")

df = df.dropna(subset=FEATURES + ["label"])

X = df[FEATURES].values
y_raw = df["label"].values


# --------------------------------------------------------------------------------------------------
# encoding labels

'''
    Conversão das labels → números indexados.
    
        • ["jump", "stream", "gimmick", ...] → [0,1,2,...] 
        → o encoder tem que ser salvo posteriormente para a análise reversa dos dados.
    
'''

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = int(np.array(le.classes_).size)


print(f"Classes ({num_classes}): {le.classes_}")


# --------------------------------------------------------------------------------------------------
# dividindo dados do dataset:

'''
    Divisão 80/20 para treino e teste, respectivamente.
'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42, stratify=y
)


# --------------------------------------------------------------------------------------------------
# scaling

'''
    
    Basicamente, fazendo uma escala comum dentre todas as features.
    →   a escala é construída apenas com dados de treino. se fosse com
        todos os dados, os dados de teste influenciam a escala, contaminando.
'''

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------------------------------------------------------
# pesos para labels:

'''
   
    Temos injustiça entre labels, e para não sofrer com isso (nem cortar label de menor quantidade),
    podemos dar pesos às labels. Isso será útil, porque quando o modelo classificar erroneamente uma 
    label rara (o que faz sentido), o erro será maior e haverá aprendizado.

'''

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))


# --------------------------------------------------------------------------------------------------
# model:

'''
    No keras, é feito o desenvolvimento fácil do processo:
        
        → Forward pass:
        • X  --wds-->  raw1  --relu-->  act1  --wds-->  raw2  --relu-->  act2  --wds-->  raw3  --sigmoid--> act3 (outputted)   
        
        → Backprop:
        • Propagação de erros via regra da cadeia.

        Nesse modelo, usa-se:

            X:              input_size = len(FEATURES)
                                |
                                ↓
            hiddenlayer1:   128 neurons, fully connected
                                |
                               relu
                                ↓
            hiddenlayer2:   64 neurons, fully connected
                                |
                               relu
                                ↓
            Y:              output_size = len(labels)

            
        →   Dropout: desconectam-se num_neurons * parâmetro, para evitar overfitting, isto é
            DECORAR em vez de aprender. A desconexão de um neuron evita a dependência dele, ou seja,
            das conexões que ele faz, para aprender.

        →   Uso das técnicas:
            
            .   optimizer = adam: um otimizador de learning rate. Baseado nos gradientes de erro advindos
                de backprop, ajusta o lr.

            .   loss = sparse_categorical_crossentropy: sparse para "números"; categorical porque vamos
                classificar valores para labels (um processo de categorização) e crossentropy penaliza
                predições confiantes, porém erradas, com vigor.

            .   metrics = accuracy: a métrica de output que o modelo seguirá para o monitoramento humano.

'''

def build_model(input_dim: int, num_classes: int):

    model = keras.Sequential([
        
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64,activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),

    ])

    model.compile(
        
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = build_model(input_dim=len(FEATURES), num_classes=num_classes)
model.summary()


# --------------------------------------------------------------------------------------------------
# treinamento do modelo:

'''
    Algumas características notáveis:

        → validation_split: prevêm overfitting no começo do projeto segurando 10% dos dados de treino.
        → 50 epochs, sendo os pesos atualizados depois de 32 mapas.
        → passa o class_weight_dict criado na etapa de 'pesos para labels'.
        → o callbacks vai parar o treinamento se, por 5 epochs, não for observado melhora no custo.
'''


history = model.fit(

    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks= [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)


# --------------------------------------------------------------------------------------------------
# evaluação:

'''
    model.evaluate() retorna precisão, retomada e F1 para cada classe. isto é:
        
        → precisão: de todos os previstos como jump, quais eram de fato jump?
        → retomada: de todos os mapas que eram de fato jump, quantos foram previstos?
        → F1: a média desses parâmetros acima. é, no fim de tudo, a nota do modelo.
'''

loss, acc = model.evaluate(X_test, y_test, verbose="silent")
print(f"Test accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, y_pred, target_names=le.classes_))


# --------------------------------------------------------------------------------------------------
# salvar modelo:

'''
    Salvando: modelo, scaler das labels e o LabelEncoder, também das labels.
'''

model.save("models/model.keras")
pickle.dump(scaler, open("data/processed/scaler.pkl", "wb"))
pickle.dump(le, open("data/processed/le.pkl", "wb"))

print("Saved model, scaler and LE.")


# --------------------------------------------------------------------------------------------------
