import keras

def build_model(input_dim: int, num_classes: int = 14):

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

    )

