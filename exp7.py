import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

for init in ['glorot_uniform', 'he_normal']:
    for opt in ['sgd', 'adam', 'rmsprop']:

        model = models.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, 3, activation='relu',
                          kernel_initializer=init),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu',
                         kernel_initializer=init),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)
        acc = model.evaluate(x_test, y_test, verbose=0)[1]

        print(init, opt, "Accuracy:", round(acc, 4))