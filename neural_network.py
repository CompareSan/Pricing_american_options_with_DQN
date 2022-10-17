import tensorflow as tf


def build_dqn(learning_rate, n_actions, input_dims, n_neurons_1, n_neurons_2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons_1, activation='relu', input_shape=(input_dims,)))
    model.add(tf.keras.layers.Dense(n_neurons_2, activation='relu'))
    model.add(tf.keras.layers.Dense(n_actions))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError())

    return model
