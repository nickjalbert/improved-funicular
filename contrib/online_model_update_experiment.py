import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow.keras as keras


model = keras.Sequential(
        [
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(20, activation="relu"),
            #keras.layers.Dense(512, activation="relu"),
            #keras.layers.Dense(256, activation="relu"),
            #keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(2),
        ]
    )
model.build(input_shape=(1,4))
loss_fn = keras.losses.mean_squared_error
optimizer = keras.optimizers.Adam(lr=.001)

# we are going to train the model one gradient step at a time to figure
# out that from the input [a, b, c, d], the output should be  (a, d).

with mlflow.start_run():
    mlflow.tensorflow.autolog()
    for i in range(10000):
        with tf.GradientTape() as tape:
            input = tf.random.uniform(shape=(1, 4))
            y = model(input)
            y_target = tf.stack([input[0,0], input[0,3]])
            loss = loss_fn(y_target, y)
            print(f"loss: {loss}")
            mlflow.log_metric("loss", loss.numpy()[0], step=i)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


