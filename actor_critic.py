# RL Agent that plays 2048 using Actor-Critic. Coded by Andy 12/18/2019
from andy_2048 import BoardEnv
import mlflow
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_probability as tfp

params = {}
params["board_width"] = 4

params["num_episodes"] = 10000
params["max_steps_per_episode"] = 500

params["p_discount_rate"] = 0.95
params["q_discount_rate"] = 0.95

params["learning_rate"] = 0.01

with mlflow.start_run() as run:
    mlflow.log_params(params)
    p_model = keras.Sequential([keras.layers.Flatten(),
                                keras.layers.Dense(10, activation="relu"),
                                keras.layers.Dense(4, activation="softmax")])
    q_model = keras.Sequential([keras.layers.Flatten(),
                                keras.layers.Dense(10, activation="relu"),
                                keras.layers.Dense(4)])
    q_model.build(input_shape=(1, 16))
    optimizer = keras.optimizers.Adam(lr=params["learning_rate"])
    p_loss_fn = keras.losses.CategoricalCrossentropy()

    b = BoardEnv()
    done = False
    for episode_num in range(params["num_episodes"]):
        state = b.reset()
        action_probs = tf.squeeze(p_model(state[np.newaxis]), axis=0)
        dice_roll = tfp.distributions.Multinomial(total_count=1, probs=action_probs).sample(1)
        action = b.action_space[np.argmax(dice_roll)]
        game_score = 0
        for step_num in range(params["max_steps_per_episode"]):
            # compute s'
            next_state, reward, done = b.step(action)
            if np.array_equal(next_state, state):  # don't keep trying dud moves
                break
            # compute a' and grad log pi(a'|s')
            with tf.GradientTape() as p_tape:
                action_probs = tf.squeeze(p_model(next_state[np.newaxis]), axis=0)
                dice_roll = tfp.distributions.Multinomial(total_count=1, probs=action_probs).sample(1)
                p_loss = p_loss_fn(dice_roll, action_probs)
            p_grads = p_tape.gradient(p_loss, p_model.trainable_variables)
            next_action = b.action_space[np.argmax(dice_roll)]
            # compute q(s,a), q(s',a') and update q_model
            with tf.GradientTape() as q_tape:
                q_val = tf.squeeze(q_model(state[np.newaxis]))[action]
                next_q_val = tf.squeeze(q_model(next_state[np.newaxis]))[next_action]
                target_q_val = reward + (1 - done) * params["q_discount_rate"] * next_q_val
                q_loss_fn = tf.math.square(q_val - target_q_val)
            q_grads = q_tape.gradient(q_loss_fn, q_model.trainable_variables)
            optimizer.apply_gradients(zip(q_grads, q_model.trainable_variables))
            # update p_model using policy gradient formula: grad log pi(a'|s') * q_val(s',a')
            updated_grads = [i * next_q_val for i in p_grads]
            optimizer.apply_gradients(zip(updated_grads, p_model.trainable_variables))
            # get ready to loop
            # print(state)
            # print(action)
            # print(next_state)
            # print(next_action)
            # print(q_val)
            # print(reward)
            # print(next_q_val)
            # print(target_q_val)
            # print()
            state = next_state
            action = next_action
            game_score += reward
            if done:
                break
        print("%s steps in episode %s, score: %s" % (step_num, episode_num, game_score))
        mlflow.log_metric("game scores", game_score)
