# RL Agent that plays 2048 using Actor-Critic. Coded by Andy 12/18/2019
from andy_2048 import BoardEnv
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_probability as tfp

board_width = 4

num_episodes = 1
max_steps_per_episode = 500

p_discount_rate = 0.95
q_discount_rate = 0.95

p_model = keras.Sequential([keras.layers.Flatten(),
                            keras.layers.Dense(10, activation="relu"),
                            keras.layers.Dense(4, activation="softmax")])
q_model = keras.Sequential([keras.layers.Flatten(),
                            keras.layers.Dense(10, activation="relu"),
                            keras.layers.Dense(4)])
optimizer = keras.optimizers.Adam(lr=0.01)
p_loss_fn = keras.losses.CategoricalCrossentropy()
q_loss_fn = keras.losses.MSE  # It's a little weird that this and CCE (above) behave differently.

b = BoardEnv()
done = False
for episode_num in range(num_episodes):
    state = b.reset()
    action_probs = tf.squeeze(p_model(state[np.newaxis]), axis=0)
    dice_roll = tfp.distributions.Multinomial(total_count=1, probs=action_probs).sample(1)
    action = b.action_space[np.argmax(dice_roll)]
    for step_num in range(max_steps_per_episode):
        # compute s'
        next_state, reward, done = b.step(action)
        if np.array_equal(next_state, state):  # don't keep trying dud moves
            break
        # compute a' and grad log pi(a'|s')
        with tf.GradientTape() as tape1:
            action_probs = tf.squeeze(p_model(next_state[np.newaxis]), axis=0)
            dice_roll = tfp.distributions.Multinomial(total_count=1, probs=action_probs).sample(1)
            p_loss = p_loss_fn(dice_roll, action_probs)
        p_grads = tape1.gradient(p_loss, p_model.trainable_variables)
        next_action = b.action_space[np.argmax(dice_roll)]
        # compute q(s,a), q(s',a') and update q_model
        with tf.GradientTape() as tape2:
            q_val = tf.squeeze(q_model(state[np.newaxis]))[action]
            next_q_val = tf.squeeze(q_model(next_state[np.newaxis]))[next_action]
            target_q_val = reward + (1 - done) * q_discount_rate * next_q_val
            print(q_val)
            print(reward)
            print(next_q_val)
            print(target_q_val)
            q_loss = q_loss_fn(q_val, target_q_val)
        optimizer.minimize(q_loss, q_model.trainable_variables)
        # update p_model using policy gradient formula: grad log pi(a'|s') * q_val(s',a')
        optimizer.apply_gradients((p_grads * next_q_val, p_model.trainable_variables))
        # get ready to loop
        state = next_state
        action = next_action
        if done:
            break
    print("%s steps in episode %s" % (step_num, episode_num))

