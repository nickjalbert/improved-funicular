# RL Agent that plays 2048 using Actor-Critic. Coded by Andy 12/18/2019
from envs.nick_2048 import Nick2048
import logging
import mlflow
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_probability as tfp

#logging.basicConfig(level=logging.DEBUG)



params = {}
params["num_episodes"] = 1000
params["max_steps_per_episode"] = 500
params["alpha"] = 0.95
params["learning_rate"] = 0.001

with mlflow.start_run() as run:
    mlflow.log_params(params)
    q_models = [keras.Sequential(
        [
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1),
        ]
    )] * Nick2048.action_space.n
    [m.build(input_shape=(1, 16)) for m in q_models]
    optimizer = keras.optimizers.Adam(lr=params["learning_rate"])

    b = Nick2048()
    done = False
    for episode_num in range(params["num_episodes"]):
        state = b.reset()
        state_val = None
        action = b.action_space.sample()
        game_score = 0
        # Pseudo code for our Sarsa learning algo:
        #   for each step in the rollout:
        #     action = fancy_argmax_a(q(get_afterstate(s),a))
        #     q_val = model(get_afterstate(s),action)
        #     next_s, r = b.step(a)
        #     next_action = fancy_argmax_a(q(get_afterstate(next_s), a))
        #     next_q_val = model(get_afterstate(new_s),next_action)
        #     update q_model using loss(q_val - (r + next_q_val)
        for step_num in range(params["max_steps_per_episode"]):
            with tf.GradientTape() as q_tape:
                logging.debug(f"state:\n{np.asarray(state).reshape([4,4])}")
                candidate_actions = list(range(b.action_space.n))
                afterstates = [b.get_afterstate(state, action) for action in candidate_actions]
                afterstate_q_vals = [tf.squeeze(q_models[i](np.array(afterstates[i])[np.newaxis])) for i in candidate_actions]
                logging.debug(f"afterstate_q_vals : {afterstate_q_vals}")
                # pick action by rolling dice according to relative values of afterstates
                while True:
                    dice_roll = tfp.distributions.Multinomial(
                        total_count=10, logits=afterstate_q_vals
                    ).sample(1)
                    action_index = np.argmax(dice_roll)
                    action = candidate_actions[action_index]
                    next_state, reward, done, _ = b.step(action)
                    if next_state != state:  # you found a valid move
                        break
                    else:  # that wasn't a valid move, but one must exist since we weren't done after last step.
                        logging.debug(f"action {action} was invalid, removing it from candidate and rolling dice again")
                        assert len(candidate_actions) > 1, "No actions changed the board but we are not done."
                    afterstate_q_vals = afterstate_q_vals[:action_index] + afterstate_q_vals[action_index + 1:]
                    candidate_actions = candidate_actions[:action_index] + candidate_actions[action_index + 1:]
                logging.debug(f"action: {action}")
                logging.debug(f"afterstate:\n{np.asarray(afterstates[action]).reshape([4,4])}")
                afterstate_q_val = afterstate_q_vals[action_index]
                logging.debug(f"afterstate_q_val: {afterstate_q_val}")
                logging.debug(f"reward: {reward}")
                logging.debug(f"next_state:\n{np.asarray(next_state).reshape([4,4])}")

                # update q_model via TD learning using q(s,a) (which we computed last loop iter) and q(s',a')
                next_candidate_actions = list(range(b.action_space.n))
                next_afterstates = [b.get_afterstate(next_state, action) for action in next_candidate_actions]
                next_afterstate_q_vals = [tf.squeeze(q_models[i](np.array(next_afterstates[i])[np.newaxis]))
                                          for i in next_candidate_actions]
                logging.debug(f"next_afterstate_q_vals: {next_afterstate_q_vals}")
                next_action = np.argmax(next_afterstate_q_vals)
                logging.debug(f"next_action: {next_action}")
                next_afterstate_q_val = next_afterstate_q_vals[next_action]
                target_afterstate_q_val = (
                        reward + (1 - done) * params["alpha"] * next_afterstate_q_val
                )
                logging.debug(f"next_afterstate_q_val: {next_afterstate_q_val}")
                logging.debug(f"target_afterstate_q_val: {target_afterstate_q_val}")
                val_loss_fn = tf.math.square(afterstate_q_val - target_afterstate_q_val)
            val_grads = q_tape.gradient(val_loss_fn, q_models[action].trainable_variables)
            optimizer.apply_gradients(zip(val_grads, q_models[action].trainable_variables))
            logging.debug("\n")

            # get ready to loop
            state = next_state
            state_val = next_afterstate_q_val
            action = next_action
            game_score += reward
            if done:
                break
        print("%s steps in episode %s, score: %s" % (step_num + 1, episode_num, game_score))
        mlflow.log_metric("game scores", game_score)
