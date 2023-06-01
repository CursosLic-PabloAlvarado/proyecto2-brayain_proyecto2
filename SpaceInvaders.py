import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque

from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
import pathlib
import gymnasium as gym

ale = ALEInterface()

def roms() -> list[pathlib.Path]:
    pass

ale.loadROM(SpaceInvaders)

env = gym.make('ALE/SpaceInvaders-v5')

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

main_nn = keras.Sequential([
    keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(210, 160, 3)),
    keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
    keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(n_outputs)
])

target_nn = keras.models.clone_model(main_nn)

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.mean_squared_error

replay_buffer = deque(maxlen=10000)

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:

        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray) and isinstance(state[1], dict):
            Q_values = main_nn.predict(state[0][np.newaxis])
        else:
            Q_values = main_nn.predict(state[np.newaxis])
            
        return np.argmax(Q_values[0])
        
    

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch], dtype=object)
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


    
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, _,_ = env.step(action)
    if next_state.dtype == np.uint8:
        replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward

discount_rate = 0.99
    

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target_nn.predict(next_states.astype('float32'))
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions.astype('int32'), n_outputs)
    with tf.GradientTape() as tape:
        for i in range(len(states)):
            if isinstance(states[i], tuple):
                states[i] = states[i-1]
        all_Q_values = main_nn(tf.convert_to_tensor(np.stack([np.array(state, dtype=object) for state in states]).astype('float32')))
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values.astype('float32'), Q_values))
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

    

for episode in range(600):
    
    if episode == 590:
        env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    
    obs = env.reset()
    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward = play_one_step(env, obs, epsilon)

        if episode > 50:
            training_step(16)
    print(f"Episode: {episode}")
    
main_nn.save('my_dqn.h5')



