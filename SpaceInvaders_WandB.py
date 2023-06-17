import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import wandb

from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
import pathlib
import gymnasium as gym
import os.path

print(tf.__version__)
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ale = ALEInterface()
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
    next_state, reward, done, _,info = env.step(action)
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
            if isinstance(states[i], tuple) and len(states[i]) == 2 and isinstance(states[i][0], np.ndarray) and isinstance(states[i][1], dict):
                states[i] = states[i][0]
            elif states[i].shape != (210, 160, 3):
                states[i] = states[i-1]
        all_Q_values = main_nn(tf.convert_to_tensor(np.stack([np.array(state, dtype=object) for state in states]).astype('float32')))
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values.astype('float32'), Q_values))
        print(loss)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss

def play_one_step_train(env, state, model):
    # Verificar si el estado es una tupla
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
        state = state[0]

    # Utilizar el modelo para predecir acciones
    Q_values = model.predict(state[np.newaxis])
    action = np.argmax(Q_values[0])
    
    next_state, reward, done, _,_ = env.step(action)
    if next_state.dtype == np.uint8:
        replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward

model_file = 'C:/Users/Pedro/Downloads/Proyecto 2/proyecto2-brayain_proyecto2/proyecto2-brayain_proyecto2/my_dqn_2.h5'

if os.path.isfile(model_file):
    model = keras.models.load_model(model_file)

    env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    obs = env.reset()

    while True:
        obs, reward = play_one_step_train(env, obs, model)


else:
    wandb.init(project="SI Project")
    for episode in range(400):
        
        obs = env.reset()
        
        for step in range(150):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward = play_one_step(env, obs, epsilon)

            if episode > 70:
                loss = training_step(70)

                wandb.log({"episode": episode, "total_reward": reward, "loss": loss})
        print(f"Episode: {episode}")    
    main_nn.save('my_dqn_f.h5')



