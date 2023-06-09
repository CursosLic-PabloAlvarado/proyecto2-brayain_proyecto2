import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import os.path
import wandb
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.v_head(x)
        return v


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            mu, sigma = self.actor(state)
            dist = Normal(mu, sigma)
            action = dist.sample()
            action = action.clamp(-1.0, 1.0) # recorta la acción para que esté dentro de los límites
            log_prob = dist.log_prob(action).sum(-1).item() # convierte a escalar
            value = self.critic(state).item() # convierte a escalar
            return action.numpy(), log_prob, value

    def update(self, states, actions, log_probs_old, returns):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old).unsqueeze(-1)
        returns = torch.FloatTensor(returns).unsqueeze(-1)

        for _ in range(10): # actualiza varias veces
            mu, sigma = self.actor(states)
            dist = Normal(mu, sigma)
            log_probs_new = dist.log_prob(actions).sum(-1).unsqueeze(-1)
            ratio = (log_probs_new - log_probs_old).exp()
            values_new = self.critic(states)

            advantages_new = returns - values_new
            actor_loss_new = -(ratio * advantages_new).mean()
            critic_loss_new = advantages_new.pow(2).mean()

            # actualiza el actor
            self.actor_optimizer.zero_grad()
            actor_loss_new.backward(retain_graph=True)
            self.actor_optimizer.step()

            # actualiza el crítico
            self.critic_optimizer.zero_grad()
            critic_loss_new.backward(retain_graph=True)
            self.critic_optimizer.step()
        return actor_loss_new.item(), critic_loss_new.item()

def compute_returns(rewards, values_next, gamma=0.99):
    R = values_next
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0,R)
    return returns

def extract_state(output):
    if isinstance(output, tuple):
        state = output[0]
    else:
        state = output
    return state


env = gym.make('Ant-v4',
               render_mode='human',
               ctrl_cost_weight=0.25,
               use_contact_forces=True,
               healthy_reward=1.5, 
               healthy_z_range=(0.2, 1.0),
               terminate_when_unhealthy=False)


state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]

ppo=PPO(state_dim=state_dim,
       action_dim=action_dim)

num_episodes=1000
num_steps=1000

if os.path.isfile('C:/Users/Pedro/Downloads/Proyecto 2/proyecto2-brayain_proyecto2/proyecto2-brayain_proyecto2/ppo_actor_model.pt')and os.path.isfile('C:/Users/Pedro/Downloads/Proyecto 2/proyecto2-brayain_proyecto2/proyecto2-brayain_proyecto2/ppo_critic_model.pt'):

    # Si los archivos de entrenamiento existen, carga el modelo desde los archivos
    ppo = PPO(state_dim=state_dim, action_dim=action_dim)
    ppo.actor.load_state_dict(torch.load('C:/Users/Pedro/Downloads/Proyecto 2/proyecto2-brayain_proyecto2/proyecto2-brayain_proyecto2/ppo_actor_model.pt'))
    ppo.critic.load_state_dict(torch.load('C:/Users/Pedro/Downloads/Proyecto 2/proyecto2-brayain_proyecto2/proyecto2-brayain_proyecto2/ppo_critic_model.pt'))

    # Código para mostrar en pantalla como se mueve el objeto con los datos de entrenamiento guardados
    state = extract_state(env.reset())
    for step in range(num_steps):
        action, _, _ = ppo.select_action(state)
        next_state, reward, _, _, _ = env.step(action)
        env.render()
        state = extract_state(next_state)
    env.close()
else:
    wandb.init(project="Spider Project")
    # Si los archivos de entrenamiento no existen, entrena el modelo y guarda los archivos
    for episode in range(num_episodes):
        state=extract_state(env.reset())
        rewards=[]
        states=[]
        actions=[]
        log_probs=[]
        values=[]

        for step in range(num_steps):
            action,log_prob,value=ppo.select_action(state)
            next_state,reward,_ ,_ ,_=env.step(action)

            rewards.append(reward)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

            state=next_state

        _,_,value_next=ppo.select_action(state)
        returns=compute_returns(rewards,value_next)

        actor_loss, critic_loss = ppo.update(states,actions,log_probs,returns)

        print(f'Episode: {episode+1}, Reward: {sum(rewards)}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}')
        wandb.log({"episode": episode+1, "total_reward": reward,"actor loss":actor_loss, "critic loss":critic_loss})
    # Guarda el modelo en un archivo después de que haya sido entrenado
    torch.save(ppo.actor.state_dict(), 'ppo_actor_model.pt')
    torch.save(ppo.critic.state_dict(), 'ppo_critic_model.pt')
