# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utils import soft_update
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, state_size, action_size, num_agents, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # initialize 2 DDPGAgents.

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents),
                             DDPGAgent(state_size, action_size, num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, states_of_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(states.unsqueeze(0), noise) for agent, states in
                   zip(self.maddpg_agent, states_of_all_agents)]
        actions = [action.squeeze() for action in actions]
        return actions

    def target_act(self, states_of_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(states, noise) for agent, states in
                          zip(self.maddpg_agent, states_of_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        states, full_states, actions, rewards, next_states, next_full_states, dones = samples
        batch_size = full_states.shape[0]

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        target_actions = self.target_act(next_states)

        # turn a list of 2x2 into a batch_size x (action_size * num_agent)
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(next_full_states, target_actions.to(device))

        y = rewards[agent_number] + self.discount_factor * q_next * (1 - dones[agent_number])

        q = agent.critic(full_states, actions.view(batch_size, -1))

        critic_loss = F.mse_loss(q, y.detach())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.maddpg_agent[i].actor(state) if i == agent_number
                   else self.maddpg_agent[i].actor(state).detach()
                   for i, state in enumerate(states)]

        full_q_input = torch.cat(q_input, dim=1)

        actor_loss = -agent.critic(full_states, full_q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        self.update_targets()
        return critic_loss.item(), actor_loss.item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
