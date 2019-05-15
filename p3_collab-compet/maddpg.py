import torch

from ddpg import DDPGAgent
from utils import soft_update, to_tensor, to_full


class MADDPG(object):
    def __init__(self, state_size, action_size, num_agents, discount_factor=0.95, tau=0.02):
        super().__init__()

        self.ddpg_agents = [DDPGAgent(state_size, action_size, num_agents) for _ in range(num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.critic = torch.nn.SmoothL1Loss()

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.ddpg_agents]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.ddpg_agents]
        return target_actors

    def act(self, states_of_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(states, noise) for agent, states in zip(self.ddpg_agents, states_of_all_agents)]
        return actions

    def target_act(self, states_of_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(states, noise) for ddpg_agent, states in
                          zip(self.ddpg_agents, states_of_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        states, full_states, actions, rewards, next_states, next_full_states, done = map(to_tensor, samples)

        agent = self.ddpg_agents[agent_number]

        critic_loss = self.update_critic(actions, agent, agent_number, done, full_states, next_full_states, next_states,
                                         rewards)
        actor_loss = self.update_actor(agent, agent_number, full_states, states)

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_critic(self, actions, agent, agent_number, done, full_states, next_full_states, next_states, rewards):
        agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            target_actions = self.target_act(next_states)
            full_target_actions = to_full(target_actions)

            q_next = agent.target_critic(next_full_states, full_target_actions)
        q_target = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (
                1 - done[agent_number].view(-1, 1))
        full_actions = to_full(actions)
        q = agent.critic(full_states, full_actions)
        critic_loss = self.critic(q, q_target.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        return critic_loss

    def update_actor(self, agent, agent_number, full_states, states):
        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.ddpg_agents[i].actor(state) if i == agent_number else self.ddpg_agents[i].actor(state).detach()
                   for i, state in enumerate(states)]
        full_q_input = to_full(q_input)
        actor_loss = -agent.critic(full_states, full_q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()
        return actor_loss

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.ddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
