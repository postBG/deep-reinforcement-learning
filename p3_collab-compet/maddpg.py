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

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        """Takes in a 2 X 24 so need to process 1 X 24 states per each agent"""

        actions = []

        for i in range(self.num_agents):
            action = self.maddpg_agent[i].act(obs_all_agents[i, :].view(1, -1), noise)  # without view dim is [24]
            actions.append(action.squeeze())  # without squeeze dim is (1 X 2 X 2) in the main code

        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """

        target_actions = []

        for i in range(self.num_agents):
            action = self.maddpg_agent[i].target_act(obs_all_agents[:, i, :], noise)  # without view dim is [24]
            target_actions.append(action)  # without squeeze dim is (1 X 2 X 2) in the main code

        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        state, full_state, action, reward, next_state, full_next_state, done = samples

        '''
        instead of loading in full_state, and full_next_state, I could just sample state and next_state and then combine 
        them into Batch_size * State_size tensors as well.

        '''

        batch_size = full_state.shape[0]

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network

        # change the shape to batch_size X num_agents X remaining.
        # by changing the shape we can select local observations by agents by selecting [:, i, :] for i in range(num_agents)

        target_actions = self.target_act(next_state.view(batch_size, self.num_agents, -1))

        # turn a list of 2x2 into a batch_size x (action_size * num_agent)
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(full_next_state, target_actions.to(device))

        # shape of reward is batch_size X num_agents so [:, agent_number] is needed to pick the rewards for the specific agent
        # that is being updated.
        y = reward[:, agent_number] + self.discount_factor * q_next * (1 - done[:, agent_number])

        q = agent.critic(full_state, action.view(batch_size, -1))

        critic_loss = F.mse_loss(q, y.detach())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [
            self.maddpg_agent[i].actor(state.view([batch_size, self.num_agents, -1])[:, i, :]) if i == agent_number else
            self.maddpg_agent[i].actor(state.view([batch_size, self.num_agents, -1])[:, i, :]).detach() for i in
            range(self.num_agents)]

        full_q_input = torch.cat(q_input, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already

        # get the policy gradient
        actor_loss = -agent.critic(full_state, full_q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        # soft update the models
        # having update in here as well makes the model converge faster
        # I could increase the trasnfer rate as well. instead of having updates called twice.
        self.update_targets()
        return critic_loss.item(), actor_loss.item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)







