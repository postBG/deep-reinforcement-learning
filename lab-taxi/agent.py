import numpy as np
from collections import defaultdict


def get_epsilon_greedy_prob(action_values, epsilon):
    # Greedy Action이 여러 개인 경우도 고려
    nA = action_values.size
    greedy_action_indicator = (action_values >= np.max(action_values)).astype(np.float)
    num_greedy_action = np.sum(greedy_action_indicator)
    normed_greedy_action_prob = (1 - epsilon) / num_greedy_action
    epsilon_greedy_prob = (greedy_action_indicator * normed_greedy_action_prob) + (epsilon / nA)
    return epsilon_greedy_prob


def update_Q_expected_sarsa(Q, state, action, reward, alpha, gamma, epsilon, next_state=None):
    epsilon_greedy_prob = get_epsilon_greedy_prob(Q[next_state], epsilon)
    Qsa_next = np.dot(epsilon_greedy_prob, Q[next_state])
    return Q[state][action] + alpha * (reward + gamma * Qsa_next - Q[state][action])


def choose_epsilon_greedy_action(action_values, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(action_values)
    return np.random.randint(0, action_values.size)


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 1.
        self.epsilon = 0.005
        self.gamma = 0.5
        self.steps = 1.

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return choose_epsilon_greedy_action(self.Q[state], self.epsilon)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.steps += 1
        self.epsilon = max(0.005, 1 / self.steps)
        self.Q[state][action] = update_Q_expected_sarsa(self.Q, state, action, reward, self.alpha, self.gamma,
                                                        self.epsilon, next_state)
