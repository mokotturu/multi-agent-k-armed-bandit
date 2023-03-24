import numpy as np
import matplotlib.pyplot as plt

N_AGENTS = 3
N_STEPS = 10000


class Searcher:
    
        def __init__(self, mean, std, n_steps):
            self.mean = mean
            self.std = std
            self.n_steps = n_steps
            self.actions = np.random.normal(mean, std, n_steps)


class Rescuer:
    def __init__(self, picking_func, name="None", n_agents=N_AGENTS, n_steps=N_STEPS, agents=None):
        self.name = name
        self.picking_func = picking_func
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.agents = agents
        self.rewards = []

    def pick(self, agents, n_agents, steps):
        picks = {agent: [] for agent in agents} 

        for step in range(1, steps+1):
            step_rewards = {agent: None for agent in agents}
            
            for agent in agents:
                step_rewards[agent] = self.picking_func(picks[agent], step)


            best_agent = max(step_rewards, key=step_rewards.get)
            

            picks[best_agent].append(best_agent.actions[step])

            if step == 2:
                for pick in picks:
                    print(picks[pick])

                for agent in step_rewards:
                    print(step_rewards[agent])

        return picks
    
    def run(self):
        picks = self.pick(self.agents, self.n_agents, self.n_steps)
        self.rewards = sum([sum(picks[agent]) for agent in self.agents])
        selections = {agent: len(picks[agent]) for agent in self.agents}
        return self.rewards, selections, self.name
                  

             


def random_pick(history, current_step):
    rando = np.random.randint(0, 10)
    return rando

def mean_pick(history, current_step):
    return np.mean(history)

def std_pick(history, current_step):
    if len(history) == 0:
        return 0
    return np.std(history)

def max_pick(history, current_step):
    if len(history) == 0:
        return 0
    return np.max(history)

# def min_pick(history, current_step):
#     return np.min(history)

# def median_pick(history, current_step):

#     return np.median(history)

def mean_std_pick(history, current_step):
    if len(history) == 0:
        return 0
    return np.mean(history) / np.std(history)

def UCB1(history, current_step):
    if len(history) == 0:
        return np.mean(history) + np.sqrt(2 * np.log(current_step) / 1)
    return np.mean(history) + np.sqrt(2 * np.log(current_step) / len(history))

def UCB2(history, current_step):
    if len(history) == 0:
        return np.mean(history) + np.sqrt(2 * np.log(current_step) / 1) + np.sqrt(2 * np.log(current_step) / 1) 
    return np.mean(history) + np.sqrt(2 * np.log(current_step) / len(history)) + np.sqrt(2 * np.log(current_step) / len(history)) * np.std(history)


agents = []

A = Searcher(-10, 0, N_STEPS)
B = Searcher(-20, 12, N_STEPS)
C = Searcher(500, 5, N_STEPS)

agents.append(A)
agents.append(B)
agents.append(C)

             
random_rescuer = Rescuer(random_pick, agents=agents, n_agents=1, n_steps=1000, name="random")

mean_rescuer = Rescuer(mean_pick, agents=agents, n_agents=1, n_steps=1000, name="mean")

std_rescuer = Rescuer(std_pick, agents=agents, n_agents=1, n_steps=1000,    name="std")

max_rescuer = Rescuer(max_pick, agents=agents, n_agents=1, n_steps=1000, name="max")

# min_rescuer = Rescuer(min_pick, agents=agents, n_agents=1, n_steps=1000)

# median_rescuer = Rescuer(median_pick, agents=agents, n_agents=1, n_steps=1000)

mean_std_rescuer = Rescuer(mean_std_pick, agents=agents, n_agents=1, n_steps=1000,  name="mean_std")

UCB1_rescuer = Rescuer(UCB1, agents=agents, n_agents=1, n_steps=1000, name="UCB1")

UCB2_rescuer = Rescuer(UCB2, agents=agents, n_agents=1, n_steps=1000, name="UCB2")

rescuers = [random_rescuer, mean_rescuer, std_rescuer, max_rescuer, mean_std_rescuer, UCB1_rescuer, UCB2_rescuer]

for rescuer in rescuers:
    rewards, selections, name = rescuer.run()
    print(f"Reward for {name} : {rewards}")
    print(f"Selections for {name} :")
    for agent in selections:
        print(f"{selections[agent]}")



    






    



