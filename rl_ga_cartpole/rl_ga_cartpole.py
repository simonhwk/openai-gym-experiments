# Evolving NN RL parameters with GA

import gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

print("Action Space: ", env.action_space)
print("Observation Space: ", env.observation_space)

print(gym.__version__)
class NeuralNetwork:
    def __init__(self, input_size, output_size):
        # weight matrix encodes two seprate linear models, one for each possible action
        self.weights = np.random.randn(input_size, output_size)
        print("Weight Matrix Shape:", self.weights.shape)


    def predict(self, observation):
        # Sometimes observation returns ([observation], {})
        # Therefore, if it's tuple take the first element        
        if isinstance(observation, tuple):
            observation_array = observation[0]
        # If not just use the observation as it is
        else:
            observation_array = observation
        print("print observation_array:", observation_array)
        
        observation_array = np.array(observation_array).reshape(1, -1)
        return np.argmax(np.dot(observation_array, self.weights))

    
def evaluate_population(population):
        rewards = []
        for agent in population:
            observation = env.reset()
            total_reward = 0
            for _ in range(200):
                action = agent.predict(observation)
                print("Action:", action)

                observation, reward, terminated, truncated, info = env.step(action)
                print("observation type: ",type(observation))
                total_reward += reward
                if terminated:
                    break
            rewards.append(total_reward)
        return rewards
    
def crossover(parent1, parent2):
        child = NeuralNetwork(parent1.weights.shape[0], parent1.weights.shape[1])
        for i in range (parent1.weights.shape[0]):
            if np.random.rand() < 0.5:
                child.weights[i, :] = parent1.weights[i, :]
            else:
                child.weights[i, :] = parent2.weights[i, :]
        return child
    
def mutate(agent):
        agent.weights += np.random.randn(agent.weights.shape[0], agent.weights.shape[1]) * 0.01
        return agent
    
# Hyperparameter Initialization

POPULATION_SIZE = 100
GENERATIONS = 40
TOP_K = 10
MUTATION_RATE = 0.2

population = [NeuralNetwork(4,2) for _ in range(POPULATION_SIZE)]

for generation in range(GENERATIONS):
    rewards = evaluate_population(population)
    top_agents = np.argsort(rewards)[-TOP_K:]

    new_population = []
    for i in range(POPULATION_SIZE):
        parent1 = population[np.random.choice(top_agents)]
        parent2 = population[np.random.choice(top_agents)]
        child = crossover(parent1, parent2)
        if np.random.rand() < MUTATION_RATE:
            child = mutate(child)
        new_population.append(child)
    
    population = new_population 
    # happens every once in generation, the population changes

best_agent = population[np.argmax(evaluate_population(population))]
observation = env.reset()
total_reward = 0
for _ in range(200):
    env.render()
    # observation is the state from CartPole environment [cart pos, cart velocity, pole angle, and pole velocity at the tip]
    # action [0 or 1]
    action = best_agent.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated:
        break

print(f"Best Agent Reward: {total_reward}")

env.close()

