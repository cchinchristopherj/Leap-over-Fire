Leap over Fire
=========================

AI Beats Game using Deep Q Learning

Advancements in Reinforcement Learning (RL) have allowed AI to learn how to perform complex actions and behaviors, such as famously beating a human player at the game of Go, solely based on reward signals from their environment instead of annotated input/output pairs as in supervised learning. In this game, an AI will use one of the most popular algorithms from RL, Deep Q Learning, to learn how to safely shoot a ball over a fire. 

Web Application
=========================

Starting Screen:

![humanplayer](https://github.com/cchinchristopherj/Leap-over-Fire/blob/cchinchristopherj-patch-1/Images/humanplayer.png)

During the "Player: You" mode of the game, the fire will change to a different position for every session of the game and it is the user's task to find an optimal configuration of the sliders (controlling the strength and direction of the shot), that safely shoots the ball to the other side. Most likely, after getting a feel for the sliders' effect on the force of the shot, the user will change the setting of the sliders according to the current direction of the fire. 

Interestingly, an AI agent will not take this approach. The "Let AI Play" button can be pressed to switch to "Player: AI" mode, during which the AI will set the positions of the sliders based on its growing set of learned experiences.

![aiplayer](https://github.com/cchinchristopherj/Leap-over-Fire/blob/cchinchristopherj-patch-1/Images/aiplayer.png)

Deep Reinforcement Learning
=========================

Classically, Deep Q Learning is implemented via Q-tables, which have as many rows as possible states and as many columns as possible actions, with the value of each cell being the maximum expected future reward (Q-value) for the given state and action. As the agent performs actions and receives rewards, these Q-values are updated using the Bellman Equation. New Q-values are a function of the received reward for the chosen action and the maximum expected future reward given all possible actions in the new state. 

![bellmanequation](https://github.com/cchinchristopherj/Leap-over-Fire/blob/cchinchristopherj-patch-1/bellmanequation.png)

*Bellman Equation. Image Source: [Diving Deeper into Reinforcement Learning with Q-Learning](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe)
    
For this game, classical Q-tables will be replaced by deep learning: the AI agent will be represented by a shallow multi-layer perceptron, which receives as input variables describing the spatial extent of the fire (a human would likewise be able to see where the fire is located on the screen). The output of the neural network is the approximate Q-value of each action for each state, with there being sixteen possible actions (sixteen possible quantized configurations of the sliders controlling the rightward and upward force exerted on the ball). 

![deeplearning_qvalues](https://github.com/cchinchristopherj/Leap-over-Fire/blob/cchinchristopherj-patch-1/deeplearning_qvalues.png)

*Deep Learning Replaces Q-Tables. Image Source: [An introduction to Deep Q-Learning: let's play Doom](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)

The action corresponding to the highest Q-value is the one chosen to be executed and a reward is received if the ball makes it safely to the other side. 

It is important to note that, at the beginning, the AI agent has no knowledge of the environment or state space of possible actions. The exploration-exploitation tradeoff concept from RL comes in handy to enable the AI to learn initially through random "exploration" and settle eventually on "exploitation" of its cumulative knowledge to choose the best actions. 

![exploration_exploitation](https://github.com/cchinchristopherj/Leap-over-Fire/blob/cchinchristopherj-patch-1/exploration_exploitation.png)

*Exploration-Exploitation Tradeoff. Image Source: [Diving Deeper into Reinforcement Learning with Q-Learning](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe)

A global value called "epsilon" determines which of these two strategies should be adopted. (If a random number between 0 and 1 is less than the value of "epsilon", an exploration strategy is taken, i.e. a randomly chosen action. On the other hand, if a random number between 0 and 1 is greater than the value of "epsilon," an exploitation strategy is taken, i.e. an action based on prior experience. "Epsilon" is initialized to 1 and decreased gradually with every session of the game so that exploitation is adopted more often than exploration over time.
