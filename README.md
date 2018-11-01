# mario_rl

## Environment
Reward Function
The reward function assumes the objective of the game is to move as far right as possible (increase the agent's x value), as fast as possible, without dying. To model this game, three separate variables compose the reward:

v: the difference in agent x values between states
in this case this is instantaneous velocity for the given step
v = x1 - x0
x0 is the x position before the step
x1 is the x position after the step
moving right ⇔ v > 0
moving left ⇔ v < 0
not moving ⇔ v = 0
c: the difference in the game clock between frames
the penalty prevents the agent from standing still
c = c0 - c1
c0 is the clock reading before the step
c1 is the clock reading after the step
no clock tick ⇔ c = 0
clock tick ⇔ c < 0
d: a death penalty that penalizes the agent for dying in a state
this penalty encourages the agent to avoid death
alive ⇔ d = 0
dead ⇔ d = -15

## RL Learning
### DQN
Vaillna DQN (no double, no deuling, no PER)
