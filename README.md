# mario_rl

## Environment
Reward Function
### Reward Function

The reward function assumes the objective of the game is to move as far right
as possible (increase the agent's _x_ value), as fast as possible, without
dying. To model this game, three separate variables compose the reward:

1.  _v_: the difference in agent _x_ values between states
    -   in this case this is instantaneous velocity for the given step
    -   _v = x1 - x0_
        -   _x0_ is the x position before the step
        -   _x1_ is the x position after the step
    -   moving right ⇔ _v > 0_
    -   moving left ⇔ _v < 0_
    -   not moving ⇔ _v = 0_
2.  _c_: the difference in the game clock between frames
    -   the penalty prevents the agent from standing still
    -   _c = c0 - c1_
        -   _c0_ is the clock reading before the step
        -   _c1_ is the clock reading after the step
    -   no clock tick ⇔ _c = 0_
    -   clock tick ⇔ _c < 0_
3.  _d_: a death penalty that penalizes the agent for dying in a state
    -   this penalty encourages the agent to avoid death
    -   alive ⇔ _d = 0_
    -   dead ⇔ _d = -15_

_r = v + c + d_

## RL Learning
### DQN
Vaillna DQN (no double, no deuling, no PER)
