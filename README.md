## Practices on RL Algorithms   :couch_and_lamp:

Each of the folder within the project contains a Rust crate that specifically solve simple RL algorithms. 
They require `cargo` to compile and run along with some specific configuration (eg. a map as in racetrack example).

### Racetrack :racing_car: 

`path: /racetrack`

A Monte-carlo Control algorithm for driving a race car around a right-turn, where only the agent can only control the 
velocity increment/decrement at each time step. 

#### Lessons Learned
 - Basic design and practical implementation of learning agent and environment with Rust. 
 - Figuring out which component of the problem is the agent and environment fosters clean code and make implementation easier.
 - Behavior policy can greatly influences the target policy, in the case, the former is uniformly random policy and the latter is a deterministic policy. 
 - Off-policy MC control can be really slow depending on how both policy are chosen. I needed around 80,000 iteration before it converged.
 - Pre-determining several obvious deterministic conditions can prevents infinite loops and misoptimized condition. For example, I found that it is more efficient to make the car take a `forward` action at the beginning. 
 - One surprising encounter was the fact that importance sampling weight can get really large, which may cause numerical overflow. Note that I might be wrong on this.
 
#### What Next?
 - Optionally create an on-policy estimation of optimal policy.
 - Implement discounting-aware importance sampling.
 - Learn per-decision importance sampling and see if it can help.

### Windy Gridworld :wind_chime:

`path: /windy-gridworld`

A Sarsa (on-policy temporal difference control) for estimating the most promising route to travel from `A` to `B` within the grid world, given 9 actions to choose and vertical stochastic wind effects. 

#### Lessons Learned
 - An elegant way to implement action sampling from policy and custom distribution in Rust
 - Epsilon greedy algorithm have a huge effect on how the algorithm perform on the later episodes
 - Efficient implementation of Sarsa algorithm

#### What Next?
 - Try tweaking learning rate and epsilon
 - Plot the learning efficiency with reward data
 - Implement an Expected Sarsa instead of ordinary Sarsa
 - Try Q-learning as opposed to on-policy algorithm
