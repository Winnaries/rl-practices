use crate::{Action, Agent, EpsilonGreedy, Reward, World};
use rand::Rng;

pub struct PureTD {
    pub world: World,
    pub alpha: f64,
    pub gamma: f64,
    pub _epsilon: f64,
    values: Vec<Vec<[f64; 4]>>,
    pub policy: Vec<Vec<EpsilonGreedy<Action>>>,
}

impl PureTD {
    pub fn new(world: World, alpha: f64, gamma: f64, epsilon: f64) -> Self {
        let mut values = vec![];
        let mut policy = vec![];

        for i in 0..world.height {
            values.push(vec![]);
            policy.push(vec![]);

            for _ in 0..world.width {
                values[i as usize].push([0.0; 4]);
                policy[i as usize].push(EpsilonGreedy::new(epsilon));
            }
        }

        Self {
            world,
            alpha,
            gamma,
            values,
            policy,
            _epsilon: epsilon,
        }
    }
}

impl Agent for PureTD {
    #[allow(non_snake_case)]
    fn step(&mut self) -> Reward {
        let pos = self.world.pos;
        let mut rng = rand::thread_rng();
        let action = rng.sample(self.policy[pos.y as usize][pos.x as usize]);
        let (reward, next_pos) = self.world.take(action);

        let Q_prime = self.values[next_pos.y as usize][next_pos.x as usize];
        let Q = self.values[pos.y as usize][pos.x as usize].as_mut();
        let idx = action as usize;

        let mut max = f64::NEG_INFINITY;
        for &v in Q_prime.iter() {
            max = if v > max { v } else { max };
        }

        Q[idx] += self.alpha * (reward.value() + self.gamma * max - Q[idx]);

        let mut argmax = 0;
        for (i, &v) in Q.iter().enumerate() {
            argmax = if v > Q[argmax] { i as usize } else { argmax };
        }

        self.policy[pos.y as usize][pos.x as usize].prefer(Action::from(argmax as u8));

        reward
    }
}
