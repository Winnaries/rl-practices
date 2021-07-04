use crate::{Action, Agent, EpsilonGreedy, Position, Reward, World};
use rand::{self, Rng};
use std::collections::{HashMap, HashSet};

pub struct DynaQPlus {
    pub world: World,
    pub alpha: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub planning: u32,
    pub _epsilon: f64,
    step: u32,
    latest: HashMap<(Position, Action), u32>,
    history: HashMap<Position, HashSet<Action>>,
    values: Vec<Vec<[f64; 4]>>,
    model: Vec<Vec<[(Reward, Position); 4]>>,
    pub policy: Vec<Vec<EpsilonGreedy<Action>>>,
}

impl DynaQPlus {
    pub fn new(
        world: World,
        planning: u32,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        kappa: f64,
    ) -> Self {
        let mut values = vec![];
        let mut model = vec![];
        let mut policy = vec![];
        let history = HashMap::new();
        let latest = HashMap::new();
        let step = 0;

        for i in 0..world.height {
            values.push(vec![]);
            model.push(vec![]);
            policy.push(vec![]);

            for _ in 0..world.width {
                values[i as usize].push([0.0; 4]);
                model[i as usize].push([(Reward(0.0), Position::new()); 4]);
                policy[i as usize].push(EpsilonGreedy::new(epsilon));
            }
        }

        Self {
            step,
            world,
            alpha,
            gamma,
            kappa,
            values,
            model,
            latest,
            policy,
            history,
            planning,
            _epsilon: epsilon,
        }
    }

    #[allow(non_snake_case)]
    fn plan(&mut self, step: u32) {
        let mut rng = rand::thread_rng();
        let history: Vec<&Position> = self.history.keys().collect();
        for _ in 0..step {
            let idx = rng.gen_range(0..history.len());
            let &pos = history[idx];
            let actions: Vec<&Action> = self.history.get(&pos).unwrap().iter().collect();

            let idx = rng.gen_range(0..actions.len());
            let &action = actions[idx];

            let (reward, next_pos) = self.model[pos.y as usize][pos.x as usize][action as usize];

            let Q_prime = self.values[next_pos.y as usize][next_pos.x as usize];
            let Q = self.values[pos.y as usize][pos.x as usize].as_mut();
            let idx = action as usize;

            let mut max = f64::NEG_INFINITY;
            for &v in Q_prime.iter() {
                max = if v > max { v } else { max };
            }

            let since = *self.latest.get(&(pos, action)).unwrap() as f64;
            let extra = self.kappa * since.sqrt();
            Q[idx] += self.alpha * (reward.value() + extra + self.gamma * max - Q[idx]);

            let mut argmax = 0;
            for (i, &v) in Q.iter().enumerate() {
                argmax = if v > Q[argmax] { i as usize } else { argmax };
            }

            self.policy[pos.y as usize][pos.x as usize].prefer(Action::from(argmax as u8));
        }
    }
}

impl Agent for DynaQPlus {
    #[allow(non_snake_case)]
    fn step(&mut self) -> Reward {
        let pos = self.world.pos;
        let mut rng = rand::thread_rng();
        let action = rng.sample(self.policy[pos.y as usize][pos.x as usize]);
        let (reward, next_pos) = self.world.take(action);
        self.step += 1;

        if let Some(experience) = self.history.get_mut(&pos) {
            experience.insert(action);
        } else {
            let mut actions = HashSet::new();
            actions.insert(action);
            self.history.insert(pos, actions);
        }

        if let Some(step) = self.latest.get_mut(&(pos, action)) {
            *step = self.step;
        } else {
            self.latest.insert((pos, action), self.step);
        }

        let Q_prime = self.values[next_pos.y as usize][next_pos.x as usize];
        let Q = self.values[pos.y as usize][pos.x as usize].as_mut();
        let M = self.model[pos.y as usize][pos.x as usize].as_mut();
        let idx = action as usize;

        let mut max = f64::NEG_INFINITY;
        for &v in Q_prime.iter() {
            max = if v > max { v } else { max };
        }

        Q[idx] += self.alpha * (reward.value() + self.gamma * max - Q[idx]);
        M[idx] = (reward, next_pos);

        let mut argmax = 0;
        for (i, &v) in Q.iter().enumerate() {
            argmax = if v > Q[argmax] { i as usize } else { argmax };
        }

        self.policy[pos.y as usize][pos.x as usize].prefer(Action::from(argmax as u8));

        self.plan(self.planning);

        reward
    }
}
