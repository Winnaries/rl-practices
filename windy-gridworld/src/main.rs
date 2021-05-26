use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::ops::Add;

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
struct Position {
    x: i32,
    y: i32,
}

impl Add<Position> for Position {
    type Output = Position;

    fn add(self, rhs: Position) -> Self::Output {
        Position {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
enum Action {
    TopLeft = 0,
    Top,
    TopRight,
    Left,
    Still,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl From<u8> for Action {
    fn from(idx: u8) -> Self {
        use Action::*;
        match idx {
            0 => TopLeft,
            1 => Top,
            2 => TopRight,
            3 => Left,
            5 => Right,
            6 => BottomLeft,
            7 => Bottom,
            8 => BottomRight,
            _ => Still,
        }
    }
}

impl From<Action> for Position {
    fn from(action: Action) -> Self {
        use Action::*;

        let (x, y) = match action {
            Top => (0, -1),
            Right => (1, 0),
            Bottom => (0, 1),
            Left => (-1, 0),
            TopLeft => (-1, -1),
            TopRight => (1, -1),
            BottomRight => (1, 1),
            BottomLeft => (-1, 1),
            Still => (0, 0),
        };

        Position { x, y }
    }
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        Action::from(rng.gen_range(0..9))
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct Reward(i32);

impl Reward {
    fn value(self) -> i32 {
        return self.0;
    }
}

struct Map<const W: usize, const H: usize> {
    pos: Position,
    goal: Position,
    start: Position,
    wind: [i32; W],
    stochastic: bool,
}

impl<const W: usize, const H: usize> Map<W, H> {
    const ORIG: Position = Position { x: 0, y: 0 };

    const EDGE: Position = Position {
        x: W as i32 - 1,
        y: H as i32 - 1,
    };

    fn new(start: Position, goal: Position, wind: [i32; W], stochastic: bool) -> Self {
        Map {
            wind,
            goal,
            start,
            stochastic,
            pos: start,
        }
    }

    fn begin(&mut self) -> Position {
        self.pos = self.start;
        self.start
    }

    fn finish(&self) -> bool {
        self.pos == self.goal
    }

    fn take(&mut self, action: Action) -> (Position, Reward) {
        let delta = Position::from(action);
        let mut new_pos = self.pos + delta;
        let mut rng = rand::thread_rng();
        let wind_spike = if self.stochastic {
            rng.gen_range(-1..2)
        } else {
            0
        };

        if new_pos.x < Self::ORIG.x
            || new_pos.y < Self::ORIG.y
            || new_pos.x > Self::EDGE.x
            || new_pos.y > Self::EDGE.y
        {
            (self.pos, Reward(-1))
        } else {
            new_pos.y = (new_pos.y - self.wind[self.pos.x as usize] + wind_spike)
                .max(Self::ORIG.y)
                .min(Self::EDGE.y);
            self.pos = new_pos;

            if new_pos == self.goal {
                (new_pos, Reward(0))
            } else {
                (new_pos, Reward(-1))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct EpsilonGreedy<T> {
    action: T,
    epsilon: f64,
}

impl EpsilonGreedy<Action> {
    const RANGE: u32 = 9;

    fn new(epsilon: f64) -> Self {
        EpsilonGreedy {
            epsilon,
            action: rand::random(),
        }
    }

    fn prefer(&mut self, action: Action) {
        self.action = action;
    }

    fn reroll(&mut self) {
        self.action = rand::random();
    }
}

impl Distribution<Action> for EpsilonGreedy<Action> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        let explore = rng.gen_bool(self.epsilon);
        if explore {
            let idx = rng.gen_range(0..Self::RANGE);
            Action::from(idx as u8)
        } else {
            self.action
        }
    }
}

struct Agent<const W: usize, const H: usize> {
    map: Map<W, H>,
    alpha: f64,
    gamma: f64,
    values: [[[f64; 9]; W]; H],
    policy: [[EpsilonGreedy<Action>; W]; H],
}

impl<const W: usize, const H: usize> Agent<W, H> {
    fn new(map: Map<W, H>, exp_rate: f64, alpha: f64, gamma: f64) -> Self {
        let mut agent = Agent {
            map,
            alpha,
            gamma,
            values: [[[0.0; 9]; W]; H],
            policy: [[EpsilonGreedy::new(exp_rate); W]; H],
        };

        for i in agent.policy.as_mut() {
            for j in i {
                j.reroll();
            }
        }

        agent
    }

    fn play(&mut self) -> Reward {
        self.map.begin();

        let mut reward = 0;
        let mut rng = rand::thread_rng();
        let mut action = rng.sample(self.policy[self.map.pos.y as usize][self.map.pos.x as usize]);

        while !self.map.finish() {
            let pos = self.map.pos;
            let value = self.values[pos.y as usize][pos.x as usize][action as usize];

            let (next_pos, next_reward) = self.map.take(action);
            let next_action = rng.sample(self.policy[next_pos.y as usize][next_pos.x as usize]);
            let next_value =
                self.values[next_pos.y as usize][next_pos.x as usize][next_action as usize];

            self.values[pos.y as usize][pos.x as usize][action as usize] +=
                self.alpha * (next_reward.value() as f64 + self.gamma * next_value - value);

            let mut max = f64::NEG_INFINITY;
            let mut argmax = Action::TopLeft;
            for (i, &value) in self.values[pos.y as usize][pos.x as usize]
                .iter()
                .enumerate()
            {
                if value > max {
                    max = value;
                    argmax = Action::from(i as u8);
                }
            }

            self.policy[pos.y as usize][pos.x as usize].prefer(argmax);
            action = next_action;
            reward += next_reward.value();
        }

        Reward(reward)
    }
}

fn main() {
    let start = Position { x: 0, y: 3 };
    let goal = Position { x: 7, y: 3 };

    let wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0];
    let map = Map::<10, 7>::new(start, goal, wind, true);

    let mut agent = Agent::new(map, 0.1, 0.5, 1.0);
    let mut total_reward = 0;

    println!("Episode | Total Reward | Episode Reward");
    for i in 0..=1000 {
        let reward = agent.play();
        total_reward += reward.value();
        println!("{:>7} | {:>12} | {:14}", i, -total_reward, -reward.value());
    }
}
