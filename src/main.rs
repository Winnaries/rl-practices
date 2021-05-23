use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::{
    collections::HashMap,
    io::{self, BufRead},
    ops::{AddAssign, Sub, SubAssign},
};
use std::{env, fs::File, ops::Add};

/// Position of an agent, a tuple of `(x, y)`
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
struct Position {
    x: i32,
    y: i32,
}

impl Add<Velocity> for Position {
    type Output = Position;

    fn add(self, delta: Velocity) -> Position {
        let x = self.x + delta.x;
        let y = self.y + delta.y;

        Position { x, y }
    }
}

impl AddAssign<Velocity> for Position {
    fn add_assign(&mut self, delta: Velocity) {
        self.x += delta.x;
        self.y += delta.y;
    }
}

impl Sub<Position> for Position {
    type Output = Position;

    fn sub(self, delta: Position) -> Position {
        let x = self.x + delta.x;
        let y = self.y + delta.y;

        Position { x, y }
    }
}

impl SubAssign<Position> for Position {
    fn sub_assign(&mut self, delta: Position) {
        self.x -= delta.x;
        self.y -= delta.y;
    }
}

/// Velocity vector of an agent, a tuple of
/// `(V_horizontal, V_vertical)`
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
struct Velocity {
    x: i32,
    y: i32,
}

impl Velocity {
    fn new() -> Self {
        Velocity { x: 0, y: 0 }
    }
}

impl Add<Velocity> for Velocity {
    type Output = Velocity;

    fn add(self, delta: Velocity) -> Velocity {
        let x = self.x + delta.x;
        let y = self.y + delta.y;

        let x = x.min(5).max(0);
        let y = y.max(-5).min(0);

        if x == 0 && y == 0 {
            return self;
        }

        Velocity { x, y }
    }
}

impl AddAssign<Velocity> for Velocity {
    fn add_assign(&mut self, delta: Velocity) {
        let x = self.x + delta.x;
        let y = self.y + delta.y;

        let x = x.min(5).max(0);
        let y = y.max(-5).min(0);

        if x != 0 && y != 0 {
            self.x = x;
            self.y = y;
        }
    }
}

/// A choice of action of 3x3
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

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        Action::from(rng.gen_range(0..9))
    }
}

impl From<Action> for Velocity {
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

        Velocity { x, y }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct Episode {
    states: Vec<(Position, Velocity)>,
    actions: Vec<Action>,
    rewards: Vec<i32>,
}

impl Episode {
    fn new() -> Self {
        Episode {
            states: vec![],
            actions: vec![],
            rewards: vec![],
        }
    }
}

/// Learning agent with position and velocity states
struct Agent {
    gamma: f32,
    position: Position,
    velocity: Velocity,
    environment: Environment,
    policy: HashMap<(Position, Velocity), Action>,
    values: HashMap<(Position, Velocity), [[f32; 3]; 3]>,
    counter: HashMap<(Position, Velocity), [[f32; 3]; 3]>,
}

impl Agent {
    fn new(environment: Environment) -> Self {
        Agent {
            environment,
            gamma: 1.0,
            position: Position { x: 0, y: 0 },
            velocity: Velocity { x: 0, y: 0 },
            policy: HashMap::new(),
            values: HashMap::new(),
            counter: HashMap::new(),
        }
    }

    fn race(&mut self) -> Episode {
        let mut episode = Episode::new();
        let mut reward: Option<i32> = None;

        self.position = self.environment.begin();
        self.velocity = Velocity { x: 0, y: 0 };

        while reward == None || reward != Some(0) {
            episode.states.push((self.position, self.velocity));

            let action = self.policy.get(&(self.position, self.velocity));
            let mut action = match action {
                Some(&x) => x,
                None => rand::random(),
            };

            if self.environment.start.contains(&self.position)
                && self.velocity == Velocity::new()
                && (action != Action::Top || action != Action::TopRight)
            {
                let mut rng = rand::thread_rng();
                let action_idx = rng.gen_range(1..3);
                action = Action::from(action_idx as u8);
            }

            println!("{:?}", (self.position, self.velocity, action));
            episode.actions.push(action);

            let (next_pos, next_vel, next_reward) = self.step(action);

            episode.rewards.push(next_reward);
            reward = Some(next_reward);

            self.position = next_pos;
            self.velocity = next_vel;
        }

        episode
    }

    fn play(&mut self) -> Episode {
        let mut episode = Episode::new();
        let mut reward: Option<i32> = None;

        self.position = self.environment.begin();
        self.velocity = Velocity { x: 0, y: 0 };

        while reward == None || reward != Some(0) {
            episode.states.push((self.position, self.velocity));

            let mut action: Action = rand::random();

            // Prevent from being still at the start!
            if self.environment.start.contains(&self.position)
                && self.velocity == Velocity::new()
                && action == Action::Still
            {
                while action == Action::Still {
                    action = rand::random();
                }
            }

            episode.actions.push(action);

            let (next_pos, next_vel, next_reward) = self.step(action);

            episode.rewards.push(next_reward);
            reward = Some(next_reward);

            self.position = next_pos;
            self.velocity = next_vel;
        }

        episode
    }

    fn learn(&mut self, episode: Episode) {
        let states = episode.states.iter();
        let mut actions = episode.actions;
        let mut rewards = episode.rewards;

        actions.reverse();
        rewards.reverse();

        let mut g: f32 = 0.0;
        let mut w: f32 = 1.0;

        for (time, state) in states.rev().enumerate() {
            let action = actions[time];
            g = self.gamma * g + rewards[time] as f32;

            if !self.counter.contains_key(state) {
                self.counter.insert(*state, [[0.0; 3]; 3]);
            }

            if !self.values.contains_key(state) {
                self.values.insert(*state, [[-1_000.0; 3]; 3]);
            }

            if !self.policy.contains_key(state) {
                self.policy.insert(*state, Action::Still);
            }

            let c = self.counter.get_mut(state).unwrap();
            let q = self.values.get_mut(state).unwrap();
            let pi = self.policy.get_mut(state).unwrap();

            let action_idx = action as usize;

            c[action_idx / 3][action_idx % 3] += w;

            let ratio = w / c[action_idx / 3][action_idx % 3];
            q[action_idx / 3][action_idx % 3] += ratio * (g - q[action_idx / 3][action_idx % 3]);

            let mut argmax: u8 = 0;
            let mut max: f32 = f32::NEG_INFINITY;
            for (i, row) in q.iter().enumerate() {
                for (j, &cell) in row.iter().enumerate() {
                    if cell > max {
                        max = cell;
                        argmax = 3 * i as u8 + j as u8;
                    }
                }
            }

            *pi = Action::from(argmax);

            if Action::from(argmax) != action {
                break;
            }

            w *= 1.0 / (1.0 / 9.0);
        }
    }

    fn step(&mut self, action: Action) -> (Position, Velocity, i32) {
        let delta = Velocity::from(action);
        let next_vel = self.velocity + delta;
        let next_pos = self.position + next_vel;
        let outcome = self.environment.check(next_pos, self.position);

        match outcome {
            Outcome::Finish => (next_pos, next_vel, 0),
            Outcome::Continue => (next_pos, next_vel, -1),
            Outcome::Restart => (self.environment.begin(), Velocity::new(), -1),
        }
    }
}

/// An enum that describe the type of a cell
/// in a race track eg. start cell and finish cell
#[derive(Eq, PartialEq, Clone, Copy)]
enum Track {
    Boundary = 0,
    Road,
    Start,
    Finish,
    Corrupt,
}

#[derive(Eq, PartialEq, Clone, Copy)]
enum Outcome {
    Continue = 0,
    Finish,
    Restart,
}

struct Environment {
    track: Vec<Vec<Track>>,
    start: Vec<Position>,
    finish: Vec<Position>,
}

impl Environment {
    fn from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = io::BufReader::new(file);
        let lines = reader.lines();

        let mut track: Vec<Vec<Track>> = vec![];
        let mut start: Vec<Position> = vec![];
        let mut finish: Vec<Position> = vec![];

        for (y, line) in lines.enumerate() {
            let line = line?;
            let characters = line.split_whitespace();

            track.push(vec![]);

            for (x, c) in characters.enumerate() {
                let cell = match c {
                    "-" => Track::Boundary,
                    "o" => Track::Road,
                    "s" => Track::Start,
                    "f" => Track::Finish,
                    _ => Track::Corrupt,
                };

                if cell == Track::Corrupt {
                    return Err(io::Error::from(io::ErrorKind::InvalidData));
                }

                track[y].push(cell);

                match cell {
                    Track::Start => start.push(Position {
                        x: x as i32,
                        y: y as i32,
                    }),
                    Track::Finish => finish.push(Position {
                        x: x as i32,
                        y: y as i32,
                    }),
                    _ => (),
                }
            }
        }

        Ok(Environment {
            track,
            start,
            finish,
        })
    }

    fn begin(&self) -> Position {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.start.len());
        self.start[idx]
    }

    fn check(&self, dest: Position, curr: Position) -> Outcome {
        if !self.possible(dest, curr) {
            return Outcome::Restart;
        }

        if self.cross(dest, curr) {
            return Outcome::Finish;
        }

        if dest.x < 0
            || dest.x as usize >= self.track[0].len()
            || dest.y < 0
            || dest.y as usize >= self.track.len()
        {
            return Outcome::Restart;
        }

        Outcome::Continue
    }

    fn possible(&self, dest: Position, curr: Position) -> bool {
        if curr.x > dest.x || curr.y < dest.y || curr.x >= self.track[0].len() as i32 || curr.y < 0
        {
            return false;
        }

        if self.track[curr.y as usize][curr.x as usize] == Track::Boundary
            || self.track[curr.y as usize][curr.x as usize] == Track::Corrupt
        {
            return false;
        }

        if dest == curr {
            return true;
        }

        let a = curr + Velocity { x: 0, y: -1 };
        let b = curr + Velocity { x: 1, y: 0 };
        let c = curr + Velocity { x: 1, y: -1 };

        a == dest
            || b == dest
            || c == dest
            || self.possible(dest, a)
            || self.possible(dest, b)
            || self.possible(dest, c)
    }

    fn cross(&self, dest: Position, curr: Position) -> bool {
        if curr.x > dest.x || curr.y < dest.y || curr.x >= self.track[0].len() as i32 || curr.y < 0
        {
            return false;
        }

        if self.finish.contains(&curr) {
            return true;
        }

        let a = curr + Velocity { x: 0, y: -1 };
        let b = curr + Velocity { x: 1, y: 0 };
        let c = curr + Velocity { x: 1, y: -1 };

        self.cross(dest, a) || self.cross(dest, b) || self.cross(dest, c)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let environment = Environment::from_file(&args[1]).unwrap();
    let mut agent = Agent::new(environment);

    for i in 0..=80000 {
        let episode = agent.play();
        agent.learn(episode);

        if i % 100 == 0 {
            println!("\nRound {}", i);

            let record = agent.race();
            let time = record
                .rewards
                .clone()
                .into_iter()
                .reduce(|a, b| a + b)
                .unwrap();

            println!("Time: {:>6}", time);
        }
    }
}
