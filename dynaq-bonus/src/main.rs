use dynaq::DynaQ;
use dynaq_plus::DynaQPlus;
use plotters::prelude::*;
use pure_td::PureTD;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::env;
use std::io::{self, BufRead};
use std::iter::Iterator;
use std::{fs::File, ops::Add};

mod dynaq;
mod dynaq_plus;
mod pure_td;

#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
pub struct Position {
    x: i32,
    y: i32,
}

impl Position {
    pub fn new() -> Self {
        Position { x: 0, y: 0 }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct Reward(f64);

impl Reward {
    fn value(&self) -> f64 {
        self.0
    }
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

#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
pub enum Action {
    Top = 0,
    Left,
    Right,
    Bottom,
}

impl From<u8> for Action {
    fn from(idx: u8) -> Self {
        use Action::*;
        match idx % 4 {
            0 => Top,
            1 => Left,
            2 => Right,
            3 => Bottom,
            _ => Self::from(idx % 4),
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
        };

        Position { x, y }
    }
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        Action::from(rng.gen_range(0..4))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Cell {
    Normal = 0,
    Start,
    Goal,
    Block,
    Corrupt,
}

#[derive(Clone, Debug)]
pub struct World {
    width: i32,
    height: i32,
    pos: Position,
    start: Position,
    grid: Vec<Vec<Cell>>,
}

impl World {
    pub fn from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = io::BufReader::new(file);
        let lines = reader.lines();

        let mut grid: Vec<Vec<Cell>> = vec![];
        let mut start: Option<Position> = None;

        for (y, line) in lines.enumerate() {
            let line = line?;
            let characters = line.split_whitespace();

            grid.push(vec![]);

            for (x, c) in characters.enumerate() {
                use Cell::*;
                let cell = match c {
                    "-" => Normal,
                    "s" => Start,
                    "g" => Goal,
                    "#" => Block,
                    _ => Corrupt,
                };

                if cell == Corrupt {
                    return Err(io::Error::from(io::ErrorKind::InvalidData));
                }

                grid[y].push(cell);

                if cell == Start {
                    start = Some(Position {
                        x: x as i32,
                        y: y as i32,
                    })
                }
            }
        }

        if start == None {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }

        let start = start.unwrap();

        Ok(World {
            width: grid[0].len() as i32,
            height: grid.len() as i32,
            pos: start,
            start,
            grid,
        })
    }

    pub fn take(&mut self, action: Action) -> (Reward, Position) {
        let delta = Position::from(action);
        let next_pos = self.pos + delta;

        if next_pos.y < 0 || next_pos.x < 0 || next_pos.y >= self.height || next_pos.x >= self.width
        {
            return (Reward(0.0), self.pos);
        }

        use Cell::*;
        let next_cell = self.grid[next_pos.y as usize][next_pos.x as usize];
        let next_state = match next_cell {
            Goal => (Reward(1.0), self.start),
            Block => (Reward(0.0), self.pos),
            _ => (Reward(0.0), next_pos),
        };

        self.pos = next_state.1;
        next_state
    }
}

trait Agent {
    fn step(&mut self) -> Reward;
}

#[derive(Clone, Copy)]
pub struct EpsilonGreedy<T> {
    action: T,
    epsilon: f64,
}

impl EpsilonGreedy<Action> {
    const RANGE: u32 = 4;

    fn new(epsilon: f64) -> Self {
        EpsilonGreedy {
            epsilon,
            action: rand::random(),
        }
    }

    fn prefer(&mut self, action: Action) {
        self.action = action;
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

fn plot(history: Vec<Vec<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let length = history[0].len() as i32;
    let range = (0..length).map(|x| x as f32);
    let max = history
        .iter()
        .map(|x| x.iter().sum::<f64>() as i32)
        .max()
        .unwrap_or(1_000);

    let data = history.iter().map(|rewards| {
        rewards.into_iter().scan(0.0, |state, x| {
            *state += x;
            Some(*state as f32)
        })
    });

    let root = BitMapBackend::new("dynaq-bonus/plots/result.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.margin(40, 40, 40, 40);
    let mut chart = ChartBuilder::on(&root)
        .caption("Pure TD Agent", ("sans-serif", 18).into_font())
        .x_label_area_size(60)
        .y_label_area_size(120)
        .build_cartesian_2d(0f32..length as f32, 0f32..max as f32 * 1.1)?;

    chart.configure_mesh().x_labels(5).y_labels(5).draw()?;

    let colors = [RED, GREEN, BLUE];
    for (i, x) in data.enumerate() {
        chart.draw_series(LineSeries::new(range.clone().zip(x), &colors[i % 3]))?;
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let world_a = World::from_file(&args[1]).unwrap();
    let world_b = World::from_file(&args[2]).unwrap();

    let mut reward_history = vec![];
    let mut td_agent = PureTD::new(world_a.clone(), 0.1, 0.95, 0.1);
    let mut dynaq_agent = DynaQ::new(world_a.clone(), 5, 0.1, 0.95, 0.1);
    let mut dynaq_plus_agent = DynaQPlus::new(world_a.clone(), 5, 0.1, 0.95, 0.1, 1e-4);

    for _ in 0..3 {
        reward_history.push(vec![]);
    }

    for i in 1..=50000 {
        reward_history[0].push(td_agent.step().value());
        reward_history[1].push(dynaq_agent.step().value());
        reward_history[2].push(dynaq_plus_agent.step().value());

        if i == 10000 {
            td_agent.world.grid = world_b.grid.clone();
            dynaq_agent.world.grid = world_b.grid.clone();
            dynaq_plus_agent.world.grid = world_b.grid.clone();
        }
    }

    plot(reward_history).unwrap();
}
