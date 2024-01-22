use std::collections::{BinaryHeap, HashSet};
use std::hash::{Hash, Hasher};
use whitebox_raster::Raster;

pub enum Neighbor {
    UpLeft,
    Up,
    UpRight,
    Right,
    DownRight,
    Down,
    DownLeft,
    Left,
}

#[derive(Clone, Debug)]
pub struct DEMCell {
    pub row: isize,
    pub column: isize,
    pub elevation: f64,
}

impl DEMCell {
    pub fn from_raster(row: isize, column: isize, raster: &Raster) -> DEMCell {
        DEMCell {
            row,
            column,
            elevation: raster.get_value(row, column),
        }
    }

    pub fn get_neighbor(&self, neighbor: Neighbor, raster: &Raster) -> DEMCell {
        match neighbor {
            Neighbor::UpLeft => DEMCell::from_raster(self.row - 1, self.column - 1, raster),
            Neighbor::Up => DEMCell::from_raster(self.row - 1, self.column, raster),
            Neighbor::UpRight => DEMCell::from_raster(self.row - 1, self.column + 1, raster),
            Neighbor::Right => DEMCell::from_raster(self.row, self.column + 1, raster),
            Neighbor::DownRight => DEMCell::from_raster(self.row + 1, self.column + 1, raster),
            Neighbor::Down => DEMCell::from_raster(self.row + 1, self.column, raster),
            Neighbor::DownLeft => DEMCell::from_raster(self.row + 1, self.column - 1, raster),
            Neighbor::Left => DEMCell::from_raster(self.row, self.column - 1, raster),
        }
    }

    pub fn get_neighbors(&self, raster: &Raster) -> Vec<DEMCell> {
        vec![
            self.get_neighbor(Neighbor::UpLeft, raster),
            self.get_neighbor(Neighbor::Up, raster),
            self.get_neighbor(Neighbor::UpRight, raster),
            self.get_neighbor(Neighbor::Right, raster),
            self.get_neighbor(Neighbor::DownRight, raster),
            self.get_neighbor(Neighbor::Down, raster),
            self.get_neighbor(Neighbor::DownLeft, raster),
            self.get_neighbor(Neighbor::Left, raster),
        ]
    }
}

impl PartialOrd for DEMCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.elevation.partial_cmp(&other.elevation)
    }
}

impl Ord for DEMCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for DEMCell {}

impl PartialEq for DEMCell {
    fn eq(&self, other: &DEMCell) -> bool {
        self.row == other.row && self.column == other.column
    }
}

impl Hash for DEMCell {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.row.hash(state);
        self.column.hash(state);
    }
}

pub enum DEMCellType {
    Flowable(DEMCell),
    Pit(DEMCell),
    NoData(DEMCell),
}

impl DEMCellType {
    pub fn from_raster(row: isize, column: isize, raster: &Raster) -> DEMCellType {
        let cell = DEMCell::from_raster(row, column, raster);
        if cell.elevation == raster.configs.nodata {
            DEMCellType::NoData(cell.clone());
        }
        for neighbor in cell.get_neighbors(raster) {
            if neighbor.elevation == raster.configs.nodata {
                return DEMCellType::Flowable(cell.clone());
            } else if neighbor.elevation < cell.elevation {
                return DEMCellType::Flowable(cell.clone());
            }
        }
        return DEMCellType::Pit(cell);
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum PitStatus {
    Unseen(DEMCell),
    Solved(DEMCell),
    Unsolved(DEMCell),
}

/// Manages the order and status of pit breaches.
struct PitResolutionTracker {
    unseen: Vec<PitStatus>,
    in_progress: HashSet<PitStatus>,
    breached: HashSet<PitStatus>,
    not_breached: HashSet<PitStatus>,
}

impl PitResolutionTracker {
    fn new(sorted_pits: BinaryHeap<PitStatus>) -> PitResolutionTracker {
        PitResolutionTracker {
            unseen: sorted_pits.into_sorted_vec(),
            in_progress: HashSet::new(),
            breached: HashSet::new(),
            not_breached: HashSet::new(),
        }
    }

    /// Returns an unseen pit with the lowest elevation that is not within a specified
    /// distance of any pits in progress.
    #[allow(unused_variables)]
    fn get_next_unseen(&mut self, min_distance: f64) -> Option<PitStatus> {
        todo!();
        // loop over the unseen pits in reverse order so that the call to remove()
        // does not need to shift many elements.
    }
}
