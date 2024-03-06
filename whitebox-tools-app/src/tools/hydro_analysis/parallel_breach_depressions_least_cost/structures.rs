use std::collections::HashSet;

use anyhow::{anyhow, Result};
use ordered_float::OrderedFloat;
use whitebox_raster::Raster;

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct CellData {
    pub index: (usize, usize),
    pub value: OrderedFloat<f64>,
}

impl CellData {
    fn with_new_value(self: Self, new_value: OrderedFloat<f64>) -> Self {
        Self {
            index: self.index,
            value: new_value,
        }
    }

    pub fn distance(self: &Self, other: &CellData) -> OrderedFloat<f64> {
        let (row1, column1) = self.index;
        let (row2, column2) = other.index;
        let square_dist = ((row2 as isize - row1 as isize).pow(2)
            + (column2 as isize - column1 as isize).pow(2)) as f64;
        OrderedFloat(square_dist.sqrt())
    }
}

pub enum InitialState {
    NoData,
    Flowable,
    RawPit,
}

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum CellState {
    NoData(CellData),
    Flowable(CellData),
    RawPit(CellData),
    RaisedPit(CellData),
    UnsolvedPit(CellData),
}

#[derive(Debug)]
pub enum CellTransition {
    RaisePit(OrderedFloat<f64>),
    Breach(OrderedFloat<f64>),
    MarkUnsolved,
}

impl CellState {
    pub fn new(
        initial_state: InitialState,
        index: (usize, usize),
        value: OrderedFloat<f64>,
    ) -> CellState {
        match initial_state {
            InitialState::NoData => Self::NoData(CellData { index, value }),
            InitialState::Flowable => Self::Flowable(CellData { index, value }),
            InitialState::RawPit => Self::RawPit(CellData { index, value }),
        }
    }

    pub fn from_raster(raster: &Raster, row: isize, column: isize) -> CellState {
        let nodata = OrderedFloat(raster.configs.nodata);
        let index = (row as usize, column as usize);
        let value = OrderedFloat(raster.get_value(row, column));

        // Check if the CellState is NoData
        if value == nodata {
            return CellState::new(InitialState::NoData, index, value);
        }

        // Check if the CellState is Flowable. That is, at least one neighbor has a value
        // that is equal to NoData or lower than the central cell value
        let neighbor_positions = [
            (row - 1, column + 0), // Up
            (row - 1, column + 1), // Up-Right
            (row + 0, column + 1), // Right
            (row + 1, column + 1), // Down-Right
            (row + 1, column + 0), // Down
            (row + 1, column - 1), // Down-Left
            (row + 0, column - 1), // Left
            (row - 1, column - 1), // Up-Left
        ];
        let neighbor_values: Vec<OrderedFloat<f64>> = neighbor_positions
            .iter()
            .map(|pos| OrderedFloat(raster.get_value(pos.0, pos.1)))
            .collect();
        for neighbor_value in neighbor_values {
            if neighbor_value == nodata || neighbor_value < value {
                return CellState::new(InitialState::Flowable, index, value);
            }
        }

        // If the CellState is neither Flowable nor NoData, it must be a RawPit
        return CellState::new(InitialState::RawPit, index, value);
    }

    pub fn transition(self: Self, transition: CellTransition) -> Result<CellState> {
        match (self, transition) {
            // NoData is immutable; return the same state and data
            (CellState::NoData(data), _) => Ok(CellState::NoData(data)),

            // The only valid transition starting from Flowable is Breach
            (CellState::Flowable(data), CellTransition::Breach(new_value)) => {
                Ok(CellState::Flowable(data.with_new_value(new_value)))
            }
            (CellState::Flowable(_), transition) => {
                Err(anyhow!("Valid transitions from Flowable is Breach. Received transition: {:?}", transition))
            },

            // The only valid transition starting from RawPit is RaisePit
            (CellState::RawPit(data), CellTransition::RaisePit(new_value)) => {
                Ok(CellState::RaisedPit(data.with_new_value(new_value)))
            }
            (CellState::RawPit(_), transition) => {
                Err(anyhow!("Valid transitions from RawPit is RaisePit. Received transition: {:?}", transition))
            },

            // The only valid transitions starting from RaisedPit are Breach and MarkUnsolved
            (CellState::RaisedPit(data), CellTransition::Breach(new_value)) => {
                Ok(CellState::Flowable(data.with_new_value(new_value)))
            }
            (CellState::RaisedPit(data), CellTransition::MarkUnsolved) => {
                Ok(CellState::UnsolvedPit(data))
            }
            (CellState::RaisedPit(_), transition) => {
                Err(anyhow!("Valid transitions from RaisedPit are Breach and MarkUnsolved. Received transition: {:?}", transition))
            },

            // There are no valid transitions starting from NoData
            (CellState::UnsolvedPit(_), transition) => {
                Err(anyhow!("There are no valid transitions from UnsolvedPit. Received transition: {:?}", transition))
            },
        }
    }

    pub fn get_data(self: &Self) -> &CellData {
        match self {
            CellState::NoData(data) => data,
            CellState::Flowable(data) => data,
            CellState::RawPit(data) => data,
            CellState::RaisedPit(data) => data,
            CellState::UnsolvedPit(data) => data,
        }
    }

    pub fn get_value(self: &Self) -> OrderedFloat<f64> {
        self.get_data().value
    }

    pub fn get_index(self: &Self) -> (usize, usize) {
        self.get_data().index
    }

    pub fn distance(self: &Self, other: &CellState) -> OrderedFloat<f64> {
        self.get_data().distance(other.get_data())
    }
}

pub struct PitResolutionTracker {
    pub unseen: Vec<CellState>,
    pub in_progress: HashSet<CellState>,
    pub solved: HashSet<CellState>,
    pub unsolved: HashSet<CellState>,
}

impl PitResolutionTracker {
    /// Create a new instance of PitResolutionTracker with unseen pits ordered from highest
    /// to lowest value
    fn new(raised_pits: Vec<CellState>) -> PitResolutionTracker {
        let mut sorted = raised_pits.clone();
        sorted.sort_by(|a, b| b.get_value().cmp(&a.get_value()));

        PitResolutionTracker {
            unseen: sorted,
            in_progress: HashSet::new(),
            solved: HashSet::new(),
            unsolved: HashSet::new(),
        }
    }

    fn get_next_available_pit(self: &mut Self, max_dist: isize) -> Option<CellState> {
        // No more unseen pits to solve
        if self.unseen.is_empty() {
            return None;
        }

        // If no pits are in progress, return the lowest value pit without preforming distance calcs
        if self.in_progress.is_empty() {
            let pit = self.unseen.pop()?;
            self.in_progress.insert(pit.clone());
            return Some(pit);
        }

        // Find the lowest value pit that is at least two times the max distance from any pit
        // currently in progress
        let min_dist_between = OrderedFloat((2 * max_dist) as f64);
        for (i, pit) in self.unseen.iter().enumerate().rev() {
            for in_progress_pit in self.in_progress.iter() {
                if pit.distance(in_progress_pit) < min_dist_between {
                    continue;
                }
            }
            let pit = self.unseen.remove(i);
            self.in_progress.insert(pit.clone());
            return Some(pit);
        }
        return None;
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use whitebox_raster::{Raster, RasterConfigs};

    use super::{CellData, CellState, CellTransition, InitialState, PitResolutionTracker};

    #[test]
    fn internal_state_access() {
        let state = CellState::Flowable(CellData {
            index: (1, 1),
            value: OrderedFloat(20.2),
        });
        let data = match state {
            CellState::Flowable(data) => data,
            _ => panic!("Unreachable"),
        };
        assert_eq!(data.index, (1, 1));
        assert_eq!(data.value, OrderedFloat(20.2));
    }
    #[test]
    fn new() {
        let state = CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(20.2));
        assert!(matches!(state, CellState::RawPit(..)));
    }

    fn get_mock_raster() -> Raster {
        let configs: RasterConfigs = RasterConfigs {
            rows: 3,
            columns: 3,
            nodata: -9999.0,
            ..Default::default()
        };
        let mut raster = Raster::initialize_using_config("test.tif", &configs);
        raster.set_row_data(0, vec![5.1, 5.2, 5.3]);
        raster.set_row_data(1, vec![6.1, 1.0, 6.3]);
        raster.set_row_data(2, vec![7.1, 7.2, 7.3]);

        raster
    }

    #[test]
    fn from_raster() {
        let raster = get_mock_raster();

        // The edge is Flowable because it has NoData neighbors
        assert_eq!(
            CellState::new(InitialState::Flowable, (0, 0), OrderedFloat(5.1)),
            CellState::from_raster(&raster, 0, 0)
        );
        // The center is RawPit because all neighbors are higher
        assert_eq!(
            CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(1.0)),
            CellState::from_raster(&raster, 1, 1)
        );
        // Cells out of range are NoData
        assert_eq!(
            CellState::new(InitialState::NoData, (5, 5), OrderedFloat(-9999.0)),
            CellState::from_raster(&raster, 5, 5)
        );
    }

    #[test]
    fn raise_pit_transition() {
        let state = CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(20.2));
        let new_value = OrderedFloat(42.1);
        let raise_pit = CellTransition::RaisePit(new_value);
        let new_state = state.transition(raise_pit);
        if let Ok(CellState::RaisedPit(data)) = new_state {
            assert_eq!(data.value, new_value);
        } else {
            panic!("Unreachable")
        }
    }

    #[test]
    fn no_data_is_immutable() {
        let state = CellState::new(InitialState::NoData, (1, 1), OrderedFloat(-9999.0));
        let new_state = state
            .clone()
            .transition(CellTransition::Breach(OrderedFloat(100.0)))
            .unwrap();
        assert_eq!(state, new_state);
    }

    #[test]
    fn get_data() {
        let state = CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(20.2));
        assert_eq!(state.get_data().index, (1, 1));
        assert_eq!(state.get_data().value, OrderedFloat(20.2));
    }

    #[test]
    fn distance() {
        let state = CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(20.2));
        let other = CellState::new(InitialState::Flowable, (2, 2), OrderedFloat(42.1));
        assert_eq!(state.distance(&other), OrderedFloat(2.0f64.sqrt()));

        let other = CellState::new(InitialState::Flowable, (4, 5), OrderedFloat(42.1));
        assert_eq!(state.distance(&other), OrderedFloat(5.0f64));
    }

    #[test]
    fn new_pit_resolution_tracker_sorts_unseen_highest_to_lowest() {
        let raised_pits = vec![
            CellState::RaisedPit(CellData {
                index: (1, 1),
                value: OrderedFloat(1.0),
            }),
            CellState::RaisedPit(CellData {
                index: (2, 2),
                value: OrderedFloat(2.0),
            }),
            CellState::RaisedPit(CellData {
                index: (3, 3),
                value: OrderedFloat(3.0),
            }),
            CellState::RaisedPit(CellData {
                index: (4, 4),
                value: OrderedFloat(4.0),
            }),
        ];

        let tracker = PitResolutionTracker::new(raised_pits);
        assert_eq!(tracker.unseen[0].get_value().into_inner(), 4.0_f64);
        assert_eq!(tracker.unseen[1].get_value().into_inner(), 3.0_f64);
        assert_eq!(tracker.unseen[2].get_value().into_inner(), 2.0_f64);
        assert_eq!(tracker.unseen[3].get_value().into_inner(), 1.0_f64);
    }
}
