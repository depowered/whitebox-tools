use anyhow::{anyhow, Result};
use ordered_float::OrderedFloat;
use pathfinding::prelude::Matrix;
use whitebox_raster::Raster;

#[derive(PartialEq, Eq)]
pub struct CellIndex {
    row: usize,
    column: usize,
}

impl From<(usize, usize)> for CellIndex {
    fn from(index: (usize, usize)) -> Self {
        let (row, column) = index;
        CellIndex { row, column }
    }
}

impl CellIndex {
    pub fn euclidean_distance(self: &Self, other: &CellIndex) -> f64 {
        let square_dist =
            ((other.row - self.row).pow(2) + (other.column - self.column).pow(2)) as f64;
        square_dist.sqrt()
    }

    pub fn as_usize(self: &Self) -> (usize, usize) {
        (self.row, self.column)
    }

    pub fn as_isize(self: &Self) -> (isize, isize) {
        (self.row as isize, self.column as isize)
    }
}

#[derive(PartialEq, Eq)]
pub struct Pit {
    pub index: CellIndex,
    pub elevation: OrderedFloat<f64>,
}

pub struct PitResolutionTracker {
    pub unseen: Vec<CellIndex>,
    pub in_progress: Vec<CellIndex>,
    pub solved: Vec<CellIndex>,
    pub unsolved: Vec<CellIndex>,
}

pub enum BreachError {
    RasterIOError(std::io::Error),
    MatrixFormatError(pathfinding::matrix::MatrixFormatError),
}

impl From<std::io::Error> for BreachError {
    fn from(error: std::io::Error) -> Self {
        BreachError::RasterIOError(error)
    }
}

impl From<pathfinding::matrix::MatrixFormatError> for BreachError {
    fn from(error: pathfinding::matrix::MatrixFormatError) -> Self {
        BreachError::MatrixFormatError(error)
    }
}

pub enum CellKind {
    Flowable,
    Pit,
    Nodata,
}

impl CellKind {
    pub fn from_matrix(
        index: (usize, usize),
        matrix: &Matrix<OrderedFloat<f64>>,
        nodata: OrderedFloat<f64>,
    ) -> CellKind {
        let elevation = match matrix.get(index) {
            Some(elevation) => *elevation,
            None => return CellKind::Nodata,
        };
        let neighbor_elevations = Self::get_neighbor_elevations(index, matrix);
        for neighbor_elevation in neighbor_elevations {
            if neighbor_elevation == nodata || neighbor_elevation < elevation {
                return CellKind::Flowable;
            }
        }
        return CellKind::Pit;
    }

    pub fn get_neighbor_elevations(
        index: (usize, usize),
        matrix: &Matrix<OrderedFloat<f64>>,
    ) -> Vec<OrderedFloat<f64>> {
        let mut neighbor_elevations = vec![];
        for neighbor_index in matrix.neighbours(index, true) {
            if let Some(elevation) = matrix.get(neighbor_index) {
                neighbor_elevations.push(*elevation);
            }
        }
        neighbor_elevations
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CellData {
    index: (usize, usize),
    value: OrderedFloat<f64>,
}

impl CellData {
    fn with_new_value(self: Self, new_value: OrderedFloat<f64>) -> Self {
        Self {
            index: self.index,
            value: new_value,
        }
    }
}

enum InitalState {
    NoData,
    Flowable,
    RawPit,
}

#[derive(Debug, Eq, PartialEq)]
enum CellState {
    NoData(CellData),
    Flowable(CellData),
    RawPit(CellData),
    RaisedPit(CellData),
    UnsolvedPit(CellData),
}

enum CellTransition {
    RaisePit(OrderedFloat<f64>),
    Breach(OrderedFloat<f64>),
    MarkUnsolved,
}

impl CellState {
    fn new(
        inital_state: InitalState,
        index: (usize, usize),
        value: OrderedFloat<f64>,
    ) -> CellState {
        match inital_state {
            InitalState::NoData => Self::NoData(CellData { index, value }),
            InitalState::Flowable => Self::Flowable(CellData { index, value }),
            InitalState::RawPit => Self::RawPit(CellData { index, value }),
        }
    }

    fn from_raster(raster: &Raster, row: isize, column: isize) -> CellState {
        let nodata = OrderedFloat(raster.configs.nodata);
        let index = (row as usize, column as usize);
        let value = OrderedFloat(raster.get_value(row, column));

        // Check if the CellState is NoData
        if value == nodata {
            return CellState::new(InitalState::NoData, index, value);
        }

        // Check if the CellState is Flowable. That is, at least one neighbor has a value
        // that is equal to nodata or lower than the central cell value
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
                return CellState::new(InitalState::Flowable, index, value);
            }
        }

        // If the CellState is neither Flowable nor NoData, it must be a RawPit
        return CellState::new(InitalState::RawPit, index, value);
    }

    fn transition(self: Self, transition: CellTransition) -> Result<CellState> {
        match (self, transition) {
            // There are no valid transitions starting from NoData
            (CellState::NoData(_), _) => Err(anyhow!("NoData is immutable")),

            // The only valid transition starting from Flowable is Breach
            (CellState::Flowable(data), CellTransition::Breach(new_value)) => {
                Ok(CellState::Flowable(data.with_new_value(new_value)))
            }
            (CellState::Flowable(_), _) => Err(anyhow!("Error")),

            // The only valid transition starting from RawPit is RaisePit
            (CellState::RawPit(data), CellTransition::RaisePit(new_value)) => {
                Ok(CellState::RaisedPit(data.with_new_value(new_value)))
            }
            (CellState::RawPit(_), _) => Err(anyhow!("Error")),

            // The only valid transitions starting from RaisedPit are Breach and MarkUnsolved
            (CellState::RaisedPit(data), CellTransition::Breach(new_value)) => {
                Ok(CellState::Flowable(data.with_new_value(new_value)))
            }
            (CellState::RaisedPit(data), CellTransition::MarkUnsolved) => {
                Ok(CellState::UnsolvedPit(data))
            }
            (CellState::RaisedPit(_), _) => Err(anyhow!("Error")),

            // There are no valid transitions starting from NoData
            (CellState::UnsolvedPit(_), _) => Err(anyhow!("Error")),
        }
    }

    fn get_data(self: &Self) -> &CellData {
        match self {
            CellState::NoData(data) => data,
            CellState::Flowable(data) => data,
            CellState::RawPit(data) => data,
            CellState::RaisedPit(data) => data,
            CellState::UnsolvedPit(data) => data,
        }
    }

    fn distance(self: &Self, other: &CellState) -> OrderedFloat<f64> {
        let (row1, column1) = self.get_data().index;
        let (row2, column2) = other.get_data().index;
        let square_dist = ((row2 - row1).pow(2) + (column2 - column1).pow(2)) as f64;
        OrderedFloat(square_dist.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use whitebox_raster::{Raster, RasterConfigs};

    use super::{CellData, CellState, CellTransition, InitalState};

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
        let state = CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(20.2));
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
            CellState::new(InitalState::Flowable, (0, 0), OrderedFloat(5.1)),
            CellState::from_raster(&raster, 0, 0)
        );
        // The center is RawPit because all neighbors are higher
        assert_eq!(
            CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(1.0)),
            CellState::from_raster(&raster, 1, 1)
        );
        // Cells out of range are NoData
        assert_eq!(
            CellState::new(InitalState::NoData, (5, 5), OrderedFloat(-9999.0)),
            CellState::from_raster(&raster, 5, 5)
        );
    }

    #[test]
    fn raise_pit_transition() {
        let state = CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(20.2));
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
        let state = CellState::new(InitalState::NoData, (1, 1), OrderedFloat(-9999.0));
        let new_state = state.transition(CellTransition::Breach(OrderedFloat(100.0)));
        if let Ok(_) = new_state {
            panic!("Unreachable")
        }
    }

    #[test]
    fn get_data() {
        let state = CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(20.2));
        assert_eq!(state.get_data().index, (1, 1));
        assert_eq!(state.get_data().value, OrderedFloat(20.2));
    }

    #[test]
    fn distance() {
        let state = CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(20.2));
        let other = CellState::new(InitalState::Flowable, (2, 2), OrderedFloat(42.1));
        assert_eq!(state.distance(&other), OrderedFloat(2.0f64.sqrt()));

        let other = CellState::new(InitalState::Flowable, (4, 5), OrderedFloat(42.1));
        assert_eq!(state.distance(&other), OrderedFloat(5.0f64));
    }
}
