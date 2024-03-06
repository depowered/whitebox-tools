use anyhow::{anyhow, Result};
use ordered_float::OrderedFloat;
use whitebox_raster::Raster;

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct CellData {
    row: isize,
    column: isize,
    value: OrderedFloat<f64>,
}

impl CellData {
    pub fn new(row: isize, column: isize, value: f64) -> Self {
        CellData {
            row,
            column,
            value: OrderedFloat(value),
        }
    }

    pub fn get_value(self: &Self) -> OrderedFloat<f64> {
        self.value
    }

    pub fn with_new_value(self: Self, new_value: OrderedFloat<f64>) -> Self {
        Self {
            row: self.row,
            column: self.column,
            value: new_value,
        }
    }

    pub fn get_matrix_index(self: &Self, offset: isize) -> (usize, usize) {
        let matrix_row = (self.row + offset) as usize;
        let matrix_column = (self.column + offset) as usize;
        (matrix_row, matrix_column)
    }

    pub fn distance(self: &Self, other: &CellData) -> OrderedFloat<f64> {
        let square_dist =
            ((other.row - self.row).pow(2) + (other.column - self.column).pow(2)) as f64;
        OrderedFloat(square_dist.sqrt())
    }
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
    pub fn from_raster(raster: &Raster, row: isize, column: isize) -> CellState {
        let nodata = OrderedFloat(raster.configs.nodata);
        let value = OrderedFloat(raster.get_value(row, column));

        // Check if the CellState is NoData
        if OrderedFloat(value) == OrderedFloat(nodata) {
            return CellState::NoData(CellData::new(row, column, value.into_inner()));
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
                return CellState::Flowable(CellData::new(row, column, value.into_inner()));
            }
        }

        // If the CellState is neither Flowable nor NoData, it must be a RawPit
        return CellState::RawPit(CellData::new(row, column, value.into_inner()));
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

    pub fn get_matrix_index(self: &Self, offset: isize) -> (usize, usize) {
        self.get_data().get_matrix_index(offset)
    }

    pub fn distance(self: &Self, other: &CellState) -> OrderedFloat<f64> {
        self.get_data().distance(other.get_data())
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use whitebox_raster::{Raster, RasterConfigs};

    use super::{CellData, CellState, CellTransition};

    #[test]
    fn internal_state_access() {
        let state = CellState::Flowable(CellData {
            row: 1,
            column: 1,
            value: OrderedFloat(20.2),
        });
        let data = match state {
            CellState::Flowable(data) => data,
            _ => panic!("Unreachable"),
        };
        assert_eq!(data.row, 1);
        assert_eq!(data.column, 1);
        assert_eq!(data.value, OrderedFloat(20.2));
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
            CellState::Flowable(CellData::new(0, 0, 5.1)),
            CellState::from_raster(&raster, 0, 0)
        );
        // The center is RawPit because all neighbors are higher
        assert_eq!(
            CellState::RawPit(CellData::new(1, 1, 1.0)),
            CellState::from_raster(&raster, 1, 1)
        );
        // Cells out of range are NoData
        assert_eq!(
            CellState::NoData(CellData::new(5, 5, -9999.0)),
            CellState::from_raster(&raster, 5, 5)
        );
    }

    #[test]
    fn raise_pit_transition() {
        let state = CellState::RawPit(CellData::new(1, 1, 20.2));
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
        let state = CellState::NoData(CellData::new(1, 1, -9999.0));
        let new_state = state
            .clone()
            .transition(CellTransition::Breach(OrderedFloat(100.0)))
            .unwrap();
        assert_eq!(state, new_state);
    }

    #[test]
    fn get_data() {
        let state = CellState::RawPit(CellData::new(1, 1, 20.2));
        assert_eq!(state.get_data(), &CellData::new(1, 1, 20.2));
    }

    #[test]
    fn distance() {
        let state = CellState::RawPit(CellData::new(1, 1, 20.2));
        let other = CellState::Flowable(CellData::new(2, 2, 42.1));
        assert_eq!(state.distance(&other), OrderedFloat(2.0f64.sqrt()));

        let other = CellState::Flowable(CellData::new(4, 5, 42.1));
        assert_eq!(state.distance(&other), OrderedFloat(5.0f64));
    }
}
