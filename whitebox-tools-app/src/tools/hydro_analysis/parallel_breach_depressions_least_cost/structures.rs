use ordered_float::OrderedFloat;
use pathfinding::prelude::Matrix;

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
    Edge,
    Pit,
    Nodata,
}

impl CellKind {
    pub fn from_matrix(
        index: (usize, usize),
        matrix: &Matrix<OrderedFloat<f64>>,
        nodata: &OrderedFloat<f64>,
    ) -> CellKind {
        let elevation = match matrix.get(index) {
            Some(elevation) => elevation,
            None => return CellKind::Nodata,
        };
        for neighbor_index in matrix.neighbours(index, true) {
            match matrix.get(neighbor_index) {
                Some(neighbor_elevation) => {
                    if neighbor_elevation == nodata || neighbor_elevation < elevation {
                        return CellKind::Flowable;
                    }
                }
                None => return CellKind::Edge,
            }
        }
        return CellKind::Pit;
    }
}
