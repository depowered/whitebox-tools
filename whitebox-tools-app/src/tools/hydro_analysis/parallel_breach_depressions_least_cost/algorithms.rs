use super::structures::{DEMCellType, PitStatus};
use pathfinding::matrix::{Matrix, MatrixFormatError};
use std::collections::BinaryHeap;
use whitebox_raster::{Raster, RasterConfigs};

pub fn raster_to_matrix(raster: &Raster) -> Result<Matrix<f64>, MatrixFormatError> {
    let mut values: Vec<f64> = vec![];
    for row in 0..raster.configs.rows as isize {
        values.append(&mut raster.get_row_data(row))
    }
    Matrix::from_vec(raster.configs.rows, raster.configs.columns, values)
}

pub fn matrix_to_raster(matrix: &Matrix<f64>, file_name: &str, configs: &RasterConfigs) -> Raster {
    let mut raster = Raster::initialize_using_config(file_name, configs);
    for (row, values) in matrix.iter().enumerate() {
        raster.set_row_data(row as isize, values.to_owned())
    }
    raster
}

pub fn identify_pits(raster: &Raster) -> BinaryHeap<PitStatus> {
    let num_rows = raster.configs.rows as isize;
    let num_columns = raster.configs.columns as isize;

    let mut minheap: BinaryHeap<PitStatus> = BinaryHeap::new();
    for row in 0..num_rows {
        for column in 0..num_columns {
            let cell_type = DEMCellType::from_raster(row, column, raster);
            match cell_type {
                DEMCellType::Flowable(_) => continue,
                DEMCellType::NoData(_) => continue,
                DEMCellType::Pit(cell) => minheap.push(PitStatus::Unseen(cell)),
            }
        }
    }
    return minheap;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pathfinding::matrix::Matrix;
    use whitebox_raster::{Raster, RasterConfigs};

    fn get_mock_raster() -> Raster {
        let configs: RasterConfigs = RasterConfigs {
            rows: 3,
            columns: 3,
            ..Default::default()
        };
        let mut raster = Raster::initialize_using_config("test.tif", &configs);
        raster.set_row_data(0, vec![1.0, 2.0, 3.0]);
        raster.set_row_data(1, vec![4.0, 5.0, 6.0]);
        raster.set_row_data(2, vec![7.0, 8.0, 9.0]);

        raster
    }

    fn get_mock_matrix() -> Matrix<f64> {
        Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap()
    }

    #[test]
    fn test_raster_to_matrix() {
        let input = get_mock_raster();
        let expected = get_mock_matrix();

        assert_eq!(raster_to_matrix(&input).unwrap(), expected);
    }

    #[test]
    fn test_matrix_to_raster() {
        let input = get_mock_matrix();
        let expected = get_mock_raster();

        let result = matrix_to_raster(&input, "test.tif", &expected.configs);

        assert_eq!(result.get_row_data(0), expected.get_row_data(0));
        assert_eq!(result.get_row_data(1), expected.get_row_data(1));
        assert_eq!(result.get_row_data(2), expected.get_row_data(2));
    }

    fn test_indexes_match() {
        let raster = get_mock_raster();
        let matrix = get_mock_matrix();

        assert_eq!(&raster.get_value(0, 1), matrix.get((0, 1)).unwrap());
        assert_eq!(&raster.get_value(2, 1), matrix.get((2, 1)).unwrap());
    }
}
