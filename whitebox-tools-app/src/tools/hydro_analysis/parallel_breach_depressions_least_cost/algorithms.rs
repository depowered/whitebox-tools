use super::structures::*;
use super::ValidatedArgs;
use ordered_float::OrderedFloat;
use pathfinding::prelude::{Matrix, MatrixFormatError};
use whitebox_raster::{Raster, RasterConfigs};

pub fn parallel_breach_depressions_least_cost(args: ValidatedArgs) -> Result<(), BreachError> {
    let raster = Raster::new(args.input_file.as_str(), "r")?;
    let configs = raster.configs.clone();
    let matrix = raster_to_matrix(raster)?;
    let pits = identify_pits(&matrix, configs.nodata.clone());
    let (matrix, pit_resolution_tracker) = resolve_pits(
        pits,
        args.max_cost,
        args.max_dist,
        args.flat_increment,
        args.minimize_dist,
    );
    let matrix = match args.fill_deps {
        true => fill_remaining_pits(matrix, pit_resolution_tracker.unsolved),
        false => matrix,
    };
    let mut output = matrix_to_raster(matrix, args.output_file.as_str(), &configs);
    match output.write() {
        Ok(_) => return Ok(()),
        Err(e) => return Err(BreachError::from(e)),
    }
}

pub fn raster_to_matrix(raster: Raster) -> Result<Matrix<OrderedFloat<f64>>, MatrixFormatError> {
    let mut values: Vec<f64> = vec![];
    for row in 0..raster.configs.rows as isize {
        values.append(&mut raster.get_row_data(row))
    }
    let values = values.into_iter().map(|v| OrderedFloat(v)).collect();
    Matrix::from_vec(raster.configs.rows, raster.configs.columns, values)
}

pub fn matrix_to_raster(
    matrix: Matrix<OrderedFloat<f64>>,
    file_name: &str,
    configs: &RasterConfigs,
) -> Raster {
    let mut raster = Raster::initialize_using_config(file_name, configs);
    for (row, values) in matrix.iter().enumerate() {
        let values: Vec<f64> = values.into_iter().map(|v| v.into_inner()).collect();
        raster.set_row_data(row as isize, values.to_owned())
    }
    raster
}

pub fn identify_pits(matrix: &Matrix<OrderedFloat<f64>>, nodata: f64) -> Vec<CellIndex> {
    let mut pits: Vec<CellIndex> = vec![];
    let nodata = OrderedFloat(nodata);
    for index in matrix.keys() {
        if let CellKind::Pit = CellKind::from_matrix(index, matrix, &nodata) {
            pits.push(CellIndex::from(index));
        }
    }
    pits
}

#[allow(unused_variables)]
pub fn resolve_pits(
    pits: Vec<CellIndex>,
    max_cost: f64,
    max_dist: isize,
    flat_increment: f64,
    minimize_dist: bool,
) -> (Matrix<OrderedFloat<f64>>, PitResolutionTracker) {
    todo!()
}

#[allow(unused_variables)]
pub fn fill_remaining_pits(
    matrix: Matrix<OrderedFloat<f64>>,
    pits: Vec<CellIndex>,
) -> Matrix<OrderedFloat<f64>> {
    todo!()
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

    fn get_mock_matrix() -> Matrix<OrderedFloat<f64>> {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(|v| OrderedFloat(v))
            .collect();
        Matrix::from_vec(3, 3, values).unwrap()
    }

    #[test]
    fn test_raster_to_matrix() {
        let input = get_mock_raster();
        let expected = get_mock_matrix();

        assert_eq!(raster_to_matrix(input).unwrap(), expected);
    }

    #[test]
    fn test_matrix_to_raster() {
        let input = get_mock_matrix();
        let expected = get_mock_raster();

        let result = matrix_to_raster(input, "test.tif", &expected.configs);

        assert_eq!(result.get_row_data(0), expected.get_row_data(0));
        assert_eq!(result.get_row_data(1), expected.get_row_data(1));
        assert_eq!(result.get_row_data(2), expected.get_row_data(2));
    }

    fn test_indexes_match() {
        let raster = get_mock_raster();
        let matrix = get_mock_matrix();

        assert_eq!(
            raster.get_value(0, 1),
            matrix.get((0, 1)).unwrap().into_inner()
        );
        assert_eq!(
            raster.get_value(2, 1),
            matrix.get((2, 1)).unwrap().into_inner()
        );
    }
}
