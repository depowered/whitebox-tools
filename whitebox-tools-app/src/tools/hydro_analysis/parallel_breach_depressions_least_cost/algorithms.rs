use super::structures::*;
use super::ValidatedArgs;
use anyhow::Result;
use ordered_float::OrderedFloat;
use pathfinding::prelude::{dijkstra, Matrix, MatrixFormatError};
use whitebox_raster::{Raster, RasterConfigs};

pub fn parallel_breach_depressions_least_cost(args: ValidatedArgs) -> Result<()> {
    let raster = Raster::new(args.input_file.as_str(), "r")?;
    let configs = raster.configs.clone();
    let mut matrix = raster_to_matrix(raster)?;
    let raw_pits = get_raw_pits(&matrix);
    let raised_pits = raise_pits(&mut matrix, raw_pits, OrderedFloat(args.flat_increment));
    let pit_resolution_tracker = resolve_pits(
        &mut matrix,
        raised_pits,
        OrderedFloat(args.max_cost),
        args.max_dist,
        OrderedFloat(args.flat_increment),
        args.minimize_dist,
    );
    if args.fill_deps {
        fill_remaining_pits(&mut matrix, pit_resolution_tracker.unsolved);
    }
    let mut output = matrix_to_raster(matrix, args.output_file.as_str(), &configs);
    output.write()?;
    Ok(())
}

fn raster_to_matrix(raster: Raster) -> Result<Matrix<CellState>, MatrixFormatError> {
    let mut values: Vec<CellState> = vec![];
    for row in 0..raster.configs.rows as isize {
        for column in 0..raster.configs.columns as isize {
            values.push(CellState::from_raster(&raster, row, column))
        }
    }
    Matrix::from_vec(raster.configs.rows, raster.configs.columns, values)
}

fn matrix_to_raster(matrix: Matrix<CellState>, file_name: &str, configs: &RasterConfigs) -> Raster {
    let mut raster = Raster::initialize_using_config(file_name, configs);
    for (row, values) in matrix.iter().enumerate() {
        let values: Vec<f64> = values
            .into_iter()
            .map(|v| v.get_data().value.into_inner())
            .collect();
        raster.set_row_data(row as isize, values)
    }
    raster
}

fn get_raw_pits(matrix: &Matrix<CellState>) -> Vec<CellState> {
    matrix
        .values()
        .filter(|state| matches!(**state, CellState::RawPit(_)))
        .cloned()
        .collect()
}

fn get_neighbor_states(matrix: &Matrix<CellState>, index: (usize, usize)) -> Vec<CellState> {
    matrix
        .neighbours(index, true)
        .filter_map(|i| matrix.get(i))
        .cloned()
        .collect()
}

fn raise_pits(
    matrix: &mut Matrix<CellState>,
    raw_pits: Vec<CellState>,
    flat_increment: OrderedFloat<f64>,
) -> Vec<CellState> {
    let mut raised_pits = vec![];
    for raw_pit in raw_pits {
        let index = raw_pit.get_data().index;

        let min_neighbor_value: OrderedFloat<f64> = get_neighbor_states(matrix, index)
            .iter()
            .map(|state| state.get_data().value)
            .min()
            .unwrap();

        let raised_pit = raw_pit
            .transition(CellTransition::RaisePit(
                min_neighbor_value - flat_increment,
            ))
            .unwrap();

        *matrix.get_mut(index).unwrap() = raised_pit.clone();
        raised_pits.push(raised_pit);
    }
    raised_pits
}

fn get_cost_to_successor(
    node: &CellState,
    neighbor: &CellState,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
) -> OrderedFloat<f64> {
    let zero_cost = OrderedFloat(0.0f64);

    if let CellState::NoData(_) = neighbor {
        return zero_cost;
    };

    let node_dist_from_pit = raised_pit.distance(node);
    let neighbor_dist_from_pit = node_dist_from_pit + node.distance(neighbor);
    let accum_flat_increment = neighbor_dist_from_pit * flat_increment;

    let pit_value = raised_pit.get_data().value;
    let neighbor_value = neighbor.get_data().value;

    let cost = neighbor_value - (pit_value + accum_flat_increment);

    if cost < zero_cost {
        return zero_cost;
    }
    return cost;
}

/// Returns a list of successors for a given node, along with the cost for moving from the
/// node to the successor. This cost must be non-negative.
fn dijkstra_successors(
    node: &CellState,
    matrix: &Matrix<CellState>,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
    max_cost: OrderedFloat<f64>,
) -> Vec<(CellState, OrderedFloat<f64>)> {
    let neighbors = get_neighbor_states(matrix, node.get_data().index);
    let mut costs = vec![];
    for neighbor in neighbors {
        costs.push((
            neighbor.clone(),
            get_cost_to_successor(node, &neighbor, raised_pit, flat_increment),
        ))
    }
    // filter by max_cost
    let successors = costs.into_iter().filter(|cost| cost.1 < max_cost).collect();

    successors
}

/// Checks whether the goal has been reached.
fn dijkstra_success(
    node: &CellState,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
) -> bool {
    let pit_value = raised_pit.get_data().value;
    let distance = raised_pit.distance(node);
    let is_success = |value: OrderedFloat<f64>| value < (pit_value - distance * flat_increment);
    match node {
        CellState::NoData(_) => true,
        CellState::RawPit(_) => false,
        CellState::Flowable(data) => is_success(data.value),
        CellState::RaisedPit(data) => is_success(data.value),
        CellState::UnsolvedPit(data) => is_success(data.value),
    }
}

fn find_path_with_dijkstra(
    start: &CellState,
    matrix: &Matrix<CellState>,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
    max_cost: OrderedFloat<f64>,
) -> Option<(Vec<CellState>, OrderedFloat<f64>)> {
    dijkstra(
        start,
        |node| dijkstra_successors(node, matrix, raised_pit, flat_increment, max_cost),
        |node| dijkstra_success(node, start, flat_increment),
    )
}

#[allow(unused_variables)]
fn resolve_pits(
    matrix: &mut Matrix<CellState>,
    raised_pits: Vec<CellState>,
    max_cost: OrderedFloat<f64>,
    max_dist: isize,
    flat_increment: OrderedFloat<f64>,
    minimize_dist: bool,
) -> PitResolutionTracker {
    todo!()
}

#[allow(unused_variables)]
fn fill_remaining_pits(matrix: &mut Matrix<CellState>, unsloved_pits: Vec<CellState>) {
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
        raster.set_row_data(0, vec![5.1, 5.2, 5.3]);
        raster.set_row_data(1, vec![6.1, 1.0, 6.3]);
        raster.set_row_data(2, vec![7.1, 7.2, 7.3]);

        raster
    }

    fn get_mock_matrix() -> Matrix<CellState> {
        let values = vec![
            CellState::new(InitalState::Flowable, (0, 0), OrderedFloat(5.1)),
            CellState::new(InitalState::Flowable, (0, 1), OrderedFloat(5.2)),
            CellState::new(InitalState::Flowable, (0, 2), OrderedFloat(5.3)),
            CellState::new(InitalState::Flowable, (1, 0), OrderedFloat(6.1)),
            CellState::new(InitalState::RawPit, (1, 1), OrderedFloat(1.0)),
            CellState::new(InitalState::Flowable, (1, 2), OrderedFloat(6.3)),
            CellState::new(InitalState::Flowable, (2, 0), OrderedFloat(7.1)),
            CellState::new(InitalState::Flowable, (2, 1), OrderedFloat(7.2)),
            CellState::new(InitalState::Flowable, (2, 2), OrderedFloat(7.3)),
        ];
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

    #[test]
    fn test_indexes_match() {
        let raster = get_mock_raster();
        let matrix = get_mock_matrix();

        assert_eq!(
            OrderedFloat(raster.get_value(0, 1)),
            matrix.get((0, 1)).unwrap().get_data().value
        );
        assert_eq!(
            OrderedFloat(raster.get_value(2, 1)),
            matrix.get((2, 1)).unwrap().get_data().value
        );
    }

    #[test]
    fn test_identify_pits() {
        let matrix = get_mock_matrix();
        let pits = get_raw_pits(&matrix);

        assert_eq!(1usize, pits.len());
        assert_eq!(matrix.get((1, 1)).unwrap(), &pits[0])
    }

    #[test]
    fn test_raise_pits() {
        let mut matrix = get_mock_matrix();
        let raw_pits = get_raw_pits(&matrix);
        let raised_pits = raise_pits(&mut matrix, raw_pits, OrderedFloat(0.1));

        assert_eq!(1usize, raised_pits.len());
        assert_eq!(
            matrix.get((1, 1)).unwrap(),
            &CellState::RaisedPit(CellData {
                index: (1, 1),
                value: OrderedFloat(5.0)
            })
        )
    }
}
