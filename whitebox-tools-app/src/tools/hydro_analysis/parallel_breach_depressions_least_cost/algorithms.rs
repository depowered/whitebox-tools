use std::collections::HashSet;

use super::structures::*;
use super::ValidatedArgs;
use anyhow::Result;
use ordered_float::OrderedFloat;
use pathfinding::prelude::{dijkstra, Matrix, MatrixFormatError};
use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use whitebox_raster::{Raster, RasterConfigs};

pub fn parallel_breach_depressions_least_cost(args: ValidatedArgs) -> Result<()> {
    // Load the input raster into memory
    let raster = Raster::new(args.input_file.as_str(), "r")?;

    // Clone the input raster config for use in constructing the output raster later
    let configs = raster.configs.clone();

    // Convert the raster into a pathfinding::Matrix
    let mut matrix = raster_to_matrix(raster)?;

    // Find all the pits and prepare them for beaching by raising their value to just
    // below the value of their lowest neighbor
    let raw_pits = get_raw_pits(&matrix);
    let mut raised_pits = raise_pits(&mut matrix, raw_pits, args.flat_increment);

    // Initialize structures used for tracking breach progress and results
    let mut in_progress: HashSet<CellData> = HashSet::new();

    // Attempt to breach each pit using the main thread to prepare search matrices and apply breach
    // paths with worker threads responsible for preforming the breach path searches
    let (sender, receiver) = channel::<Vec<(CellState, CellTransition)>>();
    let num_workers = args.num_threads - 1;
    let pool = ThreadPool::new(num_workers);

    while !raised_pits.is_empty() && !in_progress.is_empty() {
        // Only prepare and dispatch a search operation if the queue is empty to limit memory
        // consumption
        if pool.queued_count() < 1 {
            // Prepare the next pit for breach path search if one is available
            if let Some(raised_pit) =
                get_next_available_pit(&mut raised_pits, &mut in_progress, args.max_dist)
            {
                let search_matrix = get_search_matrix(&matrix, &raised_pit, args.max_dist);

                // Dispatch the search operation to a worker thread
                let sender = sender.clone();
                pool.execute(move || {
                    let path = find_path_with_dijkstra(
                        &raised_pit,
                        &search_matrix,
                        args.flat_increment,
                        args.max_cost,
                        args.minimize_dist,
                    );
                    let transitions =
                        compute_cell_transitions(path, &raised_pit, args.flat_increment);
                    sender
                        .send(transitions)
                        .expect("An error occurred while sending data to the main thread.");
                });
            }
        }

        // Check for any completed search operations, handling them accordingly
        // Otherwise, continue to the top of the loop and prepare another search operation
        match receiver.try_recv() {
            Ok(transitions) => {
                // Remove the pit from the in_progress set
                let pit_data = transitions[0].0.get_data();
                in_progress.remove(pit_data);
                // Modify the matrix to reflect the pathfinding result
                apply_cell_transitions(&mut matrix, transitions)?;
            }
            Err(_) => continue,
        }
    }

    // Gather the unsolved pits and fill them if the fill depressions flag is present
    let unsolved_pits = get_unsolved_pits(&matrix);
    if args.fill_deps && !unsolved_pits.is_empty() {
        fill_remaining_pits(&mut matrix, unsolved_pits);
    }

    // Write the output raster using the same configs as the input raster
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
    for (row, states) in matrix.iter().enumerate() {
        let values: Vec<f64> = states
            .into_iter()
            .map(|v| v.get_value().into_inner())
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

fn get_unsolved_pits(matrix: &Matrix<CellState>) -> Vec<CellState> {
    matrix
        .values()
        .filter(|state| matches!(**state, CellState::UnsolvedPit(_)))
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
        let index = raw_pit.get_index();

        let min_neighbor_value: OrderedFloat<f64> = get_neighbor_states(matrix, index)
            .iter()
            .map(|state| state.get_value())
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

    // Sort pits from highest to lowest value
    raised_pits.sort_by(|a, b| b.get_value().cmp(&a.get_value()));
    raised_pits
}

fn get_next_available_pit(
    raised_pits: &mut Vec<CellState>,
    in_progress: &mut HashSet<CellData>,
    max_dist: usize,
) -> Option<CellState> {
    // No more pits to solve
    if raised_pits.is_empty() {
        return None;
    }

    // If no pits are in progress, return the lowest value pit without preforming distance checks
    if in_progress.is_empty() {
        let pit = raised_pits.pop()?;
        in_progress.insert(pit.get_data().clone());
        return Some(pit);
    }

    // Find the lowest value pit that is at least two times the max distance from any
    // pit currently in progress
    let min_dist_between = OrderedFloat((2 * max_dist) as f64);
    for (i, pit) in raised_pits.iter().enumerate().rev() {
        let pit_data = pit.get_data();
        for in_progress_pit_data in in_progress.iter() {
            if pit_data.distance(in_progress_pit_data) < min_dist_between {
                continue;
            }
        }
        let pit = raised_pits.remove(i);
        in_progress.insert(pit.get_data().clone());
        return Some(pit);
    }
    return None;
}

fn get_search_matrix(
    matrix: &Matrix<CellState>,
    raised_pit: &CellState,
    max_dist: usize,
) -> Matrix<CellState> {
    let (row, column) = raised_pit.get_index();

    // Guard against under-flowing the row or column indices
    let min_row = std::cmp::max(0, row as isize - max_dist as isize) as usize;
    let min_column = std::cmp::max(0, column as isize - max_dist as isize) as usize;

    // Guard against going out of bounds with the row or column indices
    // The matrix.slice method does not accept RangeInclusive, so we add 1 to the max so that the
    // include_start..exclude_end results in the same sized slice as include_start..=include_end
    let max_row = std::cmp::min(matrix.rows, row + max_dist) + 1;
    let max_column = std::cmp::min(matrix.columns, column + max_dist) + 1;

    matrix
        .slice(min_row..max_row, min_column..max_column)
        .expect("The slice should not be out of range")
}

fn get_cost_to_successor(
    node: &CellState,
    neighbor: &CellState,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
    minimize_dist: bool,
) -> OrderedFloat<f64> {
    let zero_cost = OrderedFloat(0.0f64);

    if let CellState::NoData(_) = neighbor {
        return zero_cost;
    };

    // Estimate the accumulated flat increment necessary to ensure flow from the pit
    // to the successor
    let node_dist_from_pit = raised_pit.distance(node);
    let neighbor_dist_from_pit = OrderedFloat(node_dist_from_pit.ceil() + 1_f64);
    let accumulated_flat_increment = neighbor_dist_from_pit * flat_increment;

    let pit_value = raised_pit.get_value();
    let neighbor_value = neighbor.get_value();
    let neighbor_dist_from_node = node.distance(neighbor);

    let cost = if minimize_dist {
        neighbor_dist_from_node * (neighbor_value - pit_value - accumulated_flat_increment)
    } else {
        neighbor_value - pit_value - accumulated_flat_increment
    };

    // The algorithm requires that costs are non-negative. Threshold negative costs to zero.
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
    minimize_dist: bool,
) -> Vec<(CellState, OrderedFloat<f64>)> {
    let neighbors = get_neighbor_states(matrix, node.get_index());
    let mut costs = vec![];
    for neighbor in neighbors {
        costs.push((
            neighbor.clone(),
            get_cost_to_successor(node, &neighbor, raised_pit, flat_increment, minimize_dist),
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
    // Estimate the accumulated flat increment necessary to ensure flow from the pit
    // to the successor
    let node_dist_from_pit = raised_pit.distance(node);
    let neighbor_dist_from_pit = OrderedFloat(node_dist_from_pit.ceil() + 1_f64);
    let accumulated_flat_increment = neighbor_dist_from_pit * flat_increment;

    let pit_value = raised_pit.get_value();
    let is_success = |value: OrderedFloat<f64>| value < (pit_value - accumulated_flat_increment);
    match node {
        CellState::NoData(_) => true,
        CellState::RawPit(_) => false,
        CellState::Flowable(data) => is_success(data.value),
        CellState::RaisedPit(data) => is_success(data.value),
        CellState::UnsolvedPit(data) => is_success(data.value),
    }
}

fn find_path_with_dijkstra(
    raised_pit: &CellState,
    matrix: &Matrix<CellState>,
    flat_increment: OrderedFloat<f64>,
    max_cost: OrderedFloat<f64>,
    minimize_dist: bool,
) -> Option<(Vec<CellState>, OrderedFloat<f64>)> {
    dijkstra(
        raised_pit,
        |node| {
            dijkstra_successors(
                node,
                matrix,
                raised_pit,
                flat_increment,
                max_cost,
                minimize_dist,
            )
        },
        |node| dijkstra_success(node, raised_pit, flat_increment),
    )
}

fn compute_cell_transitions(
    path: Option<(Vec<CellState>, OrderedFloat<f64>)>,
    raised_pit: &CellState,
    flat_increment: OrderedFloat<f64>,
) -> Vec<(CellState, CellTransition)> {
    match path {
        None => {
            let transition = (raised_pit.clone(), CellTransition::MarkUnsolved);
            return vec![transition];
        }
        Some(breach) => {
            let pit_value = raised_pit.get_value();
            let mut transitions = vec![];
            let (nodes, _) = breach;
            for (i, node) in nodes.iter().enumerate() {
                let new_value = pit_value - flat_increment * i as f64;
                let transition = (node.clone(), CellTransition::Breach(new_value));
                transitions.push(transition);
            }
            return transitions;
        }
    }
}

fn apply_cell_transitions(
    matrix: &mut Matrix<CellState>,
    transitions: Vec<(CellState, CellTransition)>,
) -> Result<()> {
    for (cell, transition) in transitions {
        let index = cell.get_index();
        *matrix.get_mut(index).expect(
            "The transitions are derived from the matrix, so the corresponding index will always be in range.",
        ) = cell.transition(transition)?;
    }
    Ok(())
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
fn fill_remaining_pits(matrix: &mut Matrix<CellState>, unsolved_pits: Vec<CellState>) {
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
            CellState::new(InitialState::Flowable, (0, 0), OrderedFloat(5.1)),
            CellState::new(InitialState::Flowable, (0, 1), OrderedFloat(5.2)),
            CellState::new(InitialState::Flowable, (0, 2), OrderedFloat(5.3)),
            CellState::new(InitialState::Flowable, (1, 0), OrderedFloat(6.1)),
            CellState::new(InitialState::RawPit, (1, 1), OrderedFloat(1.0)),
            CellState::new(InitialState::Flowable, (1, 2), OrderedFloat(6.3)),
            CellState::new(InitialState::Flowable, (2, 0), OrderedFloat(7.1)),
            CellState::new(InitialState::Flowable, (2, 1), OrderedFloat(7.2)),
            CellState::new(InitialState::Flowable, (2, 2), OrderedFloat(7.3)),
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
            matrix.get((0, 1)).unwrap().get_value()
        );
        assert_eq!(
            OrderedFloat(raster.get_value(2, 1)),
            matrix.get((2, 1)).unwrap().get_value()
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
