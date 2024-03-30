use std::collections::{BinaryHeap, HashSet};

use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::sync::{mpsc, Arc};
use std::thread;
use threadpool::ThreadPool;
use whitebox_common::structures::Array2D;
use whitebox_raster::Raster;

pub fn raise_pits(
    input: &Arc<Raster>,
    output: &mut Raster,
    num_procs: isize,
    small_num: f64,
) -> Vec<(isize, isize, f64)> {
    let rows = input.configs.rows as isize;
    let columns = input.configs.columns as isize;
    let nodata = input.configs.nodata;
    // Raise pit cells to minimize the depth of breach channels.
    let (tx, rx) = mpsc::channel();
    for tid in 0..num_procs {
        let input = input.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let (mut z, mut zn, mut min_zn): (f64, f64, f64);
            let mut flag: bool;
            let dx = [1, 1, 1, 0, -1, -1, -1, 0];
            let dy = [-1, 0, 1, 1, 1, 0, -1, -1];
            for row in (0..rows).filter(|r| r % num_procs == tid) {
                let mut data = input.get_row_data(row);
                let mut pits = vec![];
                for col in 0..columns {
                    z = input.get_value(row, col);
                    if z != nodata {
                        flag = true;
                        min_zn = f64::INFINITY;
                        for n in 0..8 {
                            zn = input.get_value(row + dy[n], col + dx[n]);
                            if zn < min_zn {
                                min_zn = zn;
                            }
                            if zn == nodata {
                                // It's an edge cell.
                                flag = false;
                                break;
                            }
                            if zn < z {
                                // There's a lower neighbour
                                flag = false;
                                break;
                            }
                        }
                        if flag {
                            data[col as usize] = min_zn - small_num;
                            pits.push((row, col, z));
                        }
                    }
                }
                tx.send((row, data, pits)).unwrap();
            }
        });
    }

    let mut undefined_flow_cells: Vec<(isize, isize, f64)> = vec![];
    for r in 0..rows {
        let (row, data, mut pits) = rx.recv().expect("Error receiving data from thread.");
        output.set_row_data(row, data);
        undefined_flow_cells.append(&mut pits);
    }
    undefined_flow_cells
}

pub fn least_cost_search(
    undefined_flow_cells: &mut Vec<(isize, isize, f64)>,
    raster: &mut Raster,
    max_dist: isize,
    max_cost: f64,
    flat_increment: f64,
    minimize_dist: bool,
    num_threads: usize,
) -> Vec<(isize, isize, f64)> {
    let small_num = flat_increment;
    let configs = raster.configs.clone();

    /* Vec is a stack and so if we want to pop the values from lowest to highest, we need to sort
    them from highest to lowest. */
    undefined_flow_cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Equal));
    let mut unsolved_pits = vec![];

    // Initialize structures used for tracking breach progress and results
    let mut in_progress: HashSet<(isize, isize)> = HashSet::new();

    // Attempt to breach each pit using the main thread to prepare search matrices and apply breach
    // paths with worker threads responsible for preforming the breach path searches
    let (sender, receiver) = mpsc::channel::<((isize, isize, f64), BreachOutcome)>();
    let num_workers = num_threads - 1;
    println!("Spawning {} worker threads", num_workers);
    let pool = ThreadPool::new(num_workers);

    while !undefined_flow_cells.is_empty() || !in_progress.is_empty() {
        // Only prepare and dispatch a search operation if the queue is empty to limit memory
        // consumption
        if pool.queued_count() < 1 {
            // Prepare the next pit for breach path search if one is available
            if let Some(cell) =
                get_next_available_pit(undefined_flow_cells, &mut in_progress, max_dist)
            {
                let search_array = SearchArray::new(cell, raster, max_dist);

                // Dispatch the search operation to a worker thread
                let sender = sender.clone();
                pool.execute(move || {
                    let outcome = try_breach(
                        search_array,
                        configs.resolution_x,
                        configs.resolution_y,
                        max_cost,
                        small_num,
                        minimize_dist,
                        max_dist,
                    );
                    sender
                        .send((cell, outcome))
                        .expect("An error occurred while sending data to the main thread.");
                });
            }
        }

        // Check for any completed search operations, handling them accordingly
        // Otherwise, continue to the top of the loop and prepare another search operation
        match receiver.try_recv() {
            Ok((cell, outcome)) => {
                match outcome {
                    BreachOutcome::PreviouslyBreached => {}
                    BreachOutcome::PathFound(path) => {
                        if path.is_empty() {
                            println!("cell: {:?}, path is empty", cell);
                        }
                        for cell in path {
                            let (row, column, value) = cell;
                            raster.set_value(row, column, value);
                        }
                    }
                    BreachOutcome::NoPathFound => {
                        unsolved_pits.push(cell); // Add it to the list for the next iteration
                    }
                }
                in_progress.remove(&(cell.0, cell.1));
            }
            Err(_) => continue,
        }
    }
    unsolved_pits
}

fn distance_between(x1: isize, y1: isize, x2: isize, y2: isize) -> f64 {
    let dx = x2 - x1;
    let dy = y1 - y2;
    let square_dist = (dx * dx + dy * dy) as f64;
    square_dist.sqrt()
}

fn get_next_available_pit(
    undefined_flow_cells: &mut Vec<(isize, isize, f64)>,
    in_progress: &mut HashSet<(isize, isize)>,
    max_dist: isize,
) -> Option<(isize, isize, f64)> {
    // No more pits to solve
    if undefined_flow_cells.is_empty() {
        return None;
    }

    // If no pits are in progress, return the lowest value pit without preforming distance checks
    if in_progress.is_empty() {
        let pit = undefined_flow_cells.pop()?;
        in_progress.insert((pit.0, pit.1));
        return Some(pit);
    }

    // Find the lowest value pit that is at least two times the max distance from any
    // pit currently in progress
    let diagonal_max_dist = (max_dist as f64).powi(2).sqrt();
    let min_dist_between = 2.0 * diagonal_max_dist;
    for (i, pit) in undefined_flow_cells.iter().enumerate().rev() {
        // let pit_data = pit.get_data();
        for in_progress_pit in in_progress.iter() {
            let dist_between = distance_between(pit.0, pit.1, in_progress_pit.0, in_progress_pit.1);
            if dist_between < min_dist_between {
                continue;
            }
        }
        let pit = undefined_flow_cells.remove(i);
        in_progress.insert((pit.0, pit.1));
        return Some(pit);
    }
    return None;
}

fn try_breach(
    search_array: SearchArray,
    resx: f64,
    resy: f64,
    max_cost: f64,
    small_num: f64,
    minimize_dist: bool,
    max_dist: isize,
) -> BreachOutcome {
    let rows = search_array.array.rows as isize;
    let columns = search_array.array.columns as isize;
    let nodata = search_array.array.nodata;

    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];
    let diagres = (resx * resx + resy * resy).sqrt();
    let cost_dist = [diagres, resx, diagres, resy, diagres, resx, diagres, resy];

    let filter_size = ((max_dist * 2 + 1) * (max_dist * 2 + 1)) as usize;
    let mut minheap = BinaryHeap::with_capacity(filter_size);

    let backlink_dir = [4i8, 5, 6, 7, 0, 1, 2, 3];
    let mut backlink: Array2D<i8> =
        Array2D::new(rows, columns, -1, -2).expect("Error constructing backlinks Array2D");
    let mut encountered: Array2D<i8> =
        Array2D::new(rows, columns, 0, -1).expect("Error constructing encountered Array2D");
    let mut path_length: Array2D<i16> =
        Array2D::new(rows, columns, 0, -1).expect("Error constructing path_lenght Array2D");
    let mut scanned_cells = vec![];

    let (row, col, z) = search_array.pit;

    // Is it still a pit cell? It may have been solved during a previous depression solution.
    let mut flag = true;
    for n in 0..8 {
        let zn = search_array.array.get_value(row + dy[n], col + dx[n]);
        if zn < z && zn != nodata {
            // It has a lower non-nodata cell
            // Resolving some other pit cell resulted in a solution for this one.
            return BreachOutcome::PreviouslyBreached;
        }
    }
    if flag {
        // Perform the cost-accumulation operation.
        encountered.set_value(row, col, 1i8);
        if !minheap.is_empty() {
            minheap.clear();
        }
        minheap.push(GridCell {
            row: row,
            column: col,
            priority: 0f64,
        });
        scanned_cells.push((row, col));
        flag = true;
        while !minheap.is_empty() && flag {
            let cell2 = minheap.pop().expect("Error during pop operation.");
            let accum = cell2.priority;
            if accum > max_cost {
                // There isn't a breach channel cheap enough
                return BreachOutcome::NoPathFound;
            }
            let mut length = path_length.get_value(cell2.row, cell2.column);
            let mut zn = search_array.array.get_value(cell2.row, cell2.column);
            let cost1 = zn - z + length as f64 * small_num;
            for n in 0..8 {
                let mut cn = cell2.column + dx[n];
                let mut rn = cell2.row + dy[n];
                if encountered.get_value(rn, cn) != 1i8 {
                    scanned_cells.push((rn, cn));
                    // not yet encountered
                    let length_n = length + 1;
                    path_length.set_value(rn, cn, length_n);
                    backlink.set_value(rn, cn, backlink_dir[n]);
                    zn = search_array.array.get_value(rn, cn);
                    let mut zout = z - (length_n as f64 * small_num);
                    if zn > zout && zn != nodata {
                        let cost2 = zn - zout;
                        let new_cost = if minimize_dist {
                            accum + (cost1 + cost2) / 2f64 * cost_dist[n]
                        } else {
                            accum + cost2
                        };
                        encountered.set_value(rn, cn, 1i8);
                        if length_n <= max_dist as i16 {
                            minheap.push(GridCell {
                                row: rn,
                                column: cn,
                                priority: new_cost,
                            });
                        }
                    } else if zn <= zout || zn == nodata {
                        // We're at a cell that we can breach to
                        let mut path = vec![];
                        while flag {
                            // Find which cell to go to from here
                            if backlink.get_value(rn, cn) > -1i8 {
                                let b = backlink.get_value(rn, cn) as usize;
                                rn += dy[b];
                                cn += dx[b];
                                zn = search_array.array.get_value(rn, cn);
                                length = path_length.get_value(rn, cn);
                                zout = z - (length as f64 * small_num);
                                if zn > zout {
                                    // output.set_value(rn, cn, zout);
                                    // path.push((rn, cn, zout));
                                    let (raster_row, raster_column) =
                                        search_array.get_raster_index(rn, cn);
                                    path.push((raster_row, raster_column, zout));
                                }
                            } else {
                                flag = false;
                            }
                        }
                        return BreachOutcome::PathFound(path);
                    }
                }
            }
        }
        if flag {
            return BreachOutcome::NoPathFound;
        }
    }
    return BreachOutcome::NoPathFound;
}

#[derive(Debug)]
enum BreachOutcome {
    PreviouslyBreached,
    NoPathFound,
    PathFound(Vec<(isize, isize, f64)>),
}

struct SearchArray {
    pit: (isize, isize, f64),
    origin: (isize, isize),
    array: Array2D<f64>,
}

impl SearchArray {
    fn new(cell: (isize, isize, f64), raster: &Raster, max_dist: isize) -> Self {
        let offset = max_dist + 1;
        let (cell_row, cell_column, cell_value) = cell;
        let pit = (offset, offset, cell_value);

        let min_row = cell_row - offset;
        let min_column = cell_column - offset;
        let origin = (min_row, min_column);

        let mut array = Array2D::new(
            2 * offset + 1,
            2 * offset + 1,
            raster.configs.nodata,
            raster.configs.nodata,
        )
        .unwrap();

        for row in 0..array.rows {
            for column in 0..array.columns {
                let value = raster.get_value(row + min_row, column + min_column);
                array.set_value(row, column, value);
            }
        }

        Self { pit, origin, array }
    }

    fn get_raster_index(self: &Self, row: isize, column: isize) -> (isize, isize) {
        let (min_row, min_column) = self.origin;
        (row + min_row, column + min_column)
    }

    fn get_raster_cell(self: &Self, row: isize, column: isize, value: f64) -> (isize, isize, f64) {
        let (raster_row, raster_column) = self.get_raster_index(row, column);
        (raster_row, raster_column, value)
    }
}

#[derive(PartialEq, Debug)]
struct GridCell {
    row: isize,
    column: isize,
    priority: f64,
}

impl Eq for GridCell {}

impl PartialOrd for GridCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for GridCell {
    fn cmp(&self, other: &GridCell) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whitebox_raster::RasterConfigs;

    fn load_raster_from_file() -> Raster {
        const INPUT: &str =
            "/Users/dpower-mac/Downloads/USGS_OPR_MN_GoodhueCounty_2020_A20_6669_deflate.tif";
        let raster = Raster::new(INPUT, "r").unwrap();
        raster
    }

    fn write_raster_to_file(raster: &mut Raster) {
        const OUTPUT: &str = "/Users/dpower-mac/Downloads/test_least_cost_search.tif";
        let mut configs = raster.configs.clone();
        configs.data_type = whitebox_raster::DataType::F64;
        let array = raster.get_data_as_array2d();
        let mut output = Raster::initialize_from_array2d(OUTPUT, &configs, &array);
        let _ = output.write();
    }

    #[test]
    fn test_least_cost_search() {
        let input = load_raster_from_file();
        let mut output = input.clone();
        let num_procs = 8;
        let max_dist = 100;
        let max_cost = f64::INFINITY;
        let flat_increment = 0.000001;
        let minimize_dist = true;
        let num_threads = 4;

        let mut undefined_flow_cells =
            raise_pits(&Arc::new(input), &mut output, num_procs, flat_increment);
        let num_pits = undefined_flow_cells.len() as isize;

        let undefined_flow_cells2 = least_cost_search(
            &mut undefined_flow_cells,
            &mut output,
            max_dist,
            max_cost,
            flat_increment,
            minimize_dist,
            num_threads,
        );

        write_raster_to_file(&mut output);

        let num_unsolved = undefined_flow_cells2.len() as isize;
        let num_solved = num_pits - num_unsolved;
        // Original: 15625, 7
        assert_eq!((num_solved, num_unsolved), (15624, 8));
    }

    fn get_mock_raster() -> Raster {
        let configs: RasterConfigs = RasterConfigs {
            rows: 9,
            columns: 9,
            nodata: -9999.0,
            ..Default::default()
        };
        let mut raster = Raster::initialize_using_config("test.tif", &configs);
        raster.set_row_data(0, vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        raster.set_row_data(1, vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]);
        raster.set_row_data(2, vec![2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]);
        raster.set_row_data(3, vec![3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]);
        raster.set_row_data(4, vec![4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8]);
        raster.set_row_data(5, vec![5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]);
        raster.set_row_data(6, vec![6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8]);
        raster.set_row_data(7, vec![7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8]);
        raster.set_row_data(8, vec![8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8]);

        raster
    }

    #[test]
    fn test_search_array_new_full_size() {
        let raster = get_mock_raster();
        let central_cell = (4, 4, raster.get_value(4, 4));
        let search_array = SearchArray::new(central_cell, &raster, 4);

        let pit = search_array.pit;
        assert_eq!(pit.0, central_cell.0 + 1);
        assert_eq!(pit.1, central_cell.1 + 1);
        assert_eq!(pit.2, central_cell.2);
        assert_eq!(search_array.array.get_value(pit.0, pit.1), central_cell.2);
    }

    #[test]
    fn test_search_array_new_partial_size() {
        let raster = get_mock_raster();
        let central_cell = (3, 3, raster.get_value(3, 3));
        let search_array = SearchArray::new(central_cell, &raster, 1);

        // Define expected Array2D
        let mut expected =
            Array2D::new(5, 5, raster.configs.nodata, raster.configs.nodata).unwrap();
        expected.set_row_data(0, vec![1.1, 1.2, 1.3, 1.4, 1.5]);
        expected.set_row_data(1, vec![2.1, 2.2, 2.3, 2.4, 2.5]);
        expected.set_row_data(2, vec![3.1, 3.2, 3.3, 3.4, 3.5]);
        expected.set_row_data(3, vec![4.1, 4.2, 4.3, 4.4, 4.5]);
        expected.set_row_data(4, vec![5.1, 5.2, 5.3, 5.4, 5.5]);

        assert_eq!(search_array.array.rows, 5);
        assert_eq!(search_array.array.columns, 5);
        for row in 0..5 {
            for column in 0..5 {
                assert_eq!(
                    search_array.array.get_value(row, column),
                    expected.get_value(row, column)
                );
            }
        }
    }

    #[test]
    fn test_get_raster_cell() {
        let raster = get_mock_raster();
        let central_cell = (3, 3, raster.get_value(3, 3));
        let search_array = SearchArray::new(central_cell, &raster, 2);

        for row in 0..search_array.array.rows as isize {
            for column in 0..search_array.array.columns as isize {
                let from_array = search_array.get_raster_cell(
                    row,
                    column,
                    search_array.array.get_value(row, column),
                );
                let raster_value = raster.get_value(from_array.0, from_array.1);
                assert_eq!(from_array.2, raster_value);
            }
        }
    }

    #[test]
    fn test_get_raster_index() {
        let raster = get_mock_raster();
        let central_cell = (3, 3, raster.get_value(3, 3));
        let search_array = SearchArray::new(central_cell, &raster, 2);

        for row in 0..search_array.array.rows as isize {
            for column in 0..search_array.array.columns as isize {
                let (raster_row, raster_col) = search_array.get_raster_index(row, column);
                let raster_value = raster.get_value(raster_row, raster_col);
                let array_value = search_array.array.get_value(row, column);
                assert_eq!(array_value, raster_value);
            }
        }
    }
}
