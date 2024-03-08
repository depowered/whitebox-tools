use anyhow::Result;
use std::collections::BinaryHeap;

use std::cmp::Ordering::{self, Equal};
use whitebox_common::structures::Array2D;
use whitebox_raster::Raster;

fn least_cost_search(
    undefined_flow_cells: &mut Vec<(isize, isize, f64)>,
    raster: Raster,
    max_dist: isize,
    max_cost: f64,
    flat_increment: f64,
    minimize_dist: bool,
) -> Result<(usize, usize)> {
    let mut output = raster;
    let small_num = flat_increment;

    /* Vec is a stack and so if we want to pop the values from lowest to highest, we need to sort
    them from highest to lowest. */
    undefined_flow_cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Equal));
    let mut undefined_flow_cells2 = vec![];

    let mut num_solved: usize = 0;
    let mut num_unsolved: usize = 0;
    while let Some(cell) = undefined_flow_cells.pop() {
        let path = try_breach(
            cell,
            &mut output,
            max_cost,
            small_num,
            minimize_dist,
            max_dist,
        );

        match path {
            Some(cells) => {
                num_solved += 1;
                if cells.len() > 0 {
                    for cell in cells {
                        let (r, c, z) = cell;
                        output.set_value(r, c, z);
                    }
                }
            }
            None => {
                undefined_flow_cells2.push(cell); // Add it to the list for the next iteration
                num_unsolved += 1;
            }
        }
    }
    Ok((num_solved, num_unsolved))
}

fn try_breach(
    cell: (isize, isize, f64),
    output: &mut Raster,
    max_cost: f64,
    small_num: f64,
    minimize_dist: bool,
    max_dist: isize,
) -> Option<Vec<(isize, isize, f64)>> {
    let rows = output.configs.rows as isize;
    let columns = output.configs.columns as isize;
    let nodata = output.configs.nodata;

    let dx = [1, 1, 1, 0, -1, -1, -1, 0];
    let dy = [-1, 0, 1, 1, 1, 0, -1, -1];
    let resx = output.configs.resolution_x;
    let resy = output.configs.resolution_y;
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

    // let mut undefined_flow_cells2 = vec![];

    let row = cell.0;
    let col = cell.1;
    let z = output.get_value(row, col);

    // Is it still a pit cell? It may have been solved during a previous depression solution.
    let mut flag = true;
    for n in 0..8 {
        let zn = output.get_value(row + dy[n], col + dx[n]);
        if zn < z && zn != nodata {
            // It has a lower non-nodata cell
            // Resolving some other pit cell resulted in a solution for this one.
            let empty: Vec<(isize, isize, f64)> = vec![];
            return Some(empty);
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
                return None;
            }
            let mut length = path_length.get_value(cell2.row, cell2.column);
            let mut zn = output.get_value(cell2.row, cell2.column);
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
                    zn = output.get_value(rn, cn);
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
                        let mut breach_path = vec![];
                        while flag {
                            // Find which cell to go to from here
                            if backlink.get_value(rn, cn) > -1i8 {
                                let b = backlink.get_value(rn, cn) as usize;
                                rn += dy[b];
                                cn += dx[b];
                                zn = output.get_value(rn, cn);
                                length = path_length.get_value(rn, cn);
                                zout = z - (length as f64 * small_num);
                                if zn > zout {
                                    // output.set_value(rn, cn, zout);
                                    breach_path.push((rn, cn, zout));
                                }
                            } else {
                                flag = false;
                            }
                        }
                        return Some(breach_path);
                    }
                }
            }
        }
    }
    return None;
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
    use whitebox_raster::DataType;

    use super::super::algorithms::{
        calculate_flat_increment, get_raw_pits, raise_pits, raster_to_matrix,
    };
    use super::*;

    fn load_raster() -> Raster {
        const INPUT: &str =
            "/Users/dpower-mac/Downloads/USGS_OPR_MN_GoodhueCounty_2020_A20_6669_deflate.tif";
        let raster = Raster::new(INPUT, "r").unwrap();
        raster
    }

    fn gather_pits() -> Vec<(isize, isize, f64)> {
        // Load the input raster into memory
        let raster = load_raster();

        // Clone the input raster config for use in constructing the output raster later
        // Set the output datatype to 64-bit float
        let mut configs = raster.configs.clone();
        configs.data_type = DataType::F64;

        // Calculate a small increment used to ensure positive flow through breach cells from the raster
        // metadata
        let flat_increment = calculate_flat_increment(&configs);

        // Convert the raster into a pathfinding::Matrix
        let mut matrix = raster_to_matrix(raster, 1).unwrap();

        // Find all the pits and prepare them for beaching by raising their value to just
        // below the value of their lowest neighbor
        let raw_pits = get_raw_pits(&matrix);
        let raised_pits = raise_pits(&mut matrix, raw_pits, flat_increment);

        let mut undefined_flow_cells = vec![];
        for raised_pit in raised_pits {
            let data = raised_pit.get_data();
            undefined_flow_cells.push((data.row, data.column, data.value.into_inner()))
        }
        undefined_flow_cells
    }

    #[test]
    fn test_least_cost_search() {
        let mut undefined_flow_cells = gather_pits();
        let mut raster = load_raster();
        let max_dist = 100;
        let max_cost = f64::INFINITY;
        let flat_increment = 0.000001;
        let minimize_dist = true;

        let (num_solved, num_unsolved) = least_cost_search(
            &mut undefined_flow_cells,
            raster,
            max_dist,
            max_cost,
            flat_increment,
            minimize_dist,
        )
        .unwrap();

        assert_eq!(num_solved, 15625);
        assert_eq!(num_unsolved, 7);
    }
}
