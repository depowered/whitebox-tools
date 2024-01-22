use super::structures::{DEMCellType, PitStatus};
use std::collections::BinaryHeap;
use whitebox_raster::Raster;

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
