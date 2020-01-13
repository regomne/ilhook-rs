use super::*;
use std::format;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

struct MemoryBlock {
    addr: u64,
    len: u64,
}
impl MemoryBlock {
    fn from_string(s: String) -> Result<Self, HookError> {
        Err(HookError::MemoryLayoutFormat)
    }
}

fn read_self_mem_layout() -> Result<Vec<MemoryBlock>, HookError> {
    let maps_path = format!("/proc/{}/maps", process::id());
    let maps = File::open(maps_path)?;
    let reader = BufReader::new(maps);

    reader
        .lines()
        .map(|line| {
            line.map_err(|e| HookError::MemoryLayoutFormat)
                .and_then(|s| MemoryBlock::from_string(s))
        })
        .collect()
}

#[derive(Clone)]
struct Bound {
    min: u64,
    max: u64,
}

impl Bound {
    fn new(init_addr: u64) -> Self {
        Self {
            min: init_addr.checked_sub(i32::max_value() as u64).unwrap_or(0),
            max: init_addr
                .checked_add(i32::max_value() as u64)
                .unwrap_or(u64::max_value()),
        }
    }

    fn to_new(self, dest: u64) -> Self {
        Self {
            min: cmp::max(
                self.min,
                dest.checked_sub(i32::max_value() as u64).unwrap_or(0),
            ),
            max: cmp::min(
                self.max,
                dest.checked_add(i32::max_value() as u64)
                    .unwrap_or(u64::max_value()),
            ),
        }
    }

    fn check(&self) -> Result<(), HookError> {
        if self.min > self.max {
            Err(HookError::InvalidParameter)
        } else {
            Ok(())
        }
    }

    fn middle(&self) -> u64 {
        self.min / 2 + self.max / 2
    }
}
