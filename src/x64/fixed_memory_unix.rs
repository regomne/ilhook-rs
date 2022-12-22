use super::{cmp, HookError};
use lazy_static::lazy_static;
use regex::Regex;
use std::format;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use libc::{
    __errno_location, c_void, mmap, munmap, sysconf, MAP_ANONYMOUS, MAP_FIXED_NOREPLACE,
    MAP_PRIVATE,
};

pub(super) struct FixedMemory {
    pub addr: u64,
    pub len: u32,
}

impl Drop for FixedMemory {
    fn drop(&mut self) {
        unsafe { munmap(self.addr as *mut c_void, self.len as usize) };
    }
}

impl FixedMemory {
    pub fn allocate(hook_addr: u64) -> Result<Self, HookError> {
        let bound = Bound::new(hook_addr);
        let block = MemoryLayout::read_self_mem_layout()?.find_memory_with_bound(&bound)?;
        let len = block.end - block.begin;
        let mut addr = unsafe {
            mmap(
                block.begin as *mut c_void,
                len as usize,
                7,
                MAP_PRIVATE | MAP_FIXED_NOREPLACE | MAP_ANONYMOUS,
                -1,
                0,
            )
        } as usize as u64;
        if addr == u64::MAX {
            return Err(HookError::MemoryProtect(
                87210000 + unsafe { *(__errno_location()) },
            ));
        }
        // If kernel doesn't support MAP_FIXED_NOREPLACE
        if addr == u64::MAX && unsafe { *(__errno_location()) } == 95 {
            addr = unsafe {
                mmap(
                    block.begin as *mut c_void,
                    len as usize,
                    7,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            } as usize as u64;
        }
        match addr {
            u64::MAX => Err(HookError::MemoryProtect(
                unsafe { *(__errno_location()) } as u32
            )),
            x if x == block.begin || (x >= bound.min && x + len <= bound.max) => Ok(Self {
                addr,
                len: len as u32,
            }),
            _ => Err(HookError::MemoryProtect(0)),
        }
    }
}

struct MemoryLayout(Vec<MemoryBlock>);

impl MemoryLayout {
    fn read_self_mem_layout() -> Result<Self, HookError> {
        let maps = File::open(format!("/proc/{}/maps", process::id()))?;
        BufReader::new(maps)
            .lines()
            .map(|line| {
                line.map_err(|_| HookError::MemoryLayoutFormat)
                    .and_then(MemoryBlock::from_string)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
    }

    fn find_memory_with_bound(&self, bnd: &Bound) -> Result<MemoryBlock, HookError> {
        //@todo fix: find memory block from middle to edge
        let page_size = unsafe { sysconf(30) } as u64; //_SC_PAGESIZE == 30
        let blocks = &self.0;
        if blocks.is_empty() {
            return Err(HookError::MemoryAllocation);
        }
        // test the first block
        if blocks[0].begin > page_size * 2 && bnd.min <= page_size {
            return Ok(MemoryBlock {
                begin: page_size,
                end: page_size * 2,
            });
        }
        for i in 1..blocks.len() {
            let gap = blocks[i].begin - blocks[i - 1].end;
            if gap >= page_size && blocks[i - 1].end >= bnd.min && blocks[i].begin < bnd.max {
                return Ok(MemoryBlock {
                    begin: blocks[i - 1].end,
                    end: blocks[i - 1].end + page_size,
                });
            }
        }
        Err(HookError::MemoryAllocation)
    }
}

#[derive(Debug)]
struct MemoryBlock {
    begin: u64,
    end: u64,
}
impl MemoryBlock {
    fn from_string(s: String) -> Result<Self, HookError> {
        lazy_static! {
            static ref RE: Regex = Regex::new("^([a-fA-F0-9]+)-([a-fA-F0-9]+)").unwrap();
        }
        //let RE = Regex::new("").unwrap();
        RE.captures(&s)
            .ok_or(HookError::MemoryLayoutFormat)
            .and_then(|cap| {
                let begin = cap.get(1).unwrap().as_str();
                let end = cap.get(2).unwrap().as_str();
                Ok(Self {
                    begin: u64::from_str_radix(begin, 16).or(Err(HookError::MemoryLayoutFormat))?,
                    end: u64::from_str_radix(end, 16).or(Err(HookError::MemoryLayoutFormat))?,
                })
            })
    }
}

struct Bound {
    min: u64,
    max: u64,
}

impl Bound {
    fn new(init_addr: u64) -> Self {
        Self {
            min: init_addr.saturating_sub(i32::MAX as u64),
            max: init_addr.saturating_add(i32::MAX as u64),
        }
    }

    fn _to_new(self, dest: u64) -> Self {
        Self {
            min: cmp::max(self.min, dest.saturating_sub(i32::MAX as u64)),
            max: cmp::min(self.max, dest.saturating_add(i32::MAX as u64)),
        }
    }

    fn _check(&self) -> Result<(), HookError> {
        if self.min > self.max {
            Err(HookError::InvalidParameter)
        } else {
            Ok(())
        }
    }
}
