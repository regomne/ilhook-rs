use super::*;

#[cfg(windows)]
use winapi::shared::minwindef::LPVOID;
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::memoryapi::{VirtualAlloc, VirtualFree, VirtualQuery};
use winapi::um::winnt::MEM_RESERVE;
#[cfg(windows)]
use winapi::um::winnt::{
    MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, PAGE_EXECUTE_READWRITE,
};

enum QueryResult {
    Success(u64),
    NotUsable(u64, u64),
    Fail,
}

pub(super) struct FixedMemory {
    pub addr: u64,
    pub len: u32,
}

impl Drop for FixedMemory {
    fn drop(&mut self) {
        unsafe { VirtualFree(self.addr as LPVOID, 0, MEM_RELEASE) };
    }
}
impl FixedMemory {
    pub fn allocate(hook_addr: u64, rel_tbl: &Vec<RelocEntry>) -> Result<Self, HookError> {
        let bound = rel_tbl
            .iter()
            .fold(Bound::new(hook_addr), |b, ent| b.to_new(ent.dest_addr));
        bound.check()?;
        let addr = FixedMemory::allocate_internal(&bound)?;

        Ok(Self { addr, len: 4096 })
    }

    fn query_and_alloc(addr: u64) -> QueryResult {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { MaybeUninit::uninit().assume_init() };
        let ret = unsafe {
            VirtualQuery(
                addr as LPVOID,
                &mut mbi,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if ret == 0 {
            QueryResult::Fail
        } else if mbi.State == MEM_FREE && mbi.RegionSize >= 4096 {
            let mem = unsafe {
                VirtualAlloc(
                    mbi.BaseAddress,
                    4096,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                )
            };
            if mem == std::ptr::null_mut() {
                let err = unsafe { GetLastError() };
                println!("reason:{}", err);
                QueryResult::NotUsable(mbi.BaseAddress as usize as u64, mbi.RegionSize as u64)
            } else {
                QueryResult::Success(mem as usize as u64)
            }
        } else {
            QueryResult::NotUsable(mbi.BaseAddress as usize as u64, mbi.RegionSize as u64)
        }
    }

    fn allocate_internal(bnd: &Bound) -> Result<u64, HookError> {
        let nb = bnd.clone();
        let mut cur_addr = nb.middle();
        while cur_addr < nb.max {
            match FixedMemory::query_and_alloc(cur_addr) {
                QueryResult::Success(addr) => {
                    return Ok(addr);
                }
                QueryResult::NotUsable(_, size) => {
                    cur_addr += if size > 0 { size } else { 4096 };
                }
                QueryResult::Fail => {
                    return Err(HookError::MemoryAllocation);
                }
            }
        }
        cur_addr = nb.middle();
        while cur_addr > nb.min {
            match FixedMemory::query_and_alloc(cur_addr) {
                QueryResult::Success(addr) => {
                    return Ok(addr);
                }
                QueryResult::NotUsable(base, _) => {
                    cur_addr = base.checked_sub(4096).unwrap_or(0);
                }
                QueryResult::Fail => {
                    return Err(HookError::MemoryAllocation);
                }
            }
        }
        Err(HookError::MemoryAllocation)
    }
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
