use super::{cmp, HookError};

use core::ffi::c_void;
use std::mem::{size_of, MaybeUninit};
use windows_sys::Win32::Foundation::{GetLastError, ERROR_INVALID_PARAMETER};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualQuery};
use windows_sys::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE,
    PAGE_EXECUTE_READWRITE,
};

enum QueryResult {
    Success(u64),
    NotUsable(u64, u64),
    OverLimit,
    Fail(u32),
}

pub(super) struct FixedMemory {
    pub addr: u64,
    pub len: u32,
}

impl Drop for FixedMemory {
    fn drop(&mut self) {
        unsafe { VirtualFree(self.addr as *mut c_void, 0, MEM_RELEASE) };
    }
}
impl FixedMemory {
    pub fn allocate(hook_addr: u64) -> Result<Self, HookError> {
        let addr = FixedMemory::allocate_internal(&Bound::new(hook_addr))?;

        Ok(Self { addr, len: 4096 })
    }

    fn query_and_alloc(addr: u64) -> QueryResult {
        #[allow(invalid_value)]
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { MaybeUninit::uninit().assume_init() };
        let ret = unsafe {
            VirtualQuery(
                addr as *mut c_void,
                &mut mbi,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if ret == 0 {
            let last_err = unsafe { GetLastError() };
            if last_err == ERROR_INVALID_PARAMETER {
                // ERROR_INVALID_PARAMETER means lpAddress specifies an address above
                // the highest memory address accessible to the process. (from MSDN)
                QueryResult::OverLimit
            } else {
                QueryResult::Fail(last_err)
            }
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
                QueryResult::NotUsable(mbi.BaseAddress as usize as u64, mbi.RegionSize as u64)
            } else {
                QueryResult::Success(mem as usize as u64)
            }
        } else {
            QueryResult::NotUsable(mbi.BaseAddress as usize as u64, mbi.RegionSize as u64)
        }
    }

    fn allocate_internal(bnd: &Bound) -> Result<u64, HookError> {
        let mut cur_addr = bnd.middle();
        while cur_addr < bnd.max {
            match FixedMemory::query_and_alloc(cur_addr) {
                QueryResult::Success(addr) => {
                    return Ok(addr);
                }
                QueryResult::NotUsable(_, size) => {
                    cur_addr += if size > 0 { size } else { 4096 };
                }
                QueryResult::OverLimit => {
                    break;
                }
                QueryResult::Fail(e) => {
                    return Err(HookError::MemoryAllocation(e));
                }
            }
        }
        cur_addr = bnd.middle();
        while cur_addr > bnd.min {
            match FixedMemory::query_and_alloc(cur_addr) {
                QueryResult::Success(addr) => {
                    return Ok(addr);
                }
                QueryResult::NotUsable(base, _) => {
                    cur_addr = base.saturating_sub(4096);
                }
                QueryResult::Fail(e) => {
                    return Err(HookError::MemoryAllocation(e));
                }
                QueryResult::OverLimit => {
                    return Err(HookError::MemoryAllocation(0));
                }
            }
        }
        Err(HookError::MemorySearching)
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

    fn middle(&self) -> u64 {
        self.min / 2 + self.max / 2
    }
}
