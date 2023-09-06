use crate::HookError;

#[cfg(windows)]
use core::ffi::c_void;
#[cfg(unix)]
use libc::{__errno_location, c_void, mprotect, sysconf};
#[cfg(windows)]
use windows_sys::Win32::Foundation::GetLastError;
#[cfg(windows)]
use windows_sys::Win32::System::Memory::VirtualProtect;

/// The trait which is called before and after the modifying of the `jmp` instruction.
/// Usually is used to suspend and resume all other threads, to avoid instruction colliding.
pub trait ThreadCallback {
    /// the callback before modifying `jmp` instruction, should return true if success.
    fn pre(&self) -> bool;
    /// the callback after modifying `jmp` instruction
    fn post(&self);
}

pub trait CodeProtectModifyingCallback {
    fn add_writable_to_protect(&self, addr: usize, len: usize) -> Result<u64, HookError>;
    fn add_executable_to_protect(&self, addr: usize, len: usize) -> Result<u64, HookError>;
    fn recover_protect(&self, addr: usize, len: usize, old_prot: u64);
}

pub struct InternalCodeProtectModifyingCallback {}
impl CodeProtectModifyingCallback for InternalCodeProtectModifyingCallback {
    fn add_writable_to_protect(&self, addr: usize, len: usize) -> Result<u64, HookError> {
        modify_mem_protect_to_rwe(addr, len)
    }

    fn add_executable_to_protect(&self, addr: usize, len: usize) -> Result<u64, HookError> {
        modify_mem_protect_to_rwe(addr, len)
    }

    fn recover_protect(&self, addr: usize, len: usize, old_prot: u64) {
        recover_mem_protect(addr, len, old_prot)
    }
}

pub const DefaultCodeProtectModifyingCallback: InternalCodeProtectModifyingCallback =
    InternalCodeProtectModifyingCallback {};

#[cfg(windows)]
fn modify_mem_protect_to_rwe(addr: usize, len: usize) -> Result<u64, HookError> {
    let mut old_prot: u32 = 0;
    let old_prot_ptr = std::ptr::addr_of_mut!(old_prot);
    // PAGE_EXECUTE_READWRITE = 0x40
    let ret = unsafe { VirtualProtect(addr as *const c_void, len, 0x40, old_prot_ptr) };
    if ret == 0 {
        Err(HookError::MemoryProtect(unsafe { GetLastError() }))
    } else {
        Ok(old_prot as u64)
    }
}

#[cfg(unix)]
fn modify_mem_protect_to_rwe(addr: usize, len: usize) -> Result<u64, HookError> {
    let page_size = unsafe { sysconf(30) }; //_SC_PAGESIZE == 30
    if len > page_size.try_into().unwrap() {
        Err(HookError::InvalidParameter)
    } else {
        //(PROT_READ | PROT_WRITE | PROT_EXEC) == 7
        let ret = unsafe {
            mprotect(
                (addr & !(page_size as usize - 1)) as *mut c_void,
                page_size as usize,
                7,
            )
        };
        if ret != 0 {
            let err = unsafe { *(__errno_location()) };
            Err(HookError::MemoryProtect(err as u32))
        } else {
            // it's too complex to get the original memory protection
            Ok(7)
        }
    }
}

#[cfg(windows)]
fn recover_mem_protect(addr: usize, len: usize, old: u64) {
    let mut old_prot: u32 = 0;
    let old_prot_ptr = std::ptr::addr_of_mut!(old_prot);
    unsafe { VirtualProtect(addr as *const c_void, len, old as u32, old_prot_ptr) };
}

#[cfg(unix)]
fn recover_mem_protect(addr: usize, _: usize, old: u64) {
    let page_size = unsafe { sysconf(30) }; //_SC_PAGESIZE == 30
    unsafe {
        mprotect(
            (addr & !(page_size as usize - 1)) as *mut c_void,
            page_size as usize,
            old as i32,
        )
    };
}
