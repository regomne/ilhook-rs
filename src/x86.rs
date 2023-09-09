mod trampoline;
#[cfg(target_arch = "x86")]
mod tests;

use iced_x86::{Decoder, DecoderOptions, Instruction};

use std::slice;

use crate::callbacks::*;
use crate::err::HookError;
use crate::utils::{MemoryProtectGuard, ThreadSuspendingGuard};
use trampoline::Trampoline;

const MAX_INST_LEN: usize = 15;
const JMP_INST_SIZE: usize = 5;

/// This is the routine used in a `jmp-back hook`, which means the EIP will jump back to the
/// original position after the routine has finished running.
///
/// # Arguments
///
/// * `regs` - The registers
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
pub type JmpBackRoutine = unsafe extern "cdecl" fn(regs: *mut Registers, user_data: usize);

/// This is the routine used in a `function hook`, which means the routine will replace the
/// original function and the EIP will `retn` directly instead of jumping back.
/// Note that the address being hooked must be the start of a function.
///
/// # Parameters
///
/// * `regs` - The registers.
/// * `ori_func_ptr` - The original function pointer. Call this after converting it to the original function type.
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
///
/// # Return value
///
/// Returns the new return value of the replaced function.
pub type RetnRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize) -> usize;

/// This is the routine used in a `jmp-addr hook`, which means the EIP will jump to the specified
/// address after the routine has finished running.
///
/// # Parameters
///
/// * `regs` - The registers.
/// * `ori_func_ptr` - The original function pointer. Call this after converting it to the original function type.
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
pub type JmpToAddrRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize);

/// This is the routine used in a `jmp-ret hook`, which means the EIP will jump to the return
/// value of the routine.
///
/// # Parameters
///
/// * `regs` - The registers.
/// * `ori_func_ptr` - The original function pointer. Call this after converting it to the original function type.
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
///
/// # Return value
///
/// Returns the address you want to jump to.
pub type JmpToRetRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize) -> usize;

/// The hooking type.
pub enum HookType {
    /// Used in a jmp-back hook
    JmpBack(JmpBackRoutine),

    /// Used in a function hook. The first element is the mnemonic of the `retn`
    /// instruction.
    Retn(usize, RetnRoutine),

    /// Used in a jmp-addr hook. The first element is the destination address
    JmpToAddr(usize, JmpToAddrRoutine),

    /// Used in a jmp-ret hook.
    JmpToRet(JmpToRetRoutine),
}

/// The common registers.
#[repr(C)]
#[derive(Debug)]
pub struct Registers {
    /// The flags register.
    pub eflags: u32,
    /// The edi register.
    pub edi: u32,
    /// The esi register.
    pub esi: u32,
    /// The ebp register.
    pub ebp: u32,
    /// The esp register.
    pub esp: u32,
    /// The ebx register.
    pub ebx: u32,
    /// The edx register.
    pub edx: u32,
    /// The ecx register.
    pub ecx: u32,
    /// The eax register.
    pub eax: u32,
}

impl Registers {
    /// Get the value by the index from register `esp`.
    ///
    /// # Parameters
    ///
    /// * cnt - The index of the arguments.
    ///
    /// # Safety
    ///
    /// Process may crash if register `esp` does not point to a valid stack.
    #[must_use]
    pub unsafe fn get_arg(&self, cnt: usize) -> u32 {
        *((self.esp as usize + cnt * 4) as *mut u32)
    }
}

/// The entry struct in ilhook.
/// Please read the main doc to view usage.
pub struct Hooker<'a> {
    addr: usize,
    hook_type: HookType,
    options: HookOptions<'a>,
    user_data: usize,
}

/// Hook options
pub struct HookOptions<'a> {
    pub thread_operating_cb: Option<&'a dyn ThreadOperatingCallback>,
    pub code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

impl<'a> Default for HookOptions<'a> {
    fn default() -> Self {
        HookOptions {
            thread_operating_cb: None,
            code_protect_cb: Some(&DEFAULT_CODE_PROTECT_MODIFYING_CALLBACK),
        }
    }
}

/// The hook result returned by `Hooker::hook`.
pub struct HookPoint<'a> {
    addr: usize,
    #[allow(dead_code)]
    trampoline: Trampoline<'a>,
    origin: Vec<u8>,
    thread_operating_cb: Option<&'a dyn ThreadOperatingCallback>,
    code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

#[cfg(not(target_arch = "x86"))]
fn env_lock() {
    panic!("This crate should only be used in arch x86_32!")
}
#[cfg(target_arch = "x86")]
fn env_lock() {}

impl<'a> Hooker<'a> {
    /// Create a new Hooker.
    ///
    /// # Parameters
    ///
    /// * `addr` - The being-hooked address.
    /// * `hook_type` - The hook type and callback routine.
    /// * `thread_cb` - The callbacks before and after hooking.
    /// * `flags` - Hook flags
    #[must_use]
    pub fn new(
        addr: usize,
        hook_type: HookType,
        options: HookOptions<'a>,
        user_data: usize,
    ) -> Self {
        env_lock();
        Self {
            addr,
            hook_type,
            user_data,
            options,
        }
    }

    /// Consumes self and execute hooking. Return the `HookPoint`.
    ///
    /// # Safety
    ///
    /// Process may crash (instead of panic!) if:
    ///
    /// 1. addr is not an accessible memory address, or is not long enough.
    /// 2. addr points to an incorrect position. (At the middle of an instruction, or where after it other instructions may jump to)
    /// 3. Wrong Retn-val if `hook_type` is `HookType::Retn`. i.e. A `cdecl` function with non-zero retn-val, or a `stdcall` function with wrong retn-val.
    /// 4. Set `NOT_MODIFY_MEMORY_PROTECT` where it should not be set.
    /// 5. hook or unhook from 2 or more threads at the same time without `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Because of memory protection colliding.
    /// 6. Other unpredictable errors.
    pub unsafe fn hook(self) -> Result<HookPoint<'a>, HookError> {
        self.hook_internal()
    }

    fn hook_internal(self) -> Result<HookPoint<'a>, HookError> {
        let (moving_insts, origin) = get_moving_insts(self.addr)?;
        let mut trampoline = Trampoline::new(&self.options.code_protect_cb);
        trampoline.generate(
            self.addr,
            self.hook_type,
            &moving_insts,
            origin.len() as u8,
            self.user_data,
        )?;
        MemoryProtectGuard::new(&self.options.code_protect_cb, self.addr, JMP_INST_SIZE).run(||{
            ThreadSuspendingGuard::new(&self.options.thread_operating_cb).run(||{
                modify_jmp(self.addr, trampoline.get_addr())
            })
        })?;
        Ok(HookPoint {
            addr: self.addr,
            trampoline,
            origin,
            thread_operating_cb: self.options.thread_operating_cb,
            code_protect_cb: self.options.code_protect_cb,
        })
    }
}

impl<'a> HookPoint<'a> {
    /// Consume self and unhook the address.
    pub unsafe fn unhook(self) -> Result<(), HookError> {
        self.unhook_internal()
    }

    fn unhook_internal(&self) -> Result<(), HookError> {
        MemoryProtectGuard::new(&self.code_protect_cb, self.addr, JMP_INST_SIZE).run(|| {
            ThreadSuspendingGuard::new(&self.thread_operating_cb).run(|| {
                recover_jmp(self.addr, &self.origin[0..JMP_INST_SIZE]);
                Ok(())
            })
        })
    }
}

// When the HookPoint drops, it should unhook automatically.
impl<'a> Drop for HookPoint<'a> {
    fn drop(&mut self) {
        self.unhook_internal().unwrap_or_default();
    }
}

fn get_moving_insts(addr: usize) -> Result<(Vec<Instruction>, Vec<u8>), HookError> {
    let code_slice =
        unsafe { slice::from_raw_parts(addr as *const u8, MAX_INST_LEN * JMP_INST_SIZE) };
    let mut decoder = Decoder::new(32, code_slice, DecoderOptions::NONE);
    decoder.set_ip(addr as u64);

    let mut total_bytes = 0;
    let mut ori_insts: Vec<Instruction> = vec![];
    for inst in &mut decoder {
        if inst.is_invalid() {
            return Err(HookError::Disassemble);
        }
        ori_insts.push(inst);
        total_bytes += inst.len();
        if total_bytes >= JMP_INST_SIZE {
            break;
        }
    }

    Ok((ori_insts, code_slice[0..decoder.position()].into()))
}

fn modify_jmp(dest_addr: usize, trampoline_addr: usize) -> Result<(), HookError> {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, JMP_INST_SIZE) };
    // jmp trampoline_addr
    buf[0] = 0xe9;
    let rel_off = trampoline_addr as i32 - (dest_addr as i32 + 5);
    buf[1..5].copy_from_slice(&rel_off.to_le_bytes());
    Ok(())
}

fn recover_jmp(dest_addr: usize, origin: &[u8]) {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, origin.len()) };
    // jmp trampoline_addr
    buf.copy_from_slice(origin);
}
