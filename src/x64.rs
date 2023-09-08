mod move_inst;
#[cfg(target_arch = "x86_64")]
mod tests;
mod trampoline;

use std::cell::RefCell;
use std::rc::Rc;
use std::slice;

use iced_x86::{Decoder, DecoderOptions, Instruction};

use crate::callbacks::*;
use crate::utils::{MemoryProtectGuard, ThreadSuspendingGuard};
use crate::HookError;
use trampoline::*;

const MAX_INST_LEN: usize = 15;

/// This is the routine used in a `jmp-back hook`, which means the RIP will jump back to the
/// original position after the routine has finished running.
///
/// # Parameters
///
/// * `regs` - The registers.
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
pub type JmpBackRoutine = unsafe extern "win64" fn(regs: *mut Registers, user_data: usize);

/// This is the routine used in a `function hook`, which means the routine will replace the
/// original function and the RIP will `retn` directly instead of jumping back.
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
    unsafe extern "win64" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize) -> usize;

/// This is the routine used in a `jmp-addr hook`, which means the RIP will jump to the specified
/// address after the routine has finished running.
///
/// # Parameters
///
/// * `regs` - The registers.
/// * `ori_func_ptr` - The original function pointer. Call this after converting it to the original function type.
/// * `user_data` - User data that was previously passed to [`Hooker::new`].
pub type JmpToAddrRoutine =
    unsafe extern "win64" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize);

/// This is the routine used in a `jmp-ret hook`, which means the RIP will jump to the return
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
    unsafe extern "win64" fn(regs: *mut Registers, ori_func_ptr: usize, user_data: usize) -> usize;

/// The hooking type.
pub enum HookType {
    /// Used in a jmp-back hook
    JmpBack(JmpBackRoutine),

    /// Used in a function hook
    Retn(RetnRoutine),

    /// Used in a jmp-addr hook. The first element is the destination address
    JmpToAddr(usize, JmpToAddrRoutine),

    /// Used in a jmp-ret hook.
    JmpToRet(JmpToRetRoutine),
}

/// Jmp type that the `jmp` instruction use.
pub enum JmpType {
    /// Rip relative jmp, use 14 bytes.
    /// `jmp qword ptr [rip+0]`
    RipRelative,

    /// Direct jmp. Use only 5 bytes. You need to specify the trampoline buffer manually.
    /// Address of the trampoline must be in +/- 2GB of the hooking address.
    /// Length of the trampoline should be at least 1024 bytes, and the memory protect must be RWE.
    Direct(*mut u8, usize),

    /// Use 2 jmp instructions to jump. First of it uses only 5 bytes,
    /// You need to look for and specify a code address to place the second jmp instruction.
    /// The address must be in +/- 2GB of the hooking address and the length must be not less than
    /// 14 bytes. Usually you may try to look for a code gap from the code section of the PE/ELF.
    /// `jmp _SecondJmp`
    /// `_SecondJmp:`
    /// `jmp qword ptr [rip+0]`
    DirectWithRipRelative(usize),
}

impl JmpType {
    fn get_jmp_inst_size(&self) -> usize {
        match &self {
            JmpType::RipRelative => 14,
            JmpType::Direct(_, _) => 5,
            JmpType::DirectWithRipRelative(_) => 5,
        }
    }
}

/// The common registers.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Registers {
    /// The xmm0 register
    pub xmm0: u128,
    /// The xmm1 register
    pub xmm1: u128,
    /// The xmm2 register
    pub xmm2: u128,
    /// The xmm3 register
    pub xmm3: u128,
    /// The r15 register
    pub r15: u64,
    /// The r14 register
    pub r14: u64,
    /// The r13 register
    pub r13: u64,
    /// The r12 register
    pub r12: u64,
    /// The r11 register
    pub r11: u64,
    /// The r10 register
    pub r10: u64,
    /// The r9 register
    pub r9: u64,
    /// The r8 register
    pub r8: u64,
    /// The rbp register
    pub rbp: u64,
    /// The rdi register
    pub rdi: u64,
    /// The rsi register
    pub rsi: u64,
    /// The rdx register
    pub rdx: u64,
    /// The rcx register
    pub rcx: u64,
    /// The rbx register
    pub rbx: u64,
    /// The rsp register
    pub rsp: u64,
    /// The flags register
    pub rflags: u64,
    /// Unused var
    pub _no_use: u64,
    /// The rax register
    pub rax: u64,
}

impl Registers {
    /// Get the value by index.
    ///
    /// # Parameters
    ///
    /// * cnt - The index of the arguments.
    ///
    /// # Safety
    ///
    /// Process may crash if register `rsp` does not point to a valid stack.
    #[must_use]
    pub unsafe fn get_stack(&self, cnt: usize) -> u64 {
        *((self.rsp as usize + cnt * 8) as *mut u64)
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
    pub first_jmp_type: JmpType,
    pub thread_operating_cb: Option<&'a dyn ThreadOperatingCallback>,
    pub code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

impl<'a> Default for HookOptions<'a> {
    fn default() -> Self {
        HookOptions {
            first_jmp_type: JmpType::RipRelative,
            thread_operating_cb: None,
            code_protect_cb: Some(&DEFAULT_CODE_PROTECT_MODIFYING_CALLBACK),
        }
    }
}

/// The hook result returned by `Hooker::hook`.
pub struct HookPoint<'a> {
    addr: usize,
    trampoline: Trampoline<'a>,
    origin: Vec<u8>,
    jmp_inst_size: usize,
    thread_operating_cb: Option<&'a dyn ThreadOperatingCallback>,
    code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

#[cfg(not(target_arch = "x86_64"))]
fn env_lock() {
    panic!("This crate should only be used in arch x86_32!")
}
#[cfg(target_arch = "x86_64")]
fn env_lock() {}

impl<'a> Hooker<'a> {
    /// Create a new Hooker.
    ///
    /// # Parameters
    ///
    /// * `addr` - The being-hooked address.
    /// * `hook_type` - The hook type and callback routine.
    /// * `options` - Hook options, see [`HookOptions`]
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
            options,
            user_data,
        }
    }

    /// Consumes self and do hooking. Return the [`HookPoint`].
    ///
    /// # Safety
    ///
    /// Process may crash (instead of panic!) if:
    ///
    /// 1. addr is not an accessible memory address, or is not long enough.
    /// 2. addr points to an incorrect position. (At the middle of an instruction, or where after it other instructions may jump to)
    /// 3. Set `NOT_MODIFY_MEMORY_PROTECT` where it should not be set.
    /// 4. hook or unhook from 2 or more threads at the same time without `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Because of memory protection colliding.
    /// 5. Other unpredictable errors.
    pub unsafe fn hook(self) -> Result<HookPoint<'a>, HookError> {
        self.hook_internal()
    }

    fn hook_internal(self) -> Result<HookPoint<'a>, HookError> {
        self.check_jmp_type()?;

        let jmp_inst_size = self.options.first_jmp_type.get_jmp_inst_size();
        let (moving_insts, origin) = get_moving_insts(self.addr, jmp_inst_size)?;
        let mut trampoline = match &self.options.first_jmp_type {
            JmpType::Direct(addr, len) => {
                let buffer = unsafe { slice::from_raw_parts_mut(*addr, *len) };
                Trampoline::with_buffer(buffer, &self.options.code_protect_cb)
            }
            _ => Trampoline::new(&self.options.code_protect_cb),
        };
        let trampoline_addr = trampoline.get_addr();
        trampoline.generate(
            self.addr,
            self.hook_type,
            &moving_insts,
            origin.len() as u8,
            self.user_data,
        )?;

        MemoryProtectGuard::new(&self.options.code_protect_cb, self.addr, jmp_inst_size).run(
            || {
                ThreadSuspendingGuard::new(&self.options.thread_operating_cb).run(|| {
                    modify_jmp(self.addr, trampoline_addr, &self.options.first_jmp_type);
                    Ok(())
                })
            },
        )?;

        Ok(HookPoint {
            addr: self.addr,
            trampoline,
            origin,
            jmp_inst_size,
            code_protect_cb: self.options.code_protect_cb,
            thread_operating_cb: self.options.thread_operating_cb,
        })
    }

    fn check_jmp_type(&self) -> Result<(), HookError> {
        match &self.options.first_jmp_type {
            JmpType::RipRelative => {}
            JmpType::Direct(addr, _) => {
                if ((*addr) as usize).abs_diff(self.addr + 5) > 0x7fff_ffff {
                    return Err(HookError::UnableToDirectJmp);
                }
            }
            JmpType::DirectWithRipRelative(second_jmp_addr) => {
                if second_jmp_addr.abs_diff(self.addr + 5) > 0x7fff_ffff {
                    return Err(HookError::UnableToDirectJmp);
                }
            }
        }
        Ok(())
    }
}

impl<'a> HookPoint<'a> {
    /// Consume self and unhook the address.
    pub unsafe fn unhook(self) -> Result<(), HookError> {
        self.unhook_internal()
    }

    fn unhook_internal(&self) -> Result<(), HookError> {
        MemoryProtectGuard::new(&self.code_protect_cb, self.addr, self.jmp_inst_size).run(|| {
            ThreadSuspendingGuard::new(&self.thread_operating_cb).run(|| {
                recover_jmp(self.addr, &self.origin[0..self.jmp_inst_size]);
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

fn get_moving_insts(
    addr: usize,
    min_bytes: usize,
) -> Result<(Vec<Instruction>, Vec<u8>), HookError> {
    let code_slice = unsafe { slice::from_raw_parts(addr as *const u8, MAX_INST_LEN * 2) };
    let mut decoder = Decoder::new(64, code_slice, DecoderOptions::NONE);
    decoder.set_ip(addr as u64);

    let mut total_bytes = 0;
    let mut ori_insts: Vec<Instruction> = vec![];
    for inst in &mut decoder {
        if inst.is_invalid() {
            return Err(HookError::Disassemble);
        }
        ori_insts.push(inst);
        total_bytes += inst.len();
        if total_bytes >= min_bytes {
            break;
        }
    }

    Ok((ori_insts, code_slice[0..decoder.position()].into()))
}

fn modify_jmp(dest_addr: usize, trampoline_addr: usize, jmp_type: &JmpType) {
    match jmp_type {
        JmpType::RipRelative => {
            let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, 14) };
            // jmp qword ptr [rip+0]
            buf[0..6].copy_from_slice(&[0xff, 0x25, 0, 0, 0, 0]);
            buf[6..14].copy_from_slice(&(trampoline_addr as u64).to_le_bytes());
        }
        JmpType::Direct(_, _) => {
            let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, 5) };
            let distance = trampoline_addr as i64 - (dest_addr as i64 + 5);
            // jmp xxx
            buf[0] = 0xe9;
            buf[1..5].copy_from_slice(&(distance as i32).to_le_bytes());
        }
        JmpType::DirectWithRipRelative(second_jmp_addr) => {
            let jmp1_buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, 5) };
            let distance = *second_jmp_addr as i64 - (dest_addr as i64 + 5);
            // jmp xxx
            jmp1_buf[0] = 0xe9;
            jmp1_buf[1..5].copy_from_slice(&(distance as i32).to_le_bytes());

            let jmp2_buf = unsafe { slice::from_raw_parts_mut(*second_jmp_addr as *mut u8, 14) };
            // jmp qword ptr [rip+0]
            jmp2_buf[0..6].copy_from_slice(&[0xff, 0x25, 0, 0, 0, 0]);
            jmp2_buf[6..14].copy_from_slice(&(trampoline_addr as u64).to_le_bytes());
        }
    }
}

fn recover_jmp(dest_addr: usize, origin: &[u8]) {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, origin.len()) };
    // jmp trampoline_addr
    buf.copy_from_slice(origin);
}
