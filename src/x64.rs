use capstone::arch::x86::X86InsnDetail;
use capstone::prelude::*;
use std::io::{Cursor, Write};
use std::pin::Pin;
use std::slice;

#[cfg(windows)]
use winapi::shared::minwindef::LPVOID;
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::memoryapi::VirtualProtect;

#[cfg(unix)]
use libc::{__errno_location, c_void, mprotect, sysconf};

use crate::err::HookError;

const MAX_INST_LEN: usize = 15;
const JMP_INST_SIZE: usize = 5;

/// The routine used in a `jmp-back hook`, which means the EIP will jump back to the
/// original position after the Routine being run.
///
/// # Arguments
///
/// * regs - The registers
/// * src_addr - The address that has been hooked
pub type JmpBackRoutine = unsafe extern "C" fn(regs: *mut Registers, src_addr: usize);

/// The routine used in a `function hook`, which means the Routine will replace the
/// original FUNCTION, and the EIP will `retn` directly instead of jumping back.
/// Note that the being-hooked address must be the head of a function.
///
/// # Arguments
///
/// * regs - The registers
/// * ori_func_ptr - Original function pointer. Call it after converted to the original function type.
/// * src_addr - The address that has been hooked
///
/// Return the new return value of the replaced function.
pub type RetnRoutine =
    unsafe extern "C" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

/// The routine used in a `jmp-addr hook`, which means the EIP will jump to the specified
/// address after the Routine being run.
///
/// # Arguments
///
/// * regs - The registers
/// * ori_func_ptr - Original function pointer. Call it after converted to the original function type.
/// * src_addr - The address that has been hooked
pub type JmpToAddrRoutine =
    unsafe extern "C" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize);

/// The routine used in a `jmp-ret hook`, which means the EIP will jump to the return
/// value of the Routine.
///
/// # Arguments
///
/// * regs - The registers
/// * ori_func_ptr - Original function pointer. Call it after converted to the original function type.
/// * src_addr - The address that has been hooked
///
/// Return the address you want to jump to.
pub type JmpToRetRoutine =
    unsafe extern "C" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

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
    /// Direct long jump. `jmp` instruction use 5 bytes, but may fail as memory allocation near the 2GB space may fail.
    /// `jmp 0xXXXXXXXX`
    Direct,

    /// Mov rax and jump. Use 11 bytes.
    /// `mov rax, 0xXXXXXXXXXXXXXXXX; jmp rax;`
    MovJmp,

    /// Use 2 jmp instructions to jump. You have to specify the position of the second jmp.
    /// `jmp 0xXXXXXXXX; some codes; mov rax, 0xXXXXXXXX; jmp rax;`
    TrampolineJmp(usize),
}

/// The common registers.
#[repr(C)]
#[derive(Debug)]
pub struct Registers {
    /// The rax register
    pub rax: u64,
    /// The rcx register
    pub rcx: u64,
    /// The rdx register
    pub rdx: u64,
    /// The rbx register
    pub rbx: u64,
    /// The rsp register
    pub rsp: u64,
    /// The rbp register
    pub rbp: u64,
    /// The rsi register
    pub rsi: u64,
    /// The rdi register
    pub rdi: u64,
    /// The r8 register
    pub r8: u64,
    /// The r9 register
    pub r9: u64,
    /// The r10 register
    pub r10: u64,
    /// The r11 register
    pub r11: u64,
    /// The r12 register
    pub r12: u64,
    /// The r13 register
    pub r13: u64,
    /// The r14 register
    pub r14: u64,
    /// The r15 register
    pub r15: u64,
    /// The flags register
    pub rflags: u64,
}

impl Registers {
    /// Get the value by index.
    ///
    /// # Arguments
    ///
    /// * cnt - The index of the arguments.
    ///
    /// # Safety
    ///
    /// Process may crash if register `rsp` does not point to a valid stack.
    pub unsafe fn get_arg(&self, cnt: usize) -> u64 {
        //TODO change to a macro?
        match cnt {
            1 => self.rcx,
            2 => self.rdx,
            3 => self.r8,
            4 => self.r9,
            x => *((self.rsp as usize + x * 8) as usize as *mut u64),
        }
    }
}

/// The trait which is called before and after the modifying of the `jmp` instruction.
/// Usually is used to suspend and resume all other threads, to avoid instruction colliding.
pub trait ThreadCallback {
    /// the callback before modifying `jmp` instruction, should return true if success.
    fn pre(&self) -> bool;
    /// the callback after modifying `jmp` instruction
    fn post(&self);
}

/// Option for thread callback
pub enum CallbackOption {
    /// Valid callback
    Some(Box<dyn ThreadCallback>),
    /// No callback
    None,
}

bitflags! {
    /// Hook flags
    pub struct HookFlags:u32 {
        /// If set, will not modify the memory protection of the destination address
        const NOT_MODIFY_MEMORY_PROTECT = 0x1;
    }
}

/// The entry struct in ilhook.
/// Please read the main doc to view usage.
pub struct Hooker {
    addr: usize,
    hook_type: HookType,
    thread_cb: CallbackOption,
    flags: HookFlags,
}

#[derive(Default)]
struct OriginalCode {
    buf: [u8; MAX_INST_LEN + JMP_INST_SIZE],
    len: u8,
}

/// The hook result returned by Hooker::hook.
pub struct HookPoint {
    addr: usize,
    stub: Pin<Box<[u8]>>,
    stub_prot: u32,
    origin: OriginalCode,
    thread_cb: CallbackOption,
    flags: HookFlags,
}

impl Hooker {
    /// Create a new Hooker.
    ///
    /// # Arguments
    ///
    /// * `addr` - The being-hooked address.
    /// * `hook_type` - The hook type and callback routine.
    /// * `thread_cb` - The callbacks before and after hooking.
    /// * `flags` - Hook flags
    pub fn new(
        addr: usize,
        hook_type: HookType,
        thread_cb: CallbackOption,
        flags: HookFlags,
    ) -> Self {
        Self {
            addr,
            hook_type,
            thread_cb,
            flags,
        }
    }

    /// Consumes self and execute hooking. Return the HookPoint.
    ///
    /// # Safety
    ///
    /// Process may crash (instead of panic!) if:
    ///
    /// 1. addr is not a accessible memory address.
    /// 2. addr points to an incorrect position. (At the middle of an instruction, or where after it other instructions may jump)
    /// 3. Set `NOT_MODIFY_MEMORY_PROTECT` where it should not be set.
    /// 4. hook or unhook from 2 or more threads at the same time without `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Because of memory protection colliding.
    /// 5. Other unpredictable errors.
    pub fn hook(self) -> Result<HookPoint, HookError> {
        let (moved_code, origin) = generate_moved_code(self.addr)?;
        let stub = generate_stub(&self, moved_code, origin.len)?;
        let stub_prot = modify_mem_protect(stub.as_ptr() as usize, stub.len())?;
        if !self.flags.contains(HookFlags::NOT_MODIFY_MEMORY_PROTECT) {
            let old_prot = modify_mem_protect(self.addr, JMP_INST_SIZE)?;
            let ret = modify_jmp_with_thread_cb(&self, stub.as_ptr() as usize);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
            ret?;
        } else {
            modify_jmp_with_thread_cb(&self, stub.as_ptr() as usize)?;
        }
        Ok(HookPoint {
            addr: self.addr,
            stub,
            stub_prot,
            origin,
            thread_cb: self.thread_cb,
            flags: self.flags,
        })
    }
}

impl HookPoint {
    /// Consume self and unhook the address.
    pub unsafe fn unhook(self) -> Result<(), HookError> {
        self.unhook_by_ref()
    }

    fn unhook_by_ref(&self) -> Result<(), HookError> {
        let ret: Result<(), HookError>;
        if !self.flags.contains(HookFlags::NOT_MODIFY_MEMORY_PROTECT) {
            let old_prot = modify_mem_protect(self.addr, JMP_INST_SIZE)?;
            ret = recover_jmp_with_thread_cb(&self);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
        } else {
            ret = recover_jmp_with_thread_cb(&self)
        }
        recover_mem_protect(self.stub.as_ptr() as usize, self.stub.len(), self.stub_prot);
        ret
    }
}

// When the HookPoint drops, it should unhook automatically.
impl Drop for HookPoint {
    fn drop(&mut self) {
        self.unhook_by_ref().unwrap_or_default();
    }
}

#[derive(Default)]
struct Inst {
    bytes: [u8; MAX_INST_LEN],
    len: u8,
    reloc_off: u8,
}

#[derive(Default)]
struct MovedCode {
    code_cnt: usize,
    code: [Inst; 5],
}
fn move_instruction(addr: usize, inst: &[u8], inst_detail: &X86InsnDetail, new_inst: &mut Inst) {}

fn generate_moved_code(addr: usize) -> Result<(MovedCode, OriginalCode), HookError> {
    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .syntax(arch::x86::ArchSyntax::Intel)
        .detail(true)
        .build()
        .expect("Failed to create Capstone object");

    let mut ret: MovedCode = Default::default();
    let code_slice =
        unsafe { slice::from_raw_parts(addr as *const u8, MAX_INST_LEN * JMP_INST_SIZE) };
    let mut code_idx = 0;
    let mut inst_idx = 0;
    while code_idx < JMP_INST_SIZE {
        let insts = match cs.disasm_count(&code_slice[code_idx..], addr as u64, 1) {
            Ok(i) => i,
            Err(_) => return Err(HookError::Disassemble),
        };
        let inst = insts.iter().nth(0).unwrap();
        let inst_detail = cs.insn_detail(&inst).unwrap();
        if let ArchDetail::X86Detail(arch_detail) = inst_detail.arch_detail() {
            move_instruction(
                addr + code_idx,
                inst.bytes(),
                &arch_detail,
                &mut ret.code[inst_idx],
            );
        } else {
            return Err(HookError::Unknown);
        }
        code_idx += inst.bytes().len();
        inst_idx += 1;
    }
    ret.code_cnt = inst_idx;
    let mut origin: OriginalCode = Default::default();
    origin.len = code_idx as u8;
    origin
        .buf
        .split_at_mut(code_idx)
        .0
        .copy_from_slice(&code_slice[..code_idx]);
    Ok((ret, origin))
}

fn generate_stub(
    hooker: &Hooker,
    moved_code: MovedCode,
    ori_len: u8,
) -> Result<Pin<Box<[u8]>>, HookError> {
    Err(HookError::Unknown)
}

#[cfg(windows)]
fn modify_mem_protect(addr: usize, len: usize) -> Result<u32, HookError> {
    Err(HookError::Unknown)
}

#[cfg(windows)]
fn recover_mem_protect(addr: usize, len: usize, old: u32) {}

fn modify_jmp(dest_addr: usize, stub_addr: usize) -> Result<(), HookError> {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, JMP_INST_SIZE) };
    // jmp stub_addr
    buf[0] = 0xe9;
    let rel_off = stub_addr as i32 - (dest_addr as i32 + 5);
    buf.split_at_mut(1)
        .1
        .copy_from_slice(&rel_off.to_le_bytes());
    Ok(())
}

fn modify_jmp_with_thread_cb(hook: &Hooker, stub_addr: usize) -> Result<(), HookError> {
    if let CallbackOption::Some(cbs) = &hook.thread_cb {
        if !cbs.pre() {
            return Err(HookError::PreHook);
        }
        let ret = modify_jmp(hook.addr, stub_addr);
        cbs.post();
        ret
    } else {
        modify_jmp(hook.addr, stub_addr)
    }
}

fn recover_jmp(dest_addr: usize, origin: &[u8]) {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, origin.len()) };
    // jmp stub_addr
    buf.copy_from_slice(origin);
}

fn recover_jmp_with_thread_cb(hook: &HookPoint) -> Result<(), HookError> {
    if let CallbackOption::Some(cbs) = &hook.thread_cb {
        if !cbs.pre() {
            return Err(HookError::PreHook);
        }
        recover_jmp(hook.addr, &hook.origin.buf[..hook.origin.len as usize]);
        cbs.post();
    } else {
        recover_jmp(hook.addr, &hook.origin.buf[..hook.origin.len as usize]);
    }
    Ok(())
}
