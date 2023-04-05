use bitflags::bitflags;
use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, Instruction, InstructionBlock,
};
use std::cmp;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::slice;

#[cfg(windows)]
use core::ffi::c_void;
#[cfg(windows)]
use windows_sys::Win32::Foundation::GetLastError;
#[cfg(windows)]
use windows_sys::Win32::System::Memory::VirtualProtect;
#[cfg(windows)]
mod fixed_memory_win;
#[cfg(windows)]
use fixed_memory_win::FixedMemory;

#[cfg(unix)]
use libc::{__errno_location, c_void, mprotect, sysconf};
#[cfg(unix)]
mod fixed_memory_unix;
#[cfg(unix)]
use fixed_memory_unix::FixedMemory;

use crate::err::HookError;

const MAX_INST_LEN: usize = 15;
const JMP_INST_SIZE: usize = 5;

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
        /// If set, will not modify the memory protection of the destination address, so that
        /// the `hook` function could be ALMOST thread-safe.
        const NOT_MODIFY_MEMORY_PROTECT = 0x1;
    }
}

/// The entry struct in ilhook.
/// Please read the main doc to view usage.
pub struct Hooker {
    addr: usize,
    hook_type: HookType,
    thread_cb: CallbackOption,
    user_data: usize,
    flags: HookFlags,
}

/// The hook result returned by `Hooker::hook`.
pub struct HookPoint {
    addr: usize,
    #[allow(dead_code)] // we only use the drop trait of the stub
    stub: FixedMemory,
    origin: Vec<u8>,
    thread_cb: CallbackOption,
    flags: HookFlags,
}

#[cfg(not(target_arch = "x86_64"))]
fn env_lock() {
    panic!("This crate should only be used in arch x86_32!")
}
#[cfg(target_arch = "x86_64")]
fn env_lock() {}

impl Hooker {
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
        thread_cb: CallbackOption,
        user_data: usize,
        flags: HookFlags,
    ) -> Self {
        env_lock();
        Self {
            addr,
            hook_type,
            thread_cb,
            user_data,
            flags,
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
    /// 3. Set `NOT_MODIFY_MEMORY_PROTECT` where it should not be set.
    /// 4. hook or unhook from 2 or more threads at the same time without `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Because of memory protection colliding.
    /// 5. Other unpredictable errors.
    pub unsafe fn hook(self) -> Result<HookPoint, HookError> {
        let (moving_insts, origin) = get_moving_insts(self.addr)?;
        let stub = generate_stub(&self, moving_insts, origin.len() as u8, self.user_data)?;
        if !self.flags.contains(HookFlags::NOT_MODIFY_MEMORY_PROTECT) {
            let old_prot = modify_mem_protect(self.addr, JMP_INST_SIZE)?;
            let ret = modify_jmp_with_thread_cb(&self, stub.addr as usize);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
            ret?;
        } else {
            modify_jmp_with_thread_cb(&self, stub.addr as usize)?;
        }
        Ok(HookPoint {
            addr: self.addr,
            stub,
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
            ret = recover_jmp_with_thread_cb(self);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
        } else {
            ret = recover_jmp_with_thread_cb(self)
        }
        ret
    }
}

// When the HookPoint drops, it should unhook automatically.
impl Drop for HookPoint {
    fn drop(&mut self) {
        self.unhook_by_ref().unwrap_or_default();
    }
}

fn get_moving_insts(addr: usize) -> Result<(Vec<Instruction>, Vec<u8>), HookError> {
    let code_slice =
        unsafe { slice::from_raw_parts(addr as *const u8, MAX_INST_LEN * JMP_INST_SIZE) };
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
        if total_bytes >= JMP_INST_SIZE {
            break;
        }
    }

    Ok((ori_insts, code_slice[0..decoder.position()].into()))
}

fn write_stub_prolog(buf: &mut impl Write) -> Result<usize, std::io::Error> {
    // push rsp
    // pushfq
    // test rsp,8
    // je _stack_aligned_16
    // ; stack not aligned to 16
    // push rax
    // sub rsp,0x10
    // mov rax, [rsp+0x20] # rsp
    // mov [rsp], rax
    // mov rax, [rsp+0x18] # rflags
    // mov [rsp+8], rax
    // mov rax, [rsp+0x10] # rax
    // mov [rsp+0x18], rax
    // mov dword ptr [rsp+0x10],1 # stack flag
    // jmp _other_registers
    // _stack_aligned_16:
    // push rax
    // push rax
    // mov rax, [rsp+0x18] # rsp
    // mov [rsp], rax
    // mov rax, [rsp+8] # rax
    // mov [rsp+0x18], rax
    // mov rax,[rsp+0x10] # rflags
    // mov [rsp+8], rax
    // mov dword ptr [rsp+0x10], 0 # stack flag
    // _other_registers:
    // push rbx
    // push rcx
    // push rdx
    // push rsi
    // push rdi
    // push rbp
    // push r8
    // push r9
    // push r10
    // push r11
    // push r12
    // push r13
    // push r14
    // push r15
    // sub rsp,0x40
    // movaps xmmword ptr ss:[rsp],xmm0
    // movaps xmmword ptr ss:[rsp+0x10],xmm1
    // movaps xmmword ptr ss:[rsp+0x20],xmm2
    // movaps xmmword ptr ss:[rsp+0x30],xmm3
    buf.write(&[
        0x54, 0x9C, 0x48, 0xF7, 0xC4, 0x08, 0x00, 0x00, 0x00, 0x74, 0x2C, 0x50, 0x48, 0x83, 0xEC,
        0x10, 0x48, 0x8B, 0x44, 0x24, 0x20, 0x48, 0x89, 0x04, 0x24, 0x48, 0x8B, 0x44, 0x24, 0x18,
        0x48, 0x89, 0x44, 0x24, 0x08, 0x48, 0x8B, 0x44, 0x24, 0x10, 0x48, 0x89, 0x44, 0x24, 0x18,
        0xC7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00, 0xEB, 0x27, 0x50, 0x50, 0x48, 0x8B, 0x44,
        0x24, 0x18, 0x48, 0x89, 0x04, 0x24, 0x48, 0x8B, 0x44, 0x24, 0x08, 0x48, 0x89, 0x44, 0x24,
        0x18, 0x48, 0x8B, 0x44, 0x24, 0x10, 0x48, 0x89, 0x44, 0x24, 0x08, 0xC7, 0x44, 0x24, 0x10,
        0x00, 0x00, 0x00, 0x00, 0x53, 0x51, 0x52, 0x56, 0x57, 0x55, 0x41, 0x50, 0x41, 0x51, 0x41,
        0x52, 0x41, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, 0x40,
        0x0F, 0x29, 0x04, 0x24, 0x0F, 0x29, 0x4C, 0x24, 0x10, 0x0F, 0x29, 0x54, 0x24, 0x20, 0x0F,
        0x29, 0x5C, 0x24, 0x30,
    ])
}

fn write_stub_epilog1(buf: &mut impl Write) -> Result<usize, std::io::Error> {
    // movaps xmm0,xmmword ptr ss:[rsp]
    // movaps xmm1,xmmword ptr ss:[rsp+0x10]
    // movaps xmm2,xmmword ptr ss:[rsp+0x20]
    // movaps xmm3,xmmword ptr ss:[rsp+0x30]
    // add rsp,0x40
    // pop r15
    // pop r14
    // pop r13
    // pop r12
    // pop r11
    // pop r10
    // pop r9
    // pop r8
    // pop rbp
    // pop rdi
    // pop rsi
    // pop rdx
    // pop rcx
    // pop rbx
    // add rsp,8
    buf.write(&[
        0x0F, 0x28, 0x04, 0x24, 0x0F, 0x28, 0x4C, 0x24, 0x10, 0x0F, 0x28, 0x54, 0x24, 0x20, 0x0F,
        0x28, 0x5C, 0x24, 0x30, 0x48, 0x83, 0xC4, 0x40, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41,
        0x5C, 0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5D, 0x5F, 0x5E, 0x5A, 0x59, 0x5B,
        0x48, 0x83, 0xC4, 0x08,
    ])
}

fn write_stub_epilog2_common(buf: &mut impl Write) -> Result<usize, std::io::Error> {
    // test dword ptr ss:[rsp+0x8],1
    // je _branch1
    // mov rax, [rsp+0x10]
    // mov [rsp+0x18], rax
    // popfq
    // pop rax
    // pop rax
    // pop rax
    // jmp _branch2
    // _branch1:
    // popfq
    // pop rax
    // pop rax
    // _branch2:
    buf.write(&[
        0xF7, 0x44, 0x24, 0x08, 0x01, 0x00, 0x00, 0x00, 0x74, 0x10, 0x48, 0x8B, 0x44, 0x24, 0x10,
        0x48, 0x89, 0x44, 0x24, 0x18, 0x9D, 0x58, 0x58, 0x58, 0xEB, 0x03, 0x9D, 0x58, 0x58,
    ])
}

fn write_stub_epilog2_jmp_ret(buf: &mut impl Write) -> Result<usize, std::io::Error> {
    // test dword ptr ss:[rsp+8],1
    // je _branch1
    // popfq
    // mov [rsp], rax
    // mov rax, [rsp+8]
    // mov rax, [rsp+0x10]
    // pop rax
    // pop rax
    // pop rax
    // jmp _branch2
    // _branch1:
    // popfq
    // mov [rsp-8],rax
    // pop rax
    // pop rax
    // _branch2:
    // jmp qword ptr ss:[rsp-0x18]
    buf.write(&[
        0xF7, 0x44, 0x24, 0x08, 0x01, 0x00, 0x00, 0x00, 0x74, 0x14, 0x9D, 0x48, 0x89, 0x04, 0x24,
        0x48, 0x8B, 0x44, 0x24, 0x08, 0x48, 0x8B, 0x44, 0x24, 0x10, 0x58, 0x58, 0x58, 0xEB, 0x08,
        0x9D, 0x48, 0x89, 0x44, 0x24, 0xF8, 0x58, 0x58, 0xFF, 0x64, 0x24, 0xE8,
    ])
}

fn jmp_addr<T: Write>(addr: u64, buf: &mut T) -> Result<(), HookError> {
    buf.write(&[0xff, 0x25, 0, 0, 0, 0])?;
    buf.write(&addr.to_le_bytes())?;
    Ok(())
}

fn move_code_to_addr(ori_insts: &Vec<Instruction>, dest_addr: u64) -> Result<Vec<u8>, HookError> {
    let block = InstructionBlock::new(ori_insts, dest_addr);
    let encoded = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
        .map_err(|_| HookError::MoveCode)?;
    Ok(encoded.code_buffer)
}

fn write_ori_func_addr<T: Write + Seek>(buf: &mut T, ori_func_addr_off: u64, ori_func_off: u64) {
    let pos = buf.stream_position().unwrap();
    buf.seek(SeekFrom::Start(ori_func_addr_off)).unwrap();
    buf.write(&ori_func_off.to_le_bytes()).unwrap();
    buf.seek(SeekFrom::Start(pos)).unwrap();
}

fn generate_jmp_back_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u64,
    moving_code: &Vec<Instruction>,
    ori_addr: usize,
    cb: JmpBackRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // mov rdx, user_data
    buf.write(&[0x48, 0xba])?;
    buf.write(&(user_data as u64).to_le_bytes())?;
    // mov rcx, rsp
    // sub rsp, 0x10
    // mov rax, cb
    buf.write(&[0x48, 0x89, 0xe1, 0x48, 0x83, 0xec, 0x10, 0x48, 0xb8])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x10
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x10])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;

    let cur_pos = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(moving_code, stub_base_addr + cur_pos)?)?;

    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;
    Ok(())
}

fn generate_retn_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u64,
    moving_code: &Vec<Instruction>,
    ori_addr: usize,
    cb: RetnRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // mov r8, user_data
    buf.write(&[0x49, 0xb8])?;
    buf.write(&(user_data as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.stream_position().unwrap() + 2;
    // mov rdx, ori_func
    // mov rcx, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xba, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe1, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    // mov [rsp + 0xc8], rax
    buf.write(&[
        0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20, 0x48, 0x89, 0x84, 0x24, 0xc8, 0x00, 0x00, 0x00,
    ])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;
    // ret
    buf.write(&[0xc3])?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        stub_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, stub_base_addr + ori_func_off);

    Ok(())
}

fn generate_jmp_addr_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u64,
    moving_code: &Vec<Instruction>,
    ori_addr: usize,
    dest_addr: usize,
    cb: JmpToAddrRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // mov r8, user_data
    buf.write(&[0x49, 0xb8])?;
    buf.write(&(user_data as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.stream_position().unwrap() + 2;
    // mov rdx, ori_func
    // mov rcx, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xba, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe1, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;
    jmp_addr(dest_addr as u64, buf)?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        stub_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, stub_base_addr + ori_func_off);

    Ok(())
}

fn generate_jmp_ret_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u64,
    moving_code: &Vec<Instruction>,
    ori_addr: usize,
    cb: JmpToRetRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // mov r8, user_data
    buf.write(&[0x49, 0xb8])?;
    buf.write(&(user_data as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.stream_position().unwrap() + 2;
    // mov rdx, ori_func
    // mov rcx, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xba, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe1, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_jmp_ret(buf)?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        stub_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, stub_base_addr + ori_func_off);

    Ok(())
}

fn generate_stub(
    hooker: &Hooker,
    moving_code: Vec<Instruction>,
    ori_len: u8,
    user_data: usize,
) -> Result<FixedMemory, HookError> {
    let fix_mem = FixedMemory::allocate(hooker.addr as u64)?;
    let p = unsafe {
        slice::from_raw_parts_mut(fix_mem.addr as usize as *mut u8, fix_mem.len as usize)
    };
    let mut buf = Cursor::new(p);

    write_stub_prolog(&mut buf)?;

    match hooker.hook_type {
        HookType::JmpBack(cb) => generate_jmp_back_stub(
            &mut buf,
            fix_mem.addr,
            &moving_code,
            hooker.addr,
            cb,
            ori_len,
            user_data,
        ),
        HookType::Retn(cb) => generate_retn_stub(
            &mut buf,
            fix_mem.addr,
            &moving_code,
            hooker.addr,
            cb,
            ori_len,
            user_data,
        ),
        HookType::JmpToAddr(dest_addr, cb) => generate_jmp_addr_stub(
            &mut buf,
            fix_mem.addr,
            &moving_code,
            hooker.addr,
            dest_addr,
            cb,
            ori_len,
            user_data,
        ),
        HookType::JmpToRet(cb) => generate_jmp_ret_stub(
            &mut buf,
            fix_mem.addr,
            &moving_code,
            hooker.addr,
            cb,
            ori_len,
            user_data,
        ),
    }?;

    Ok(fix_mem)
}

#[cfg(windows)]
fn modify_mem_protect(addr: usize, len: usize) -> Result<u32, HookError> {
    let mut old_prot: u32 = 0;
    let old_prot_ptr = std::ptr::addr_of_mut!(old_prot);
    // PAGE_EXECUTE_READWRITE = 0x40
    let ret = unsafe { VirtualProtect(addr as *const c_void, len, 0x40, old_prot_ptr) };
    if ret == 0 {
        Err(HookError::MemoryProtect(unsafe { GetLastError() }))
    } else {
        Ok(old_prot)
    }
}

#[cfg(unix)]
fn modify_mem_protect(addr: usize, len: usize) -> Result<u32, HookError> {
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
fn recover_mem_protect(addr: usize, len: usize, old: u32) {
    let mut old_prot: u32 = 0;
    let old_prot_ptr = std::ptr::addr_of_mut!(old_prot);
    unsafe { VirtualProtect(addr as *const c_void, len, old, old_prot_ptr) };
}

#[cfg(unix)]
fn recover_mem_protect(addr: usize, _: usize, old: u32) {
    let page_size = unsafe { sysconf(30) }; //_SC_PAGESIZE == 30
    unsafe {
        mprotect(
            (addr & !(page_size as usize - 1)) as *mut c_void,
            page_size as usize,
            old as i32,
        )
    };
}
fn modify_jmp(dest_addr: usize, stub_addr: usize) -> Result<(), HookError> {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, JMP_INST_SIZE) };
    // jmp stub_addr
    buf[0] = 0xe9;
    let rel_off = stub_addr as i64 - (dest_addr as i64 + 5);
    if i64::from(rel_off as i32) != rel_off {
        return Err(HookError::Unknown);
    }
    buf[1..5].copy_from_slice(&(rel_off as i32).to_le_bytes());
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
        recover_jmp(hook.addr, &hook.origin);
        cbs.post();
    } else {
        recover_jmp(hook.addr, &hook.origin);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(test)]
    fn move_inst(inst: &[u8], new_base_addr: u64) -> Vec<u8> {
        let mut decoder = Decoder::new(64, inst, DecoderOptions::NONE);
        decoder.set_ip(inst.as_ptr() as u64);
        let insts: Vec<Instruction> = decoder.iter().collect();
        let ret = BlockEncoder::encode(
            64,
            InstructionBlock::new(&insts, new_base_addr),
            BlockEncoderOptions::NONE,
        );
        assert_eq!(ret.is_ok(), true);
        ret.unwrap().code_buffer
    }

    #[test]
    fn test_move_inst_2() {
        // jmp @+2
        let inst = [0xeb, 0x02];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr + 300);
        assert_eq!(new_inst, [0xe9, 0xd3, 0xfe, 0xff, 0xff]);
    }
    #[test]
    fn test_move_inst_3() {
        // jmp @-0x20
        let inst = [0xe9, 0xe0, 0xff, 0xff, 0xff];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr + 0x00ff_ffff);
        assert_eq!(new_inst, [233, 225, 255, 255, 254]);
    }

    #[test]
    fn test_move_inst_4() {
        // call @+10
        let inst = [0xe8, 0xa, 0, 0, 0];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr - 0x3333);
        assert_eq!(new_inst, [0xe8, 0x3d, 0x33, 0x0, 0x0])
    }

    #[test]
    fn test_move_inst_8() {
        // mov rbx, [rip + 0x00000001]
        let inst = [0x48, 0x8b, 0x1d, 0x01, 0x00, 0x00, 0x00];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr + 0x4000);
        assert_eq!(new_inst, [0x48, 0x8b, 0x1d, 0x1, 0xc0, 0xff, 0xff]);
    }

    #[test]
    fn test_fixed_mem() {
        use super::FixedMemory;
        let m = FixedMemory::allocate(0x7fff_ffff);
        assert!(m.is_ok());
        let m = m.unwrap();
        assert_ne!(m.addr, 0);
        assert_eq!(m.len, 4096);
    }

    #[cfg(test)]
    #[inline(never)]
    // 5 arguments to ensure using stack instead of registers to pass parameter(s)
    extern "win64" fn foo(x: u64, _: u64, _: u64, _: u64, y: u64) -> u64 {
        println!("original foo, x:{}, y:{}", x, y);
        x * x + y
    }
    #[cfg(test)]
    unsafe extern "win64" fn on_foo(
        reg: *mut Registers,
        old_func: usize,
        user_data: usize,
    ) -> usize {
        let old_func = std::mem::transmute::<
            usize,
            extern "win64" fn(u64, u64, u64, u64, u64) -> u64,
        >(old_func);
        let arg_y = ((*reg).rsp + 0x28) as *const u64;
        old_func((*reg).rcx, 0, 0, 0, *arg_y) as usize + user_data
    }
    #[test]
    fn test_hook_function() {
        assert_eq!(foo(5, 0, 0, 0, 3), 28);
        let hooker = Hooker::new(
            foo as usize,
            HookType::Retn(on_foo),
            CallbackOption::None,
            100,
            HookFlags::empty(),
        );
        let info = unsafe { hooker.hook().unwrap() };
        assert_eq!(foo(5, 0, 0, 0, 3), 128);
        unsafe { info.unhook().unwrap() };
        assert_eq!(foo(5, 0, 0, 0, 3), 28);
    }
}
