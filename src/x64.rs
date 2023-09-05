use bitflags::bitflags;
use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Decoder, DecoderOptions, Encoder, FlowControl,
    Instruction, InstructionBlock, MemoryOperand, Mnemonic, Register,
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

struct JmpOffsetInfo {
    disp_offset: u64,
    relative_start_offset: u64,
    dest_addr: u64,
}

struct JmpRelocationInfo {
    offset_of_addr_to_relocate: u64,
    relative_start_offset: u64,
    is_jmp_to_new_insts: bool,
    dest_addr: u64, // index of new insts if jmps to new insts
}

struct NewInstInfo {
    offset: u64,
    reloc_info: Option<JmpRelocationInfo>,
}

fn move_code_to_addr(ori_insts: &Vec<Instruction>, dest_addr: u64) -> Result<Vec<u8>, HookError> {
    if ori_insts[0].ip().abs_diff(dest_addr) < 0x7fff_f000 {
        // use iced_x86 to relocate instructions when new address is in +/- 2GB
        let block = InstructionBlock::new(ori_insts, dest_addr);
        let encoded = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
            .map_err(|_| HookError::MoveCode)?;
        Ok(encoded.code_buffer)
    } else {
        let mut new_inst_info: Vec<NewInstInfo> = vec![];
        let mut buf = Cursor::new(Vec::<u8>::with_capacity(100));
        for inst in ori_insts {
            let cur_pos = buf.stream_position().unwrap();
            let off_info = relocate_inst_addr(inst, dest_addr + cur_pos, &mut buf)?;
            let reloc_info = if let Some(off_info) = off_info {
                let last_inst = ori_insts.last().unwrap();
                let (is_jmp_to_new_insts, dest_addr) = if off_info.dest_addr >= ori_insts[0].ip()
                    && off_info.dest_addr < last_inst.ip() + last_inst.len() as u64
                {
                    let idx = ori_insts
                        .iter()
                        .position(|i| i.ip() == off_info.dest_addr)
                        .ok_or(HookError::MovingCodeNotSupported)?;
                    (true, idx as u64)
                } else {
                    (false, off_info.dest_addr)
                };
                Some(JmpRelocationInfo {
                    offset_of_addr_to_relocate: cur_pos + off_info.disp_offset,
                    relative_start_offset: cur_pos + off_info.relative_start_offset,
                    is_jmp_to_new_insts,
                    dest_addr,
                })
            } else {
                None
            };
            new_inst_info.push(NewInstInfo {
                offset: cur_pos,
                reloc_info,
            });
        }
        let need_relocating_cnt = new_inst_info
            .iter()
            .filter(|i| i.reloc_info.is_some())
            .count();

        if need_relocating_cnt != 0 {
            // align the addresses to 8-byte
            let cur_addr = dest_addr + buf.stream_position().unwrap();
            let padding_cnt = ((cur_addr + 5 + 7) & !7) - cur_addr; // 5 bytes for jmp near below
                                                                    // jmp over the relocation address table
            let jmp_over_len = padding_cnt + need_relocating_cnt as u64 * 8;
            buf.write(&[0xe9])?;
            buf.write(&(jmp_over_len as u32).to_le_bytes())?;

            buf.write(&vec![0xCCu8; padding_cnt as usize])?;

            // write relocation table and relocate
            let mut table_offset = buf.stream_position().unwrap();
            for new_inst in &new_inst_info {
                if let Some(reloc_info) = &new_inst.reloc_info {
                    buf.seek(SeekFrom::Start(reloc_info.offset_of_addr_to_relocate))
                        .unwrap();
                    let disp = (table_offset - reloc_info.relative_start_offset) as u32;
                    buf.write(&disp.to_le_bytes()).unwrap();
                    buf.seek(SeekFrom::Start(table_offset)).unwrap();
                    if reloc_info.is_jmp_to_new_insts {
                        let real_dest_addr =
                            dest_addr + new_inst_info[reloc_info.dest_addr as usize].offset;
                        buf.write(&real_dest_addr.to_le_bytes())?;
                    } else {
                        buf.write(&reloc_info.dest_addr.to_le_bytes())?;
                    }
                    table_offset += 8;
                }
            }
        }
        Ok(buf.into_inner())
    }
}

fn relocate_inst_addr<T: Write>(
    inst: &Instruction,
    dest_addr: u64,
    buf: &mut T,
) -> Result<Option<JmpOffsetInfo>, HookError> {
    let mut encoder = Encoder::new(64);
    match inst.flow_control() {
        FlowControl::UnconditionalBranch => {
            // origin: jmp xxx
            // new: jmp qword ptr [rip+xxx]
            buf.write(&[0xff, 0x25, 0, 0, 0, 0])?;
            Ok(Some(JmpOffsetInfo {
                disp_offset: 2,
                relative_start_offset: 6,
                dest_addr: inst.near_branch_target(),
            }))
        }
        FlowControl::IndirectBranch if inst.is_ip_rel_memory_operand() => {
            // origin: jmp qword ptr [rip+xxx]
            // new:
            // mov [rsp-0x10], rax;
            // mov rax, xxx;
            // push [rax];
            // mov rax, [rsp-8];
            // ret
            buf.write(&[0x48, 0x89, 0x44, 0x24, 0xf0, 0x48, 0xb8])?;
            buf.write(&inst.ip_rel_memory_address().to_le_bytes())?;
            buf.write(&[0xff, 0x30, 0x48, 0x8b, 0x44, 0x24, 0xf8, 0xc3])?;
            Ok(None)
        }
        FlowControl::ConditionalBranch if inst.is_jcc_short_or_near() => {
            // origin: je a
            // new: jne @+6; jmp a;
            let mut new_inst = inst.clone();
            new_inst.negate_condition_code();
            new_inst.set_near_branch64(dest_addr + 8);
            new_inst.as_short_branch();
            encoder
                .encode(&new_inst, dest_addr)
                .map_err(|_| HookError::MoveCode)?;
            buf.write(&encoder.take_buffer())?;
            buf.write(&[0xff, 0x25, 0, 0, 0, 0])?;
            Ok(Some(JmpOffsetInfo {
                disp_offset: 4,
                relative_start_offset: 8,
                dest_addr: inst.near_branch_target(),
            }))
        }
        FlowControl::ConditionalBranch
            if inst.is_jcx_short() || inst.is_loop() || inst.is_loopcc() =>
        {
            // origin: jrcxz a
            // new: jrcxz @+2; jmp @+6; jmp a;
            let mut new_inst = inst.clone();
            new_inst.set_near_branch64(dest_addr + 4);
            encoder
                .encode(&new_inst, dest_addr)
                .map_err(|_| HookError::MoveCode)?;
            buf.write(&encoder.take_buffer())?;
            buf.write(&[0xeb, 0x06, 0xff, 0x25, 0, 0, 0, 0])?;
            Ok(Some(JmpOffsetInfo {
                disp_offset: 6,
                relative_start_offset: 10,
                dest_addr: inst.near_branch_target(),
            }))
        }
        FlowControl::Call if inst.is_call_near() || inst.is_call_far() => {
            // origin: call a
            // new: call qword ptr [rip+xxx]
            buf.write(&[0xff, 0x15, 0, 0, 0, 0])?;
            Ok(Some(JmpOffsetInfo {
                disp_offset: 2,
                relative_start_offset: 6,
                dest_addr: inst.near_branch_target(),
            }))
        }
        FlowControl::IndirectCall if inst.is_ip_rel_memory_operand() => {
            // origin: call qword ptr [rip+xxx]
            // new:
            // mov [rsp-0x18], rax
            // mov rax, xxx
            // push @retn_lower
            // mov dword ptr [rsp+4], @retn_higher
            // push qword ptr [rax]
            // mov rax, [rsp-8]
            // ret
            buf.write(&[0x48, 0x89, 0x44, 0x24, 0xf0, 0x48, 0xb8])?;
            buf.write(&inst.ip_rel_memory_address().to_le_bytes())?;
            let retn_addr = dest_addr + 0x13;
            buf.write(&[0x68])?;
            buf.write(&((retn_addr & 0xffffffff) as u32).to_le_bytes())?;
            buf.write(&[0xc7, 0x44, 0x24, 0x04])?;
            buf.write(&((retn_addr >> 32) as u32).to_le_bytes())?;
            buf.write(&[0xff, 0x30, 0x48, 0x8b, 0x44, 0x24, 0xf8, 0xc3])?;
            Ok(None)
        }
        _ if inst.is_ip_rel_memory_operand() => {
            if let Register::RSP = inst.op0_register() {
                // not support instructions writing rsp, like:
                // add rsp, qword ptr [rip+xxx]
                return Err(HookError::MovingCodeNotSupported);
            }
            let encoded = relocate_ip_rel_memory_inst(inst, dest_addr);
            buf.write(&encoded)?;
            Ok(None)
        }
        _ => {
            encoder.encode(inst, dest_addr).unwrap();
            buf.write(&encoder.take_buffer())?;
            Ok(None)
        }
    }
}

fn relocate_ip_rel_memory_inst(inst: &Instruction, dest_addr: u64) -> Vec<u8> {
    if let Mnemonic::Lea = inst.mnemonic() {
        // origin: lea eax, [rip+xxx]
        // new: mov eax, xxx
        let inst = Instruction::with2(
            Code::Mov_r64_imm64,
            inst.op0_register(),
            inst.ip_rel_memory_address(),
        )
        .unwrap();
        let mut encoder = Encoder::new(64);
        encoder.encode(&inst, dest_addr).unwrap();
        encoder.take_buffer()
    } else {
        // origin: add dword ptr [rip+xxx], rbx
        // new:
        // mov [rsp-0x10], r8
        // mov r8, xxx
        // add dword ptr [r8], rbx
        // mov r8, [rsp-0x10]
        let middle_register = if inst_not_use_rbx(inst) {
            Register::RBX
        } else if inst_not_use_r8(inst) {
            Register::R8
        } else {
            Register::R9 // An instruction uses rel memory operand may not use 3 registers
        };
        let new_inst1 = Instruction::with2(
            Code::Mov_rm64_r64,
            MemoryOperand::with_base_displ(Register::RSP, -16),
            middle_register,
        )
        .unwrap();
        let new_inst2 = Instruction::with2(
            Code::Mov_r64_imm64,
            middle_register,
            inst.ip_rel_memory_address(),
        )
        .unwrap();
        let mut new_inst3 = inst.clone();
        new_inst3.set_memory_base(middle_register);
        new_inst3.set_memory_displacement32(0);
        new_inst3.set_memory_displ_size(0);

        let stack_inc = inst.stack_pointer_increment() as i64;
        let new_inst4 = Instruction::with2(
            Code::Mov_r64_rm64,
            middle_register,
            MemoryOperand::with_base_displ(Register::RSP, -16 - stack_inc),
        )
        .unwrap();
        let new_insts = [new_inst1, new_inst2, new_inst3, new_inst4];
        let block = InstructionBlock::new(&new_insts, dest_addr);
        let encoded = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE).unwrap();
        encoded.code_buffer
    }
}

fn inst_not_use_rbx(inst: &Instruction) -> bool {
    (0..inst.op_count()).map(|i| inst.op_register(i)).all(|r| {
        !matches!(r, Register::BL)
            && !matches!(r, Register::BH)
            && !matches!(r, Register::BX)
            && !matches!(r, Register::EBX)
            && !matches!(r, Register::RBX)
    })
}
fn inst_not_use_r8(inst: &Instruction) -> bool {
    (0..inst.op_count()).map(|i| inst.op_register(i)).all(|r| {
        !matches!(r, Register::R8D)
            && !matches!(r, Register::R8L)
            && !matches!(r, Register::R8W)
            && !matches!(r, Register::R8)
    })
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
    fn move_inst(inst: &[u8], ori_base_addr: u64, new_base_addr: u64) -> Vec<u8> {
        let mut decoder = Decoder::new(64, inst, DecoderOptions::NONE);
        decoder.set_ip(ori_base_addr);
        let insts: Vec<Instruction> = decoder.iter().collect();
        let moved = move_code_to_addr(&insts, new_base_addr);
        assert_eq!(moved.is_ok(), true);
        moved.unwrap()
    }

    #[test]
    fn test_move_inst_short_1() {
        // jmp @+2
        let inst = [0xeb, 0x02];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr, addr + 300);
        assert_eq!(new_inst, [0xe9, 0xd3, 0xfe, 0xff, 0xff]);
    }

    #[test]
    fn test_move_inst_short_2() {
        // call @+10
        let inst = [0xe8, 0xa, 0, 0, 0];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr, addr - 0x3333);
        assert_eq!(new_inst, [0xe8, 0x3d, 0x33, 0x0, 0x0])
    }

    #[test]
    fn test_move_inst_short_3() {
        // mov rbx, [rip + 0x00000001]
        let inst = [0x48, 0x8b, 0x1d, 0x01, 0x00, 0x00, 0x00];
        let addr = inst.as_ptr() as u64;
        let new_inst = move_inst(&inst, addr, addr + 0x4000);
        assert_eq!(new_inst, [0x48, 0x8b, 0x1d, 0x1, 0xc0, 0xff, 0xff]);
    }

    #[test]
    fn test_move_inst_long_1() {
        // jmp @+2
        let inst = [0xeb, 0x02];
        let addr = 0x1000_0000;
        let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
        assert_eq!(new_inst, [0x48, 0x8b, 0x1d, 0x1, 0xc0, 0xff, 0xff]);
    }

    #[test]
    fn test_move_inst_long_2() {
        // jne @+0
        let inst = [0x75, 0x00];
        let addr = 0x1000_0000;
        let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
        assert_eq!(new_inst, [0x48, 0x8b, 0x1d, 0x1, 0xc0, 0xff, 0xff]);
    }

    #[test]
    fn test_iced1() {
        //let inst = [0xff, 0x25, 0x00, 0x10, 0x00, 0x00];
        let inst = [0x48, 0x83, 0xec, 0x10];
        let mut decoder = Decoder::new(64, &inst, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let insts: Vec<Instruction> = decoder.iter().collect();
        assert_eq!(insts.len(), 1);
        let inst = insts[0];

        println!("flow: {:?}", inst.flow_control());
        println!("far br 32: {:x}", inst.far_branch32());
        println!("far br selector: {:x}", inst.far_branch_selector());
        println!("near br target: {:x}", inst.near_branch_target());
        println!("is ip rel memory: {}", inst.is_ip_rel_memory_operand());
        println!("ip rel memory: {:x}", inst.ip_rel_memory_address());
        println!("sp inc: {}", inst.stack_pointer_increment());

        /*let mut inst = inst.clone();
        inst.set_memory_base(Register::RBX);
        inst.set_memory_displ_size(0);
        inst.set_memory_displacement32(0);

        let mut encoder = Encoder::new(64);
        let enc_ret = encoder.encode(&inst, 0x1000);
        match enc_ret {
            Ok(_) => {
                println!("encoded: {:?}", encoder.take_buffer());
            }
            Err(e) => {
                println!("enc err: {}", e);
            }
        }*/
    }
    #[test]
    fn test_iced2() {
        let inst = [0xe2, 0x00];
        let mut decoder = Decoder::new(64, &inst, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let insts: Vec<Instruction> = decoder.iter().collect();
        assert_eq!(insts.len(), 1);
        let inst = insts[0];
        println!("code: {:x}", inst.mnemonic() as u8);
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
