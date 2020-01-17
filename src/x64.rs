use capstone::arch::{x86::X86OperandType, ArchOperand};
use capstone::prelude::*;
use std::cmp;
use std::io::{Cursor, Write};
use std::pin::Pin;
use std::slice;

#[cfg(windows)]
use winapi::shared::minwindef::LPVOID;
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::memoryapi::VirtualProtect;
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

/// The routine used in a `jmp-back hook`, which means the EIP will jump back to the
/// original position after the Routine being run.
///
/// # Arguments
///
/// * regs - The registers
/// * src_addr - The address that has been hooked
pub type JmpBackRoutine = unsafe extern "sysv64" fn(regs: *mut Registers, src_addr: usize);

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
    unsafe extern "sysv64" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

/// The routine used in a `jmp-addr hook`, which means the EIP will jump to the specified
/// address after the Routine being run.
///
/// # Arguments
///
/// * regs - The registers
/// * ori_func_ptr - Original function pointer. Call it after converted to the original function type.
/// * src_addr - The address that has been hooked
pub type JmpToAddrRoutine =
    unsafe extern "sysv64" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize);

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
    unsafe extern "sysv64" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

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
    /// The rsp register
    pub rsp: u64,
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
    /// The flags register
    pub rflags: u64,
    /// The rax register
    pub rax: u64,
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
    pub unsafe fn get_stack(&self, cnt: usize) -> u64 {
        *((self.rsp as usize + cnt * 8) as usize as *mut u64)
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
    #[allow(dead_code)] // we only use the drop trait of the stub
    stub: FixedMemory,
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
    pub unsafe fn hook(self) -> Result<HookPoint, HookError> {
        let (moved_code, origin) = generate_moved_code(self.addr)?;
        let stub = generate_stub(&self, moved_code, origin.len)?;
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
            ret = recover_jmp_with_thread_cb(&self);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
        } else {
            ret = recover_jmp_with_thread_cb(&self)
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

#[derive(Default)]
struct Inst {
    bytes: [u8; MAX_INST_LEN],
    len: u8,
    reloc_off: u8,
    reloc_addr: u64,
}

impl Inst {
    fn new(bytes: &[u8], reloc_off: u8, reloc_addr: u64) -> Self {
        let mut s = Self {
            bytes: [0; MAX_INST_LEN],
            len: bytes.len() as u8,
            reloc_off,
            reloc_addr,
        };
        if bytes.len() > MAX_INST_LEN {
            panic!("inst len error");
        }
        s.bytes[..bytes.len()].copy_from_slice(bytes);
        s
    }
}

#[derive(PartialEq, Debug)]
struct RelocEntry {
    off: u16,
    reloc_base_off: u8,
    dest_addr: u64,
}

fn read_i32_checked(buf: &[u8]) -> i32 {
    i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

fn get_jmp_dest_from_inst(detail: &ArchDetail) -> u64 {
    let ops = detail.operands();
    assert_eq!(ops.len(), 1);
    if let ArchOperand::X86Operand(op) = &ops[0] {
        if let X86OperandType::Imm(v) = op.op_type {
            v as u64
        } else {
            panic!("not jmp?")
        }
    } else {
        panic!("not jmp?")
    }
}

fn move_instruction(addr: usize, inst: &[u8], arch_detail: &ArchDetail) -> Inst {
    let x86 = arch_detail.x86().unwrap();
    let op1 = x86.opcode()[0];
    let op2 = x86.opcode()[1];
    match op1 {
        // short jXX
        x if (x & 0xf0) == 0x70 => Inst::new(
            &[0x0f, 0x80 | (x & 0xf), 0, 0, 0, 0],
            2,
            get_jmp_dest_from_inst(&arch_detail),
        ),
        // long jXX
        0x0f if (op2 & 0xf0) == 0x80 => Inst::new(
            &[0x0f, op2, 0, 0, 0, 0],
            2,
            get_jmp_dest_from_inst(&arch_detail),
        ),
        // loop/jecxz
        x @ 0xe0..=0xe3 => Inst::new(
            &[x, 0x02, 0xeb, 0x05, 0xe9, 0, 0, 0, 0],
            5,
            get_jmp_dest_from_inst(&arch_detail),
        ),
        // short and long jmp
        0xeb | 0xe9 => Inst::new(&[0xe9, 0, 0, 0, 0], 1, get_jmp_dest_from_inst(&arch_detail)),
        // call
        0xe8 => Inst::new(&[0xe8, 0, 0, 0, 0], 1, get_jmp_dest_from_inst(&arch_detail)),
        _ if x86.modrm() == 5 => copy_rip_relative_inst(addr, inst, &arch_detail),
        _ => Inst::new(inst, 0, 0),
    }
}

fn copy_rip_relative_inst(addr: usize, inst: &[u8], arch_detail: &ArchDetail) -> Inst {
    let ops = arch_detail.operands();
    let mut disp: Option<i32> = None;
    for op in ops {
        if let ArchOperand::X86Operand(x86) = op {
            if let X86OperandType::Mem(op_mem) = x86.op_type {
                disp = Some(op_mem.disp() as i32);
                break;
            }
        }
    }
    let disp = disp.unwrap_or_else(|| panic!("unknown instruction"));
    let dest_addr = addr as i64 + inst.len() as i64 + disp as i64;
    let mut i = 2;
    while i <= inst.len() - 4 {
        if disp == read_i32_checked(&inst[i..]) {
            break;
        }
        i += 1;
    }
    if i > inst.len() - 4 {
        panic!("unknown error");
    }
    Inst::new(inst, i as u8, dest_addr as u64)
}

fn generate_moved_code(addr: usize) -> Result<(Vec<Inst>, OriginalCode), HookError> {
    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .syntax(arch::x86::ArchSyntax::Intel)
        .detail(true)
        .build()
        .expect("Failed to create Capstone object");

    let code_slice =
        unsafe { slice::from_raw_parts(addr as *const u8, MAX_INST_LEN * JMP_INST_SIZE) };
    let mut code_idx = 0;

    let mut ret: Vec<Inst> = vec![];
    while code_idx < JMP_INST_SIZE {
        let insts = match cs.disasm_count(&code_slice[code_idx..], (addr + code_idx) as u64, 1) {
            Ok(i) => i,
            Err(_) => return Err(HookError::Disassemble),
        };
        let inst = insts.iter().nth(0).unwrap();
        let inst_detail = cs.insn_detail(&inst).unwrap();

        ret.push(move_instruction(
            addr + code_idx,
            inst.bytes(),
            &inst_detail.arch_detail(),
        ));
        code_idx += inst.bytes().len();
    }
    let mut origin: OriginalCode = Default::default();
    origin.len = code_idx as u8;
    origin.buf[..code_idx].copy_from_slice(&code_slice[..code_idx]);
    Ok((ret, origin))
}

fn write_stub_prolog(buf: &mut Cursor<Vec<u8>>) -> Result<usize, std::io::Error> {
    // pushfq
    // test rsp,8
    // je _branch1
    // push rax
    // mov rax,qword ptr ss:[rsp+8]
    // mov dword ptr ss:[rsp+8],0
    // jmp _branch2
    // _branch1:
    // sub rsp,8
    // push rax
    // mov rax,qword ptr ss:[rsp+10]
    // mov dword ptr ss:[rsp+8],1
    // _branch2:
    // push rax
    // push rbx
    // push rcx
    // push rdx
    // push rsi
    // push rdi
    // push rsp
    // push rbp
    // push r8
    // push r9
    // push r10
    // push r11
    // push r12
    // push r13
    // push r14
    // push r15
    // sub rsp,40
    // movaps xmmword ptr ss:[rsp],xmm0
    // movaps xmmword ptr ss:[rsp+10],xmm1
    // movaps xmmword ptr ss:[rsp+20],xmm2
    // movaps xmmword ptr ss:[rsp+30],xmm3
    buf.write(&[
        0x9C, 0x48, 0xF7, 0xC4, 0x08, 0x00, 0x00, 0x00, 0x74, 0x10, 0x50, 0x48, 0x8B, 0x44, 0x24,
        0x08, 0xC7, 0x44, 0x24, 0x08, 0x00, 0x00, 0x00, 0x00, 0xEB, 0x12, 0x48, 0x83, 0xEC, 0x08,
        0x50, 0x48, 0x8B, 0x44, 0x24, 0x10, 0xC7, 0x44, 0x24, 0x08, 0x01, 0x00, 0x00, 0x00, 0x50,
        0x53, 0x51, 0x52, 0x56, 0x57, 0x54, 0x55, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53,
        0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, 0x40, 0x0F, 0x29, 0x04,
        0x24, 0x0F, 0x29, 0x4C, 0x24, 0x10, 0x0F, 0x29, 0x54, 0x24, 0x20, 0x0F, 0x29, 0x5C, 0x24,
        0x30,
    ])
}

fn write_stub_epilog1(buf: &mut Cursor<Vec<u8>>) -> Result<usize, std::io::Error> {
    // movaps xmm0,xmmword ptr ss:[rsp]
    // movaps xmm1,xmmword ptr ss:[rsp+10]
    // movaps xmm2,xmmword ptr ss:[rsp+20]
    // movaps xmm3,xmmword ptr ss:[rsp+30]
    // add rsp,40
    // pop r15
    // pop r14
    // pop r13
    // pop r12
    // pop r11
    // pop r10
    // pop r9
    // pop r8
    // pop rbp
    // pop rsp
    // pop rdi
    // pop rsi
    // pop rdx
    // pop rcx
    // pop rbx
    buf.write(&[
        0x0F, 0x28, 0x04, 0x24, 0x0F, 0x28, 0x4C, 0x24, 0x10, 0x0F, 0x28, 0x54, 0x24, 0x20, 0x0F,
        0x28, 0x5C, 0x24, 0x30, 0x48, 0x83, 0xC4, 0x40, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41,
        0x5C, 0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5D, 0x5C, 0x5F, 0x5E, 0x5A, 0x59,
        0x5B,
    ])
}

fn write_stub_epilog2_common(buf: &mut Cursor<Vec<u8>>) -> Result<usize, std::io::Error> {
    // test dword ptr ss:[rsp+10],1
    // je _branch1
    // popfq
    // pop rax
    // add rsp,10
    // jmp _branch2
    // _branch1:
    // popfq
    // pop rax
    // add rsp,8
    // _branch2:
    buf.write(&[
        0xF7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00, 0x74, 0x08, 0x9D, 0x58, 0x48, 0x83, 0xC4,
        0x10, 0xEB, 0x06, 0x9D, 0x58, 0x48, 0x83, 0xC4, 0x08,
    ])
}

fn write_stub_epilog2_jmp_ret(buf: &mut Cursor<Vec<u8>>) -> Result<usize, std::io::Error> {
    // test dword ptr ss:[rsp+10],1
    // je _branch1
    // popfq
    // mov qword ptr ss:[rsp+10],rax
    // pop rax
    // add rsp,10
    // jmp _branch2
    // _branch1:
    // popfq
    // mov qword ptr ss:[rsp+8],rax
    // pop rax
    // add rsp,8
    // _branch2:
    // jmp qword ptr ss:[rsp-8]
    buf.write(&[
        0xF7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00, 0x74, 0x0D, 0x9D, 0x48, 0x89, 0x44, 0x24,
        0x10, 0x58, 0x48, 0x83, 0xC4, 0x10, 0xEB, 0x0B, 0x9D, 0x48, 0x89, 0x44, 0x24, 0x08, 0x58,
        0x48, 0x83, 0xC4, 0x08, 0xFF, 0x64, 0x24, 0xF8,
    ])
}
fn write_moved_code_to_buf(
    code: Vec<Inst>,
    buf: &mut Cursor<Vec<u8>>,
    reloc_tbl: &mut Vec<RelocEntry>,
) {
    code.iter().for_each(|inst| {
        if inst.reloc_off != 0 {
            reloc_tbl.push(RelocEntry {
                off: buf.position() as u16 + inst.reloc_off as u16,
                reloc_base_off: inst.len - inst.reloc_off,
                dest_addr: inst.reloc_addr,
            });
        }
        buf.write(&inst.bytes[..inst.len as usize]).unwrap();
    });
}

fn jmp_addr(addr: u64, buf: &mut Cursor<Vec<u8>>) -> Result<(), HookError> {
    buf.write(&[0xff, 0x25, 0, 0, 0, 0])?;
    buf.write(&addr.to_le_bytes())?;
    Ok(())
}

fn generate_jmp_back_stub(
    buf: &mut Cursor<Vec<u8>>,
    rel_tbl: &mut Vec<RelocEntry>,
    moved_code: Vec<Inst>,
    ori_addr: usize,
    cb: JmpBackRoutine,
    ori_len: u8,
) -> Result<(u16, u16), HookError> {
    // mov rsi, ori_addr
    buf.write(&[0x48, 0xbe])?;
    buf.write(&(ori_addr as u64).to_le_bytes())?;
    // mov rdi, rsp
    // sub rsp, 0x10
    // mov rax, cb
    buf.write(&[0x48, 0x89, 0xe7, 0x48, 0x83, 0xec, 0x10, 0x48, 0xb8])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x10
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x10])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;

    write_moved_code_to_buf(moved_code, buf, rel_tbl);
    jmp_addr(ori_addr as u64 + ori_len as u64, buf)?;
    Ok((0, 0))
}

fn generate_retn_stub(
    buf: &mut Cursor<Vec<u8>>,
    rel_tbl: &mut Vec<RelocEntry>,
    moved_code: Vec<Inst>,
    ori_addr: usize,
    cb: RetnRoutine,
    ori_len: u8,
) -> Result<(u16, u16), HookError> {
    // mov rdx, ori_addr
    buf.write(&[0x48, 0xba])?;
    buf.write(&(ori_addr as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.position() as u16 + 2;
    // mov rsi, ori_func
    // mov rdi, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xbe, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe7, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    // mov [rsp + 0xc0], rax
    buf.write(&[
        0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20, 0x48, 0x89, 0x84, 0x24, 0xc0, 0x00, 0x00, 0x00,
    ])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;
    // ret
    buf.write(&[0xc3])?;

    let ori_func_off = buf.position() as u16;
    write_moved_code_to_buf(moved_code, buf, rel_tbl);
    jmp_addr(ori_addr as u64 + ori_len as u64, buf)?;

    Ok((ori_func_addr_off, ori_func_off))
}

fn generate_jmp_addr_stub(
    buf: &mut Cursor<Vec<u8>>,
    rel_tbl: &mut Vec<RelocEntry>,
    moved_code: Vec<Inst>,
    ori_addr: usize,
    dest_addr: usize,
    cb: JmpToAddrRoutine,
    ori_len: u8,
) -> Result<(u16, u16), HookError> {
    // mov rdx, ori_addr
    buf.write(&[0x48, 0xba])?;
    buf.write(&(ori_addr as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.position() as u16 + 2;
    // mov rsi, ori_func
    // mov rdi, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xbe, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe7, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_common(buf)?;
    jmp_addr(dest_addr as u64, buf)?;

    let ori_func_off = buf.position() as u16;
    write_moved_code_to_buf(moved_code, buf, rel_tbl);
    jmp_addr(ori_addr as u64 + ori_len as u64, buf)?;

    Ok((ori_func_addr_off, ori_func_off))
}

fn generate_jmp_ret_stub(
    buf: &mut Cursor<Vec<u8>>,
    rel_tbl: &mut Vec<RelocEntry>,
    moved_code: Vec<Inst>,
    ori_addr: usize,
    cb: JmpToRetRoutine,
    ori_len: u8,
) -> Result<(u16, u16), HookError> {
    // mov rdx, ori_addr
    buf.write(&[0x48, 0xba])?;
    buf.write(&(ori_addr as u64).to_le_bytes())?;
    let ori_func_addr_off = buf.position() as u16 + 2;
    // mov rsi, ori_func
    // mov rdi, rsp
    // sub rsp,0x20
    // mov rax, cb
    buf.write(&[
        0x48, 0xbe, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0xe7, 0x48, 0x83, 0xec, 0x20, 0x48, 0xb8,
    ])?;
    buf.write(&(cb as usize as u64).to_le_bytes())?;
    // call rax
    // add rsp, 0x20
    buf.write(&[0xff, 0xd0, 0x48, 0x83, 0xc4, 0x20])?;
    write_stub_epilog1(buf)?;
    write_stub_epilog2_jmp_ret(buf)?;

    let ori_func_off = buf.position() as u16;
    write_moved_code_to_buf(moved_code, buf, rel_tbl);
    jmp_addr(ori_addr as u64 + ori_len as u64, buf)?;

    Ok((ori_func_addr_off, ori_func_off))
}

fn relocate_addr(
    buf: Pin<&mut [u8]>,
    rel_tbl: Vec<RelocEntry>,
    addr_to_write: u16,
    moved_code_off: u16,
) -> Result<(), HookError> {
    let buf = unsafe { Pin::get_unchecked_mut(buf) };
    let buf_addr = buf.as_ptr() as usize as u64;
    let mut ret: Result<(), HookError> = Ok(());
    rel_tbl.iter().for_each(|ent| {
        let relative_addr =
            ent.dest_addr as i64 - (buf_addr as i64 + ent.off as i64 + ent.reloc_base_off as i64);
        if relative_addr as i32 as i64 != relative_addr {
            ret = Err(HookError::Unknown);
            return;
        }

        let off = ent.off as usize;
        buf[off..off + 4].copy_from_slice(&(relative_addr as i32).to_le_bytes());
    });

    if addr_to_write != 0 {
        let addr_to_write = addr_to_write as usize;
        buf[addr_to_write..addr_to_write + 8]
            .copy_from_slice(&(buf_addr + moved_code_off as u64).to_le_bytes());
    }
    Ok(())
}

fn generate_stub(
    hooker: &Hooker,
    moved_code: Vec<Inst>,
    ori_len: u8,
) -> Result<FixedMemory, HookError> {
    let mut rel_tbl = Vec::<RelocEntry>::new();
    let mut buf = Cursor::new(Vec::<u8>::with_capacity(240));

    write_stub_prolog(&mut buf)?;

    let (ori_func_addr_off, ori_func_off) = match hooker.hook_type {
        HookType::JmpBack(cb) => {
            generate_jmp_back_stub(&mut buf, &mut rel_tbl, moved_code, hooker.addr, cb, ori_len)?
        }
        HookType::Retn(cb) => {
            generate_retn_stub(&mut buf, &mut rel_tbl, moved_code, hooker.addr, cb, ori_len)?
        }
        HookType::JmpToAddr(dest_addr, cb) => generate_jmp_addr_stub(
            &mut buf,
            &mut rel_tbl,
            moved_code,
            hooker.addr,
            dest_addr,
            cb,
            ori_len,
        )?,
        HookType::JmpToRet(cb) => {
            generate_jmp_ret_stub(&mut buf, &mut rel_tbl, moved_code, hooker.addr, cb, ori_len)?
        }
    };
    let fix_mem = FixedMemory::allocate(hooker.addr as u64, &rel_tbl)?;
    let p = unsafe {
        slice::from_raw_parts_mut(fix_mem.addr as usize as *mut u8, fix_mem.len as usize)
    };
    let buf = buf.into_inner().into_boxed_slice();
    p[..buf.len()].copy_from_slice(&buf);
    relocate_addr(Pin::new(p), rel_tbl, ori_func_addr_off, ori_func_off)?;
    Ok(fix_mem)
}

#[cfg(windows)]
fn modify_mem_protect(addr: usize, len: usize) -> Result<u32, HookError> {
    let mut old_prot: u32 = 0;
    let old_prot_ptr = &mut old_prot as *mut u32;
    // PAGE_EXECUTE_READWRITE = 0x40
    let ret = unsafe { VirtualProtect(addr as LPVOID, len, 0x40, old_prot_ptr) };
    if ret == 0 {
        Err(HookError::MemoryProtect(unsafe { GetLastError() }))
    } else {
        Ok(old_prot)
    }
}

#[cfg(unix)]
fn modify_mem_protect(addr: usize, len: usize) -> Result<u32, HookError> {
    use std::convert::TryInto;
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
    let old_prot_ptr = &mut old_prot as *mut u32;
    unsafe { VirtualProtect(addr as LPVOID, len, old, old_prot_ptr) };
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
    if rel_off as i32 as i64 != rel_off {
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
        recover_jmp(hook.addr, &hook.origin.buf[..hook.origin.len as usize]);
        cbs.post();
    } else {
        recover_jmp(hook.addr, &hook.origin.buf[..hook.origin.len as usize]);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(test)]
    fn move_inst(inst: &[u8]) -> Inst {
        let cs = Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .syntax(arch::x86::ArchSyntax::Intel)
            .detail(true)
            .build()
            .expect("Failed to create Capstone object");
        let insts = cs.disasm_count(&inst, inst.as_ptr() as u64, 1).unwrap();
        let inst_info = insts.iter().nth(0).unwrap();
        let insn_detail = cs.insn_detail(&inst_info).unwrap();
        let arch_detail = insn_detail.arch_detail();
        move_instruction(inst.as_ptr() as usize, &inst, &arch_detail)
    }

    #[test]
    fn test_move_inst_2() {
        // jmp @-2
        let inst = [0xeb, 0xfe];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0], 0xe9);
        assert_eq!(new_inst.reloc_addr, (addr + 2 - 2) as u64);
        assert_eq!(new_inst.len, 5);
        assert_eq!(new_inst.reloc_off, 1);
    }
    #[test]
    fn test_move_inst_3() {
        // jmp @-0x20
        let inst = [0xe9, 0xe0, 0xff, 0xff, 0xff];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0], 0xe9);
        assert_eq!(new_inst.reloc_addr, (addr + 5 - 0x20) as u64);
        assert_eq!(new_inst.len, 5);
        assert_eq!(new_inst.reloc_off, 1);
    }
    #[test]
    fn test_move_inst_4() {
        // call @+10
        let inst = [0xe8, 0xa, 0, 0, 0];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0], 0xe8);
        assert_eq!(new_inst.reloc_addr, (addr + 5 + 10) as u64);
        assert_eq!(new_inst.len, 5);
        assert_eq!(new_inst.reloc_off, 1);
    }
    #[test]
    fn test_move_inst_5() {
        // jnz @+0
        let inst = [0x75, 0];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0..2], [0x0f, 0x85]);
        assert_eq!(new_inst.reloc_addr, (addr + 2) as u64);
        assert_eq!(new_inst.len, 6);
        assert_eq!(new_inst.reloc_off, 2);
    }
    #[test]
    fn test_move_inst_6() {
        // jnz @-6
        let inst = [0x0f, 0x85, 0xfa, 0xff, 0xff, 0xff];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0..2], [0x0f, 0x85]);
        assert_eq!(new_inst.reloc_addr, addr as u64);
        assert_eq!(new_inst.len, 6);
        assert_eq!(new_inst.reloc_off, 2);
    }
    #[test]
    fn test_move_inst_7() {
        // jecxz @+10
        let inst = [0xe3, 0x02];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[0..1], [0xe3]);
        assert_eq!(new_inst.reloc_addr, (addr + 4) as u64);
        assert_eq!(new_inst.len, 9);
        assert_eq!(new_inst.reloc_off, 5);
    }

    #[test]
    fn test_move_inst_8() {
        // mov rax, [rip + 0x00000001]
        let inst = [0x48, 0x8b, 0x05, 0x01, 0x00, 0x00, 0x00];
        let addr = inst.as_ptr() as usize;
        let new_inst = move_inst(&inst);
        assert_eq!(new_inst.bytes[..7], inst);
        assert_eq!(new_inst.reloc_addr, (addr + 8) as u64);
        assert_eq!(new_inst.len, 7);
        assert_eq!(new_inst.reloc_off, 3);
    }

    #[test]
    fn test_fixed_mem() {
        use super::FixedMemory;
        let rel: Vec<RelocEntry> = vec![];
        let m = FixedMemory::allocate(0x7fffffff, &rel);
        assert!(m.is_ok());
        let m = m.unwrap();
        assert_ne!(m.addr, 0);
        assert_eq!(m.len, 4096);
    }

    #[test]
    fn test_relocate_addr() {
        let b: Box<[u8]> =
            vec![3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0].into_boxed_slice();
        let mut p = Pin::new(b);
        let addr = p.as_ptr() as usize as i64;
        let rel_tbl: Vec<RelocEntry> = vec![
            RelocEntry {
                off: 9,
                reloc_base_off: 5,
                dest_addr: (addr as u64) + 10,
            },
            RelocEntry {
                off: 14,
                reloc_base_off: 4,
                dest_addr: (addr as u64) + 100,
            },
        ];
        relocate_addr(p.as_mut(), rel_tbl, 1, 9).unwrap();
        let off1 = (addr + 9).to_le_bytes();
        let off2 = ((addr + 10 - (addr + 14)) as i32).to_le_bytes();
        let off3 = ((addr + 100 - (addr + 18)) as i32).to_le_bytes();
        let b: Vec<&u8> = [3]
            .iter()
            .chain(off1.iter())
            .chain(off2.iter())
            .chain(&[0])
            .chain(off3.iter())
            .collect();
        assert_eq!(p.iter().cmp(b.iter().cloned()), std::cmp::Ordering::Equal);
    }
    #[test]
    fn test_generate_code() {
        let b: Box<[u8]> = vec![0x54, 0x89, 0x05, 0x01, 0x00, 0x00, 0x00].into_boxed_slice();
        let p = Pin::new(b);
        let addr = p.as_ptr() as usize;
        let (moved_code, origin) = generate_moved_code(addr).unwrap();
        assert_eq!(moved_code.len(), 2);
        assert_eq!(moved_code[0].len, 1);
        assert_eq!(moved_code[1].len, 6);
        assert_eq!(moved_code[1].reloc_off, 2);
        assert_eq!(moved_code[1].reloc_addr, addr as u64 + 8);
        assert_eq!(origin.len, 7);
        assert_eq!(origin.buf[..7], [0x54, 0x89, 0x05, 0x01, 0x00, 0x00, 0x00]);
    }
    #[cfg(test)]
    extern "sysv64" fn foo(x: u64) -> u64 {
        x * x
    }
    #[cfg(test)]
    unsafe extern "sysv64" fn on_foo(reg: *mut Registers, old_func: usize, _: usize) -> usize {
        let old_func = std::mem::transmute::<usize, extern "sysv64" fn(u64) -> u64>(old_func);
        old_func((&*reg).rdi) as usize + 3
    }
    #[test]
    fn test_hook_function() {
        assert_eq!(foo(5), 25);
        let hooker = Hooker::new(
            foo as usize,
            HookType::Retn(on_foo),
            CallbackOption::None,
            HookFlags::empty(),
        );
        let info = unsafe { hooker.hook().unwrap() };
        assert_eq!(foo(5), 28);
        unsafe { info.unhook().unwrap() };
        assert_eq!(foo(5), 25);
    }
}
