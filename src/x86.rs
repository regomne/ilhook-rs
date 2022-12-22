use bitflags::bitflags;
use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, Instruction, InstructionBlock,
};
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::slice;

#[cfg(windows)]
use core::ffi::c_void;
#[cfg(windows)]
use windows_sys::Win32::Foundation::GetLastError;
#[cfg(windows)]
use windows_sys::Win32::System::Memory::VirtualProtect;

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
/// * `src_addr` - The address that has been hooked
pub type JmpBackRoutine = unsafe extern "cdecl" fn(regs: *mut Registers, src_addr: usize);

/// The routine used in a `function hook`, which means the Routine will replace the
/// original FUNCTION, and the EIP will `retn` directly instead of jumping back.
/// Note that the being-hooked address must be the head of a function.
///
/// # Arguments
///
/// * regs - The registers
/// * `ori_func_ptr` - Original function pointer. Call it after converted to the original function type.
/// * `src_addr` - The address that has been hooked
///
/// Return the new return value of the replaced function.
pub type RetnRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

/// The routine used in a `jmp-addr hook`, which means the EIP will jump to the specified
/// address after the Routine being run.
///
/// # Arguments
///
/// * regs - The registers
/// * `ori_func_ptr` - Original function pointer. Call it after converted to the original function type.
/// * `src_addr` - The address that has been hooked
pub type JmpToAddrRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize);

/// The routine used in a `jmp-ret hook`, which means the EIP will jump to the return
/// value of the Routine.
///
/// # Arguments
///
/// * regs - The registers
/// * `ori_func_ptr` - Original function pointer. Call it after converted to the original function type.
/// * `src_addr` - The address that has been hooked
///
/// Return the address you want to jump to.
pub type JmpToRetRoutine =
    unsafe extern "cdecl" fn(regs: *mut Registers, ori_func_ptr: usize, src_addr: usize) -> usize;

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
    /// # Arguments
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

/// The hook result returned by `Hooker::hook`.
pub struct HookPoint {
    addr: usize,
    stub: Box<[u8; 100]>,
    stub_prot: u32,
    origin: Vec<u8>,
    thread_cb: CallbackOption,
    flags: HookFlags,
}

#[cfg(not(target_arch = "x86"))]
fn env_lock() {
    panic!("This crate should only be used in arch x86_32!")
}
#[cfg(target_arch = "x86")]
fn env_lock() {}

impl Hooker {
    /// Create a new Hooker.
    ///
    /// # Arguments
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
        flags: HookFlags,
    ) -> Self {
        env_lock();
        Self {
            addr,
            hook_type,
            thread_cb,
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
    /// 3. Wrong Retn-val if `hook_type` is `HookType::Retn`. i.e. A `cdecl` function with non-zero retn-val, or a `stdcall` function with wrong retn-val.
    /// 4. Set `NOT_MODIFY_MEMORY_PROTECT` where it should not be set.
    /// 5. hook or unhook from 2 or more threads at the same time without `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Because of memory protection colliding.
    /// 6. Other unpredictable errors.
    pub unsafe fn hook(self) -> Result<HookPoint, HookError> {
        let (moving_insts, origin) = get_moving_insts(self.addr)?;
        let stub = generate_stub(&self, moving_insts, origin.len() as u8)?;
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
            ret = recover_jmp_with_thread_cb(self);
            recover_mem_protect(self.addr, JMP_INST_SIZE, old_prot);
        } else {
            ret = recover_jmp_with_thread_cb(self)
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

fn write_relative_off<T: Write + Seek>(
    buf: &mut T,
    base_addr: u32,
    dst_addr: u32,
) -> Result<(), HookError> {
    let dst_addr = dst_addr as i32;
    let cur_pos = buf.stream_position().unwrap() as i32;
    let call_off = dst_addr - (base_addr as i32 + cur_pos + 4);
    buf.write(&call_off.to_le_bytes())?;
    Ok(())
}

fn move_code_to_addr(ori_insts: &Vec<Instruction>, dest_addr: u32) -> Result<Vec<u8>, HookError> {
    let block = InstructionBlock::new(ori_insts, u64::from(dest_addr));
    let encoded = BlockEncoder::encode(32, block, BlockEncoderOptions::NONE)
        .map_err(|_| HookError::MoveCode)?;
    Ok(encoded.code_buffer)
}

fn write_ori_func_addr<T: Write + Seek>(buf: &mut T, ori_func_addr_off: u32, ori_func_off: u32) {
    let pos = buf.stream_position().unwrap();
    buf.seek(SeekFrom::Start(u64::from(ori_func_addr_off)))
        .unwrap();
    buf.write(&ori_func_off.to_le_bytes()).unwrap();
    buf.seek(SeekFrom::Start(pos)).unwrap();
}

fn generate_jmp_back_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    cb: JmpBackRoutine,
    ori_len: u8,
) -> Result<(), HookError> {
    // push hooker.addr
    buf.write(&[0x68])?;
    buf.write(&ori_addr.to_le_bytes())?;

    // push ebp (Registers)
    // call XXXX (dest addr)
    buf.write(&[0x55, 0xe8])?;
    write_relative_off(buf, stub_base_addr, cb as u32)?;

    // add esp, 0x8
    buf.write(&[0x83, 0xc4, 0x08])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(moving_code, stub_base_addr + cur_pos)?)?;
    // jmp back
    buf.write(&[0xe9])?;
    write_relative_off(buf, stub_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_retn_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    retn_val: u16,
    cb: RetnRoutine,
    ori_len: u8,
) -> Result<(), HookError> {
    // push hooker.addr
    buf.write(&[0x68])?;
    buf.write(&ori_addr.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, stub_base_addr, cb as u32)?;

    // add esp, 0xc
    buf.write(&[0x83, 0xc4, 0x0c])?;
    // mov [esp+20h], eax
    buf.write(&[0x89, 0x44, 0x24, 0x20])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;
    if retn_val == 0 {
        // retn
        buf.write(&[0xc3])?;
    } else {
        // retn XX
        buf.write(&[0xc2])?;
        buf.write(&retn_val.to_le_bytes())?;
    }
    let ori_func_off = buf.stream_position().unwrap() as u32;
    write_ori_func_addr(buf, ori_func_addr_off as u32, stub_base_addr + ori_func_off);

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(moving_code, stub_base_addr + cur_pos)?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, stub_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_jmp_addr_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    dest_addr: u32,
    cb: JmpToAddrRoutine,
    ori_len: u8,
) -> Result<(), HookError> {
    // push hooker.addr
    buf.write(&[0x68])?;
    buf.write(&ori_addr.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, stub_base_addr, cb as u32)?;

    // add esp, 0xc
    buf.write(&[0x83, 0xc4, 0x0c])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;
    // jmp back
    buf.write(&[0xe9])?;
    write_relative_off(buf, stub_base_addr, dest_addr + u32::from(ori_len))?;

    let ori_func_off = buf.stream_position().unwrap() as u32;
    write_ori_func_addr(buf, ori_func_addr_off as u32, stub_base_addr + ori_func_off);

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(moving_code, stub_base_addr + cur_pos)?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, stub_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_jmp_ret_stub<T: Write + Seek>(
    buf: &mut T,
    stub_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    cb: JmpToRetRoutine,
    ori_len: u8,
) -> Result<(), HookError> {
    // push hooker.addr
    buf.write(&[0x68])?;
    buf.write(&ori_addr.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, stub_base_addr, cb as u32)?;

    // add esp, 0xc
    buf.write(&[0x83, 0xc4, 0x0c])?;
    // mov [esp-4], eax
    buf.write(&[0x89, 0x44, 0x24, 0xfc])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;
    // jmp dword ptr [esp-0x28]
    buf.write(&[0xff, 0x64, 0x24, 0xd8])?;

    let ori_func_off = buf.stream_position().unwrap() as u32;
    write_ori_func_addr(buf, ori_func_addr_off as u32, stub_base_addr + ori_func_off);

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(moving_code, stub_base_addr + cur_pos)?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, stub_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_stub(
    hooker: &Hooker,
    moving_code: Vec<Instruction>,
    ori_len: u8,
) -> Result<Box<[u8; 100]>, HookError> {
    let mut raw_buffer = Box::new([0u8; 100]);
    let stub_addr = raw_buffer.as_ptr() as u32;
    let mut buf = Cursor::new(&mut raw_buffer[..]);

    // pushad
    // pushfd
    // mov ebp, esp
    buf.write(&[0x60, 0x9c, 0x8b, 0xec])?;

    match hooker.hook_type {
        HookType::JmpBack(cb) => generate_jmp_back_stub(
            &mut buf,
            stub_addr,
            &moving_code,
            hooker.addr as u32,
            cb,
            ori_len,
        ),
        HookType::Retn(val, cb) => generate_retn_stub(
            &mut buf,
            stub_addr,
            &moving_code,
            hooker.addr as u32,
            val as u16,
            cb,
            ori_len,
        ),
        HookType::JmpToAddr(dest, cb) => generate_jmp_addr_stub(
            &mut buf,
            stub_addr,
            &moving_code,
            hooker.addr as u32,
            dest as u32,
            cb,
            ori_len,
        ),
        HookType::JmpToRet(cb) => generate_jmp_ret_stub(
            &mut buf,
            stub_addr,
            &moving_code,
            hooker.addr as u32,
            cb,
            ori_len,
        ),
    }?;

    Ok(raw_buffer)
}

fn modify_jmp(dest_addr: usize, stub_addr: usize) -> Result<(), HookError> {
    let buf = unsafe { slice::from_raw_parts_mut(dest_addr as *mut u8, JMP_INST_SIZE) };
    // jmp stub_addr
    buf[0] = 0xe9;
    let rel_off = stub_addr as i32 - (dest_addr as i32 + 5);
    buf[1..5].copy_from_slice(&rel_off.to_le_bytes());
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

#[cfg(target_arch = "x86")]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(test)]
    fn foo(x: u32) -> u32 {
        x * x
    }
    #[cfg(test)]
    unsafe extern "cdecl" fn on_foo(reg: *mut Registers, old_func: usize, _: usize) -> usize {
        let old_func = std::mem::transmute::<usize, fn(u32) -> u32>(old_func);
        old_func((*reg).get_arg(1)) as usize + 3
    }

    #[test]
    fn test_hook_function_cdecl() {
        assert_eq!(foo(5), 25);
        let hooker = Hooker::new(
            foo as usize,
            HookType::Retn(0, on_foo),
            CallbackOption::None,
            HookFlags::empty(),
        );
        let info = unsafe { hooker.hook().unwrap() };
        assert_eq!(foo(5), 28);
        unsafe { info.unhook().unwrap() };
        assert_eq!(foo(5), 25);
    }

    #[cfg(test)]
    extern "stdcall" fn foo2(x: u32) -> u32 {
        x * x
    }
    #[cfg(test)]
    unsafe extern "cdecl" fn on_foo2(reg: *mut Registers, old_func: usize, _: usize) -> usize {
        let old_func = std::mem::transmute::<usize, extern "stdcall" fn(u32) -> u32>(old_func);
        old_func((*reg).get_arg(1)) as usize + 3
    }
    #[test]
    fn test_hook_function_stdcall() {
        assert_eq!(foo2(5), 25);
        let hooker = Hooker::new(
            foo2 as usize,
            HookType::Retn(4, on_foo2),
            CallbackOption::None,
            HookFlags::empty(),
        );
        let info = unsafe { hooker.hook().unwrap() };
        assert_eq!(foo2(5), 28);
        unsafe { info.unhook().unwrap() };
        assert_eq!(foo2(5), 25);
    }
}
