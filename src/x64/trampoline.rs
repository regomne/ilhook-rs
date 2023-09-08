use std::io::{Cursor, Seek, SeekFrom, Write};

use iced_x86::Instruction;

use super::{
    move_inst::move_code_to_addr, HookType, JmpBackRoutine, JmpToAddrRoutine, JmpToRetRoutine,
    RetnRoutine,
};
use crate::callbacks::CodeProtectModifyingCallback;
use crate::err::HookError;

const TRAMPOLINE_MAX_LEN: usize = 1024;

pub(super) struct Trampoline<'a> {
    buffer: Buffer<'a>,
    trampoline_prot: Option<u64>,
    code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

enum Buffer<'a> {
    Owned(Box<[u8; TRAMPOLINE_MAX_LEN]>),
    Borrowed(&'a mut [u8]),
}

impl<'a> Buffer<'a> {
    pub fn get(&self) -> &[u8] {
        match &self {
            Buffer::Owned(b) => b.as_slice(),
            Buffer::Borrowed(b) => b,
        }
    }
    pub fn get_mut(&mut self) -> &mut [u8] {
        match self {
            Buffer::Owned(b) => b.as_mut_slice(),
            Buffer::Borrowed(b) => b,
        }
    }
}

impl<'a> Trampoline<'a> {
    pub fn new(cb: &Option<&'a dyn CodeProtectModifyingCallback>) -> Self {
        Self {
            buffer: Buffer::Owned(Box::new([0u8; TRAMPOLINE_MAX_LEN])),
            trampoline_prot: None,
            code_protect_cb: cb.clone(),
        }
    }

    pub fn with_buffer(
        buf: &'a mut [u8],
        cb: &Option<&'a dyn CodeProtectModifyingCallback>,
    ) -> Self {
        Self {
            buffer: Buffer::Borrowed(buf),
            trampoline_prot: None,
            code_protect_cb: cb.clone(),
        }
    }

    pub fn get_addr(&self) -> usize {
        let buffer = self.buffer.get();
        buffer.as_ptr() as usize
    }

    pub fn generate(
        &mut self,
        hook_addr: usize,
        hook_type: HookType,
        moving_code: &Vec<Instruction>,
        ori_len: u8,
        user_data: usize,
    ) -> Result<(), HookError> {
        let buffer = self.buffer.get_mut();
        let buffer_len = buffer.len();
        let trampoline_addr = buffer.as_ptr() as u64;
        let mut buf = Cursor::new(&mut buffer[..]);

        write_trampoline_prolog(&mut buf)?;

        match hook_type {
            HookType::JmpBack(cb) => generate_jmp_back_trampoline(
                &mut buf,
                trampoline_addr,
                moving_code,
                hook_addr,
                cb,
                ori_len,
                user_data,
            ),
            HookType::Retn(cb) => generate_retn_trampoline(
                &mut buf,
                trampoline_addr,
                moving_code,
                hook_addr,
                cb,
                ori_len,
                user_data,
            ),
            HookType::JmpToAddr(dest_addr, cb) => generate_jmp_addr_trampoline(
                &mut buf,
                trampoline_addr,
                moving_code,
                hook_addr,
                dest_addr,
                cb,
                ori_len,
                user_data,
            ),
            HookType::JmpToRet(cb) => generate_jmp_ret_trampoline(
                &mut buf,
                trampoline_addr,
                moving_code,
                hook_addr,
                cb,
                ori_len,
                user_data,
            ),
        }?;

        self.trampoline_prot = self
            .code_protect_cb
            .as_ref()
            .map(|cb| cb.set_protect_to_rwe(trampoline_addr as usize, buffer_len))
            .map_or(Ok(None), |v| v.map(Some).map_err(HookError::MemoryProtect))?;

        Ok(())
    }
}

impl<'a> Drop for Trampoline<'a> {
    fn drop(&mut self) {
        /*let buffer = self.buffer.get_mut();
        self.trampoline_prot.map(|prot| {
            self.code_protect_cb.as_ref().unwrap().recover_protect(
                buffer.as_ptr() as usize,
                buffer.len(),
                prot,
            )
        });*/
    }
}

fn generate_jmp_back_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u64,
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
    write_trampoline_epilog1(buf)?;
    write_trampoline_epilog2_common(buf)?;

    let cur_pos = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + cur_pos,
    )?)?;

    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;
    Ok(())
}

fn generate_retn_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u64,
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
    write_trampoline_epilog1(buf)?;
    write_trampoline_epilog2_common(buf)?;
    // ret
    buf.write(&[0xc3])?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, trampoline_base_addr + ori_func_off);

    Ok(())
}

fn generate_jmp_addr_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u64,
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
    write_trampoline_epilog1(buf)?;
    write_trampoline_epilog2_common(buf)?;
    jmp_addr(dest_addr as u64, buf)?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, trampoline_base_addr + ori_func_off);

    Ok(())
}

fn generate_jmp_ret_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u64,
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
    write_trampoline_epilog1(buf)?;
    write_trampoline_epilog2_jmp_ret(buf)?;

    let ori_func_off = buf.stream_position().unwrap();
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + ori_func_off,
    )?)?;
    jmp_addr(ori_addr as u64 + u64::from(ori_len), buf)?;

    write_ori_func_addr(buf, ori_func_addr_off, trampoline_base_addr + ori_func_off);

    Ok(())
}

fn write_trampoline_prolog(buf: &mut impl Write) -> Result<usize, std::io::Error> {
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

fn write_trampoline_epilog1(buf: &mut impl Write) -> Result<usize, std::io::Error> {
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

fn write_trampoline_epilog2_common(buf: &mut impl Write) -> Result<usize, std::io::Error> {
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

fn write_trampoline_epilog2_jmp_ret(buf: &mut impl Write) -> Result<usize, std::io::Error> {
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

fn write_ori_func_addr<T: Write + Seek>(buf: &mut T, ori_func_addr_off: u64, ori_func_off: u64) {
    let pos = buf.stream_position().unwrap();
    buf.seek(SeekFrom::Start(ori_func_addr_off)).unwrap();
    buf.write(&ori_func_off.to_le_bytes()).unwrap();
    buf.seek(SeekFrom::Start(pos)).unwrap();
}
