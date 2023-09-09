use std::io::{Cursor, Seek, SeekFrom, Write};

use iced_x86::{BlockEncoder, BlockEncoderOptions, Instruction, InstructionBlock};

use super::{HookType, JmpBackRoutine, JmpToAddrRoutine, JmpToRetRoutine, RetnRoutine};
use crate::callbacks::CodeProtectModifyingCallback;
use crate::err::HookError;

const TRAMPOLINE_MAX_LEN: usize = 100;

pub(super) struct Trampoline<'a> {
    buffer: Box<[u8; TRAMPOLINE_MAX_LEN]>,
    trampoline_prot: Option<u64>,
    code_protect_cb: Option<&'a dyn CodeProtectModifyingCallback>,
}

impl<'a> Trampoline<'a> {
    pub fn new(cb: &Option<&'a dyn CodeProtectModifyingCallback>) -> Self {
        Self {
            buffer: Box::new([0u8; TRAMPOLINE_MAX_LEN]),
            trampoline_prot: None,
            code_protect_cb: cb.clone(),
        }
    }

    pub fn get_addr(&self)->usize{
        self.buffer.as_ptr() as usize
    }

    pub fn generate(
        &mut self,
        addr: usize,
        hook_type: HookType,
        moving_code: &Vec<Instruction>,
        ori_len: u8,
        user_data: usize,
    ) -> Result<(), HookError> {
        let trampoline_addr = self.buffer.as_ptr() as u32;
        let mut buf = Cursor::new(&mut self.buffer[..]);

        // pushad
        // pushfd
        // mov ebp, esp
        buf.write(&[0x60, 0x9c, 0x8b, 0xec])?;

        match hook_type {
            HookType::JmpBack(cb) => generate_jmp_back_trampoline(
                &mut buf,
                trampoline_addr,
                &moving_code,
                addr as u32,
                cb,
                ori_len,
                user_data,
            ),
            HookType::Retn(val, cb) => generate_retn_trampoline(
                &mut buf,
                trampoline_addr,
                &moving_code,
                addr as u32,
                val as u16,
                cb,
                ori_len,
                user_data,
            ),
            HookType::JmpToAddr(dest, cb) => generate_jmp_addr_trampoline(
                &mut buf,
                trampoline_addr,
                &moving_code,
                addr as u32,
                dest as u32,
                cb,
                ori_len,
                user_data,
            ),
            HookType::JmpToRet(cb) => generate_jmp_ret_trampoline(
                &mut buf,
                trampoline_addr,
                &moving_code,
                addr as u32,
                cb,
                ori_len,
                user_data,
            ),
        }?;

        self.trampoline_prot = self
            .code_protect_cb
            .as_ref()
            .map(|cb| cb.set_protect_to_rwe(trampoline_addr as usize, self.buffer.len()))
            .map_or(Ok(None), |v| v.map(Some).map_err(HookError::MemoryProtect))?;

        Ok(())
    }
}

impl<'a> Drop for Trampoline<'a> {
    fn drop(&mut self) {
        self.trampoline_prot.map(|prot| {
            self.code_protect_cb.as_ref().unwrap().recover_protect(
                self.buffer.as_ptr() as usize,
                self.buffer.len(),
                prot,
            )
        });
    }
}

fn generate_jmp_back_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    cb: JmpBackRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // push user_data
    buf.write(&[0x68])?;
    buf.write(&user_data.to_le_bytes())?;

    // push ebp (Registers)
    // call XXXX (dest addr)
    buf.write(&[0x55, 0xe8])?;
    write_relative_off(buf, trampoline_base_addr, cb as u32)?;

    // add esp, 0x8
    buf.write(&[0x83, 0xc4, 0x08])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + cur_pos,
    )?)?;
    // jmp back
    buf.write(&[0xe9])?;
    write_relative_off(buf, trampoline_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_retn_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    retn_val: u16,
    cb: RetnRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // push user_data
    buf.write(&[0x68])?;
    buf.write(&user_data.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, trampoline_base_addr, cb as u32)?;

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
    write_ori_func_addr(
        buf,
        ori_func_addr_off as u32,
        trampoline_base_addr + ori_func_off,
    );

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + cur_pos,
    )?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, trampoline_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_jmp_addr_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    dest_addr: u32,
    cb: JmpToAddrRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // push user_data
    buf.write(&[0x68])?;
    buf.write(&user_data.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, trampoline_base_addr, cb as u32)?;

    // add esp, 0xc
    buf.write(&[0x83, 0xc4, 0x0c])?;
    // popfd
    // popad
    buf.write(&[0x9d, 0x61])?;
    // jmp back
    buf.write(&[0xe9])?;
    write_relative_off(buf, trampoline_base_addr, dest_addr + u32::from(ori_len))?;

    let ori_func_off = buf.stream_position().unwrap() as u32;
    write_ori_func_addr(
        buf,
        ori_func_addr_off as u32,
        trampoline_base_addr + ori_func_off,
    );

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + cur_pos,
    )?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, trampoline_base_addr, ori_addr + u32::from(ori_len))
}

fn generate_jmp_ret_trampoline<T: Write + Seek>(
    buf: &mut T,
    trampoline_base_addr: u32,
    moving_code: &Vec<Instruction>,
    ori_addr: u32,
    cb: JmpToRetRoutine,
    ori_len: u8,
    user_data: usize,
) -> Result<(), HookError> {
    // push user_data
    buf.write(&[0x68])?;
    buf.write(&user_data.to_le_bytes())?;

    // push XXXX (original function addr)
    // push ebp (Registers)
    // call XXXX (dest addr)
    let ori_func_addr_off = buf.stream_position().unwrap() + 1;
    buf.write(&[0x68, 0, 0, 0, 0, 0x55, 0xe8])?;
    write_relative_off(buf, trampoline_base_addr, cb as u32)?;

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
    write_ori_func_addr(
        buf,
        ori_func_addr_off as u32,
        trampoline_base_addr + ori_func_off,
    );

    let cur_pos = buf.stream_position().unwrap() as u32;
    buf.write(&move_code_to_addr(
        moving_code,
        trampoline_base_addr + cur_pos,
    )?)?;

    // jmp ori_addr
    buf.write(&[0xe9])?;
    write_relative_off(buf, trampoline_base_addr, ori_addr + u32::from(ori_len))
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
