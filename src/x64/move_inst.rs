use std::io::{Cursor, Seek, SeekFrom, Write};

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Encoder, FlowControl, Instruction, InstructionBlock,
    MemoryOperand, Mnemonic, Register,
};

use crate::HookError;

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

pub(super) fn move_code_to_addr(
    ori_insts: &Vec<Instruction>,
    dest_addr: u64,
) -> Result<Vec<u8>, HookError> {
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
            let padding_cnt = ((cur_addr + 5 + 7) & !7) - (cur_addr + 5); // 5 bytes for jmp near below

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
            buf.write(&[0x48, 0x89, 0x44, 0x24, 0xe8, 0x48, 0xb8])?;
            buf.write(&inst.ip_rel_memory_address().to_le_bytes())?;
            let retn_addr = dest_addr + 0x24;
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
