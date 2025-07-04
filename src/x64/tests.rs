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
    // jmp @+0
    let inst = [0xeb, 0x00];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // jmp [rip@0x400002]
    // jmp @+13
    assert_eq!(
        new_inst,
        [
            0xff, 0x25, 0x0a, 0x00, 0x00, 0x00, 0xe9, 0x0d, 0x00, 0x00, 0x00, 0xcc, 0xcc, 0xcc,
            0xcc, 0xcc, 0x02, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00
        ]
    );
}

#[test]
fn test_move_inst_long_2() {
    // jmp qword ptr [rip@400006]
    let inst = [0xff, 0x25, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // mov [rsp-0x10], rax;
    // mov rax, 400006;
    // push [rax];
    // mov rax, [rsp-8];
    // ret
    assert_eq!(
        new_inst,
        [
            0x48, 0x89, 0x44, 0x24, 0xf0, 0x48, 0xb8, 0x06, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0xff, 0x30, 0x48, 0x8b, 0x44, 0x24, 0xf8, 0xc3
        ]
    );
}

#[test]
fn test_move_inst_long_3() {
    // jne @+0
    let inst = [0x75, 0x00];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // je @+6
    // jmp [rip@0x400002]
    // jmp @+11
    assert_eq!(
        new_inst,
        [
            0x74, 0x06, 0xff, 0x25, 0x08, 0x00, 0x00, 0x00, 0xe9, 0x0b, 0x00, 0x00, 0x00, 0xcc,
            0xcc, 0xcc, 0x02, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00
        ]
    );
}

#[test]
fn test_move_inst_long_4() {
    // jrcxz @+0
    let inst = [0xe3, 0x00];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // jrcxz @+2
    // jmp @+6
    // jmp [rip@400002]
    // jmp @+9
    assert_eq!(
        new_inst,
        [
            0xe3, 0x02, 0xeb, 0x06, 0xff, 0x25, 0x06, 0x00, 0x00, 0x00, 0xe9, 0x09, 0x00, 0x00,
            0x00, 0xcc, 0x02, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00
        ]
    );
}

#[test]
fn test_move_inst_long_5() {
    // call @+0
    let inst = [0xe8, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // call [rip@400005]
    // jmp @+13
    assert_eq!(
        new_inst,
        [
            0xff, 0x15, 0x0a, 0x00, 0x00, 0x00, 0xe9, 0x0d, 0x00, 0x00, 0x00, 0xcc, 0xcc, 0xcc,
            0xcc, 0xcc, 0x05, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00
        ]
    );
}

#[test]
fn test_move_inst_long_6() {
    // call [rip@400006]
    let inst = [0xff, 0x15, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // mov [rsp-0x18], rax
    // mov rax, 400006
    // push 400024
    // mov dword ptr [rsp+4], 1
    // push qword ptr [rax]
    // mov rax, [rsp-8]
    // ret
    assert_eq!(
        new_inst,
        [
            0x48, 0x89, 0x44, 0x24, 0xe8, 0x48, 0xb8, 0x06, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x68, 0x24, 0x00, 0x40, 0x00, 0xc7, 0x44, 0x24, 0x04, 0x01, 0x00, 0x00, 0x00,
            0xff, 0x30, 0x48, 0x8b, 0x44, 0x24, 0xf8, 0xc3
        ]
    );
}

#[test]
fn test_move_inst_long_7() {
    // lea r11, [rip@400007]
    let inst = [0x4c, 0x8d, 0x1d, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // mov r11, 400007
    assert_eq!(
        new_inst,
        [0x49, 0xbb, 0x07, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00]
    );
}

#[test]
fn test_move_inst_long_8() {
    // add dword ptr [rip@400006], ebx
    let inst = [0x01, 0x1d, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // mov [rsp-0x10], r8
    // mov r8, 400006
    // add [r8], ebx
    // mov r8, [rsp-0x10]
    assert_eq!(
        new_inst,
        [
            0x4c, 0x89, 0x44, 0x24, 0xf0, 0x49, 0xb8, 0x06, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x41, 0x01, 0x18, 0x4c, 0x8b, 0x44, 0x24, 0xf0
        ]
    );
}

#[test]
fn test_move_inst_long_9() {
    // push qword ptr [rip@400006]
    let inst = [0xff, 0x35, 0, 0, 0, 0];
    let addr = 0x40_0000;
    let new_inst = move_inst(&inst, addr, addr + 0x1_0000_0000);
    // mov [rsp-0x10], rbx
    // mov rbx, 400006
    // push [rbx]
    // mov rbx, [rsp-8]
    assert_eq!(
        new_inst,
        [
            0x48, 0x89, 0x5c, 0x24, 0xf0, 0x48, 0xbb, 0x06, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0xff, 0x33, 0x48, 0x8b, 0x5c, 0x24, 0xf8
        ]
    );
}

#[test]
fn test_move_inst_long_all() {
    let inst = [
        0x74, 0x09, 0x48, 0x8B, 0x4D, 0x70, 0xE8, 0x72, 0x15, 0xF4, 0xFF, 0x8B, 0x1D, 0xEC, 0xFF,
        0xFF, 0xFF,
    ];
    let addr = 0x7fff_b81c_0a03;
    let new_inst = move_inst(&inst, addr, 0x400000);
    assert_eq!(
        new_inst,
        [
            0x75, 0x06, 0xff, 0x25, 0x28, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x4d, 0x70, 0xff, 0x15,
            0x26, 0x00, 0x00, 0x00, 0x4c, 0x89, 0x44, 0x24, 0xf0, 0x49, 0xb8, 0x00, 0x0a, 0x1c,
            0xb8, 0xff, 0x7f, 0x00, 0x00, 0x41, 0x8b, 0x18, 0x4c, 0x8b, 0x44, 0x24, 0xf0, 0xe9,
            0x12, 0x00, 0x00, 0x00, 0xcc, 0xcc, 0x12, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x80, 0x1f, 0x10, 0xb8, 0xff, 0x7f, 0x00, 0x00
        ]
    );
}

#[cfg(test)]
#[inline(never)]
// 5 arguments to ensure using stack instead of registers to pass parameter(s)
extern "win64" fn foo(x: u64, _: u64, _: u64, _: u64, y: u64) -> u64 {
    println!("original foo, x:{}, y:{}", x, y);
    x * x + y
}
#[cfg(test)]
unsafe extern "win64" fn on_foo(reg: *mut Registers, old_func: usize, user_data: usize) -> usize {
    let old_func =
        unsafe { std::mem::transmute::<usize, extern "win64" fn(u64, u64, u64, u64, u64) -> u64>(old_func) };
    let arg_y = (unsafe { (*reg).rsp } + 0x28) as *const u64;
    old_func(unsafe { (*reg).rcx }, 0, 0, 0, unsafe { *arg_y }) as usize + user_data
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
