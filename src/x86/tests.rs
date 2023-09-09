#[allow(unused_imports)]
use super::*;

#[cfg(test)]
#[inline(never)]
fn foo(x: u32) -> u32 {
    println!("original foo, x:{}", x);
    x * x
}

#[cfg(test)]
unsafe extern "cdecl" fn on_foo(
    reg: *mut Registers,
    old_func: usize,
    user_data: usize,
) -> usize {
    let old_func = std::mem::transmute::<usize, fn(u32) -> u32>(old_func);
    old_func((*reg).get_arg(1)) as usize + user_data
}

#[test]
fn test_hook_function_cdecl() {
    assert_eq!(foo(5), 25);
    let hooker = Hooker::new(
        foo as usize,
        HookType::Retn(0, on_foo),
        HookOptions::default(),
        100,
    );
    let info = unsafe { hooker.hook().unwrap() };
    assert_eq!(foo(5), 125);
    unsafe { info.unhook().unwrap() };
    assert_eq!(foo(5), 25);
}

#[cfg(test)]
#[inline(never)]
extern "stdcall" fn foo2(x: u32) -> u32 {
    println!("original foo, x:{}", x);
    x * x
}
#[cfg(test)]
unsafe extern "cdecl" fn on_foo2(
    reg: *mut Registers,
    old_func: usize,
    user_data: usize,
) -> usize {
    let old_func = std::mem::transmute::<usize, extern "stdcall" fn(u32) -> u32>(old_func);
    old_func((*reg).get_arg(1)) as usize + user_data
}
#[test]
fn test_hook_function_stdcall() {
    assert_eq!(foo2(5), 25);
    let hooker = Hooker::new(
        foo2 as usize,
        HookType::Retn(4, on_foo2),
        HookOptions::default(),
        100,
    );
    let info = unsafe { hooker.hook().unwrap() };
    assert_eq!(foo2(5), 125);
    unsafe { info.unhook().unwrap() };
    assert_eq!(foo2(5), 25);
}
