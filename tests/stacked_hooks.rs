//! Regression coverage for stacking hooks at the same address (issue #26).

#[cfg(all(feature = "x64", target_arch = "x86_64"))]
mod x64 {
    use ilhook::x64::{CallbackOption, HookFlags, HookType, Hooker, Registers};
    use std::{hint::black_box, mem::transmute};

    #[inline(never)]
    extern "win64" fn target() -> usize {
        println!("original x64 target");
        black_box(7)
    }

    unsafe extern "win64" fn callback(
        _: *mut Registers,
        previous: usize,
        data: usize,
    ) -> usize {
        let previous: extern "win64" fn() -> usize = unsafe { transmute(previous) };
        previous() * 2 + data
    }

    #[test]
    fn stacked_retn_hooks_unwind_in_reverse_order() {
        let call = black_box(target as extern "win64" fn() -> usize);
        let install = |data| unsafe {
            Hooker::new(
                call as usize,
                HookType::Retn(callback),
                CallbackOption::None,
                data,
                HookFlags::empty(),
            )
            .hook()
            .unwrap()
        };

        for _ in 0..3 {
            assert_eq!(call(), 7);
            let a = install(1);
            assert_eq!(call(), 15);
            let b = install(2);
            assert_eq!(call(), 32);
            let c = install(3);
            assert_eq!(call(), 67);

            drop(c);
            assert_eq!(call(), 32);
            drop(b);
            assert_eq!(call(), 15);
            drop(a);
            assert_eq!(call(), 7);
        }
    }
}

#[cfg(all(feature = "x86", target_arch = "x86"))]
mod x86 {
    use ilhook::x86::{CallbackOption, HookFlags, HookType, Hooker, Registers};
    use std::{hint::black_box, mem::transmute};

    #[inline(never)]
    extern "cdecl" fn target() -> usize {
        println!("original x86 target");
        black_box(7)
    }

    unsafe extern "cdecl" fn callback(
        _: *mut Registers,
        previous: usize,
        data: usize,
    ) -> usize {
        let previous: extern "cdecl" fn() -> usize = unsafe { transmute(previous) };
        previous() * 2 + data
    }

    #[test]
    fn stacked_retn_hooks_unwind_in_reverse_order() {
        let call = black_box(target as extern "cdecl" fn() -> usize);
        let install = |data| unsafe {
            Hooker::new(
                call as usize,
                HookType::Retn(0, callback),
                CallbackOption::None,
                data,
                HookFlags::empty(),
            )
            .hook()
            .unwrap()
        };

        for _ in 0..3 {
            assert_eq!(call(), 7);
            let a = install(1);
            assert_eq!(call(), 15);
            let b = install(2);
            assert_eq!(call(), 32);
            let c = install(3);
            assert_eq!(call(), 67);

            drop(c);
            assert_eq!(call(), 32);
            drop(b);
            assert_eq!(call(), 15);
            drop(a);
            assert_eq!(call(), 7);
        }
    }
}
