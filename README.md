ilhook
----

[![](http://meritbadge.herokuapp.com/ilhook)](https://crates.io/crates/ilhook)

This crate provides methods to inline hook binary codes of `x86` and `x64` instruction sets.

HOOK is a mechanism that intercepts function calls and handles them by user-defined code.

# Installation

This crate works with Cargo and is on
[crates.io](https://crates.io/crates/ilhook). Add it to your `Cargo.toml`
like so:

```toml
[dependencies]
ilhook = "2"
```

# Hook Types

Ilhook supports 4 types of hooking.

## Jmp-back hook

This type is used when you want to get some information, or modify some values
(parameters, stack vars, heap vars, etc.) at the specified timing.

Assume we have a C++ function:

```cpp
void check_serial_number(std::string& sn){
    uint32_t machine_hash = get_machine_hash();
    uint32_t sn_hash = calc_hash(sn);

    // we want to modify the result of this comparison.
    if (sn_hash == machine_hash) {
        // success
    }
    // fail
}
```

And it compiles to the asm code:

```asm
0x401054 call get_machine_hash   ;get_machine_hash()
0x401059 mov ebx, eax

; ...

0x401070 lea eax, sn
0x401076 push eax
0x401077 call calc_hash          ;calc_hash(sn)
0x40107C add esp, 4
0x40107F cmp eax, ebx            ;we want to modify the eax here!
0x401081 jnz _check_fail

; check_success
```

Now let's start:

```rust
use ilhook::x86::{Hooker, HookType, Registers, CallbackOption, HookFlags};

unsafe extern "C" fn on_check_sn(
    reg: *mut Registers,
    _: usize
) {
    println!("m_hash: {}, sn_hash: {}", (*reg).ebx, (*reg).eax);
    (*reg).eax = (*reg).ebx; //we modify the sn_hash!
}

let hooker = Hooker::new(
    0x40107F,
    HookType::JmpBack(on_check_sn),
    CallbackOption::None,
    0,
    HookFlags::empty(),
);
hooker.hook().unwrap();
```

Then `check_serial_number` will always go to the successful path.

## Function hook

This type is used when you want to replace a function with your customized
function. Note that you should only hook at the beginning of a function.

Assume we have a function:

```rust
fn foo(x: u64) -> u64 {
    x * x
}

assert_eq!(foo(5), 25);
```

And you want to let it return `old_foo(x)+x`, which means foo(5)==30.

Now let's hook:

```rust
use ilhook::x64::{Hooker, HookType, Registers, CallbackOption, HookFlags};

unsafe extern "win64" fn new_foo(
    reg: *mut Registers,
    ori_func_ptr :usize,
    _ :usize
) -> usize {
    let x = (&*reg).rdi as u64;
    let ori_func: extern "win64" fn (u64) -> u64 =
        unsafe { std::mem::transmute(ori_func_ptr) };
    (ori_func(x) + x) as usize
}

let hooker = Hooker::new(
    foo as usize,
    HookType::Retn(new_foo),
    CallbackOption::None,
    0,
    HookFlags::empty(),
);
unsafe { hooker.hook().unwrap() };
assert_eq!(foo(5), 30);
```

## Jmp-addr hook

This type is used when you want to change the original run path to any other you wanted.

The first element of the enum `HookType::JmpToAddr` indicates where you want the EIP jump
to after the callback routine returns.

## Jmp-ret hook

This type is used when you want to change the original run path to any other you wanted, and
the destination address may change by the input arguments.

The EIP will jump to the value the callback routine returns.

# Closure Hooks

This crate provides utility functions for registering hooks that use Rust
closures as callbacks rather than `extern "C" fn`s. This involves more layers of
indirection, so it's slightly less efficient, especially when the callbacks
don't actually close over any data. However, it can be useful if you do have
additional data, especially since the closure's lifetime is automatically tied
to the `ClosureHookPoint` that controls when the hook is disposed of.

The closure hook functions are `Hooker::hook_closure_jmp_back`,
`Hooker::hook_closure_retn`, `Hooker::hook_closure_jmp_to_addr`, and
`Hooker::hook_closure_jmp_to_ret`. They match the behavior of the four
`HookType`s.

Here's the same `on_check_sn` example we used above, using a closure hook
instead:

```rust
use ilhook::x86::{Registers, CallbackOption, HookFlags, hook_closure_jmp_back};

let on_check_sn = |reg| {
    println!("m_hash: {}, sn_hash: {}", (*reg).ebx, (*reg).eax);
    (*reg).eax = (*reg).ebx;
};

hook_closure_jmp_back(
    0x40107F,
    on_check_sn,
    CallbackOption::None,
    HookFlags::empty(),
).unwrap();
```

# Notes

This crate is not thread-safe if you don't specify `HookFlags::NOT_MODIFY_MEMORY_PROTECT`. Of course,
you need to modify memory protection of the destination address by yourself if you specify that.

As rust's test run parrallelly, it may crash if not specify `--test-threads=1`.
