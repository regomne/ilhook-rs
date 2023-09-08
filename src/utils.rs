use crate::callbacks::*;
use crate::err::HookError;

pub(crate) struct MemoryProtectGuard<'a> {
    cb: Option<&'a dyn CodeProtectModifyingCallback>,
    addr: usize,
    len: usize,
}

impl<'a> MemoryProtectGuard<'a> {
    pub fn new(cb: &Option<&'a dyn CodeProtectModifyingCallback>, addr: usize, len: usize) -> Self {
        Self {
            cb: cb.clone(),
            addr,
            len,
        }
    }

    pub fn run<T, F>(self, func: F) -> Result<T, HookError>
    where
        F: Fn() -> Result<T, HookError>,
    {
        let old_protect = self
            .cb
            .map(|cb| cb.set_protect_to_rwe(self.addr, self.len))
            .map_or(Ok(None), |v| v.map(Some).map_err(HookError::MemoryProtect))?;
        let ret = func();

        old_protect.map(|prot| self.cb.unwrap().recover_protect(self.addr, self.len, prot));
        ret
    }
}

pub(crate) struct ThreadSuspendingGuard<'a> {
    cb: Option<&'a dyn ThreadOperatingCallback>,
}

impl<'a> ThreadSuspendingGuard<'a> {
    pub fn new(cb: &Option<&'a dyn ThreadOperatingCallback>) -> Self {
        Self { cb: cb.clone() }
    }

    pub fn run<T, F>(self, func: F) -> Result<T, HookError>
    where
        F: Fn() -> Result<T, HookError>,
    {
        let ctx = self
            .cb
            .map(|cb| cb.suspend())
            .map_or(Ok(None), |v| v.map(Some).map_err(HookError::ThreadSuspending))?;
        let ret = func();

        ctx.map(|ctx| self.cb.unwrap().resume(ctx));
        ret
    }
}
