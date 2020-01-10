use std::io;
use thiserror::Error;

/// Hook errors.
#[derive(Error, Debug)]
pub enum HookError {
    /// Invalid parameter
    #[error("invalid parameter")]
    InvalidParameter,

    /// Error occurs when modifying the memory protect
    #[error("memory protect error, code:{0}")]
    MemoryProtect(u32),

    #[error("memory allocation error")]
    MemoryAllocation,

    /// Can't disassemble in the specified address
    #[error("disassemble error")]
    Disassemble,

    /// The pre-hook callback failed
    #[error("pre hook failed")]
    PreHook,

    /// Some io error
    #[error("io error")]
    Io(#[from] io::Error),

    /// Unknown error
    #[error("unknown error")]
    Unknown,
}
