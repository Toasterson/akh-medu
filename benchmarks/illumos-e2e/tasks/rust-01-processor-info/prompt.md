# Read CPU clock speed via a direct syscall, not psrinfo/kstat

Write a single Rust source file at `/root/cpuclock.rs` that calls the
illumos `processor_info(2)` syscall directly via FFI (declare the extern
"C" binding yourself; do not shell out to `psrinfo` or `kstat`, and do not
add external crate dependencies) for processor id 0, and prints the
`pi_clock` field (clock speed in MHz) as a bare integer, followed by a
newline, to stdout.

Compile it with `rustc -o /root/cpuclock /root/cpuclock.rs`.
