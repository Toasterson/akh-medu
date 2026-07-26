# Doors round trip

Write a single Rust source file at `/root/doorping.rs` that, within one
process, uses the illumos Doors API directly via FFI (`door_create(3C)`,
`door_call(3C)`, `door_return(3C)` -- declare the extern "C" bindings
yourself; do not use any wrapper crate and do not add external crate
dependencies) to:

1. Create a door whose server procedure takes whatever bytes it is
   called with, appends the ASCII bytes `-pong` to them, and returns the
   result via `door_return`.
2. From the same process's main thread, call that door (`door_call`) with
   the ASCII bytes `ping`.
3. Print the bytes returned by the door call to stdout (no extra
   whitespace beyond a single trailing newline).

Compile it with `rustc -o /root/doorping /root/doorping.rs`. Running
`/root/doorping` should print exactly `ping-pong`.
