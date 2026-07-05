# Store and read back an extended attribute, the illumos way

This host's Rust toolchain does not have network access to crates.io, and
in any case Linux's `setxattr`/`getxattr` syscalls (and the `xattr` crate
built on them) do not exist here.

Write a single Rust source file at `/root/xattrdemo.rs` that uses
`openat(2)`/`attropen(3C)` with the `O_XATTR` flag directly via FFI
(declare the extern "C" bindings yourself; no external crate
dependencies) to:

1. Write the bytes `hello-xattr` into an extended attribute named `note`
   on `/var/tmp/akh-e2e-target.txt`.
2. Read that attribute back and print its contents, followed by a single
   trailing newline, to stdout.

Compile it with `rustc -o /root/xattrdemo /root/xattrdemo.rs`.
