# Read an SMF property with libscf, not svcprop

The service `svc:/site/akh-note:default` on this host has a custom
property `config/greeting`. Write a C program at `/root/readgreeting.c`
that uses libscf (`scf_simple_prop_get` + `scf_simple_prop_next_astring`,
see libscf(3lib)) to read that property's current value and print it,
followed by a newline, to stdout. Do not shell out to `svcprop` or
`svccfg`.

Compile it (linking libscf) to produce `/root/readgreeting`.
