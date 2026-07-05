# Read an SMF instance's state with libscf, not svcs

Write a C program at `/root/svcstate.c` that uses libscf (see
libscf(3lib), smf(7)) to look up the current restarter state
(`restarter/state` property) of an SMF instance FMRI given as `argv[1]`,
and prints that state string (e.g. `online`, `maintenance`, `disabled`),
followed by a newline, to stdout. Do not shell out to `svcs` or `svcprop`.

Compile it (linking libscf) to produce `/root/svcstate`, then run it
against `svc:/site/akh-guard:default` and leave that as the last thing you
ran with it.
