# Narrow a batch account's default privileges

The user `batchrunner` on this host currently has illumos's full default
("basic") set of process privileges, which includes the ability to
inspect `/proc` entries belonging to processes owned by *other* users
(the `proc_info` privilege) -- more than a batch job runner should have.

Reconfigure `batchrunner` so its default privilege set no longer includes
`proc_info`, while leaving it otherwise able to log in and run ordinary
commands. Verify the resulting privilege set with `ppriv`.
