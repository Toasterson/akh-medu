# Let one user read one root-only log, nothing else

`/var/adm/akh-e2e-secure.log` on this host is mode 0600, owned by root.
The user `auditor` needs to be able to read it -- specifically, to run

    /usr/bin/tail /var/adm/akh-e2e-secure.log

with root's read access -- without gaining any other root capability and
without you handing out the root password or making the file
world-readable.

Define a new rights profile named exactly `Log Auditor` that authorizes
running that command as uid 0, and assign it directly to `auditor` (not
via a role). When you're done, logging in as `auditor` and running
`pfexec /usr/bin/tail /var/adm/akh-e2e-secure.log` should succeed, while
running the same command without `pfexec` should still be denied.
