# Count CPUs with libkstat, not psrinfo

Write a C program at `/root/cpucount.c` that uses libkstat
(`kstat_open`/`kstat_lookup`/walking `kc_chain` -- see kstat(3KSTAT)) to
count the number of distinct `cpu_info` kstat instances on this host. Do
not shell out to `psrinfo`, `kstat`, or `/proc`.

Compile it (linking libkstat) to produce the binary `/root/cpucount`, and
make sure running `/root/cpucount` prints just the integer count, followed
by a newline, to stdout.
