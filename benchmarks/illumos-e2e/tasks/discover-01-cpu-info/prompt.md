# Report CPU count and clock speed

Using native tools on this host (do not assume Linux tools like `lscpu`
or a text `/proc/cpuinfo` exist -- they don't), determine:

1. How many CPUs are online.
2. The clock speed, in MHz, of CPU 0.

Write exactly two lines to `/var/tmp/akh-cpu-report.txt` in this form:

    cpus=<n>
    clock_mhz=<n>

where `<n>` is a bare integer in each case.
