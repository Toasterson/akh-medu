# 4 GiB boot attempt

The first project-specific overlay booted OpenIndiana Hipster 2026.04 and
reached `illumos-44e9e35ce0`, but stopped making progress while performing the
initial full ZFS device scan. SSH port forwarding was established, although
the guest never returned an SSH banner. The QEMU process used about 824 MiB
resident host memory. The attempt was stopped after more than four minutes and
the guest was recreated with the image recipe's 6144 MiB builder memory.

The full serial log is retained as `attempt-4g-console.log`.
