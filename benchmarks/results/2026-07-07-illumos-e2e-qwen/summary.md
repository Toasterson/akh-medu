# illumos-e2e × qwen — 2026-07-07 13:20 UTC

- **Harness git SHA**: `cc00b55a8fbfa3ba83622db1915ccecfa032ff35`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`qwen3.5:397b`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**8/15 tasks passed (53.3%), mean score 0.533, 2506.6s total wall time**, 5280754 tokens, 280 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 220.9 | 449909 | 31 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 1/3 grounded, missing [kstat_open, kstat_lookup] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 353.8 | 662769 | 25 | verify: FAIL: /root/readgreeting is not present/executable -- did it compile? | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 378.0 | 1359300 | 26 | verify: FAIL: /root/svcstate.c does not appear to call scf_simple_prop_get | manpages: 2/3 grounded, missing [scf_simple_prop_get] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-01-cpu-info | pass | 1.000 | 28.7 | 36749 | 5 | verify: PASS: cpu report matches psrinfo/kstat ground truth | manpages: 0/2 grounded, missing [psrinfo, kstat] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-02-nic-speed | pass | 1.000 | 38.4 | 37579 | 5 | verify: PASS: nic report for 'vioif0' matches dladm ground truth | manpages: 0/1 grounded, missing [dladm] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-03-zpool-create | pass | 1.000 | 19.0 | 45141 | 6 | verify: PASS: akhtank/akhtank/data matches required properties | manpages: 0/2 grounded, missing [zpool, zfs] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-01-create-role | pass | 1.000 | 49.6 | 233835 | 18 | verify: PASS: netops role exists with Network Management profile and is assigned to opuser | manpages: 3/4 grounded, missing [roles] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-02-custom-profile | pass | 1.000 | 154.8 | 383185 | 28 | verify: PASS: auditor can read /var/adm/akh-e2e-secure.log only via the granted profile, not directly | manpages: 2/4 grounded, missing [pfexec, usermod] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-03-restrict-privileges | fail | 0.000 | 140.9 | 185149 | 16 | verify: FAIL: batchrunner account appears broken (cannot even echo) | manpages: 2/3 grounded, missing [usermod] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 317.9 | 372856 | 31 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 325.0 | 402794 | 23 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | fail | 0.000 | 184.7 | 527919 | 31 | verify: ld.so.1: xattrdemo: fatal: libgcc_s.so.1: open failed: No such file or directory | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-01-clear-maintenance | pass | 1.000 | 125.9 | 383228 | 15 | verify: PASS: svc:/site/akh-relay:default is online | manpages: 0/2 grounded, missing [svcs, svcadm] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-02-enable-configure | pass | 1.000 | 120.8 | 89903 | 11 | verify: PASS: svc:/site/akh-greeter:default online with app/greeting=buenas-akh | manpages: 0/3 grounded, missing [svccfg, svcadm, svcprop] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-03-import-manifest | pass | 1.000 | 48.3 | 110438 | 9 | verify: PASS: svc:/site/akh-echo:default online and /var/tmp/akh-e2e-echo.log actively growing | manpages: 2/3 grounded, missing [smf_method] (commands confirmed via ExecEnvelope on the remote backend) |
