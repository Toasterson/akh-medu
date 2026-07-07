# illumos-e2e × deepseek — 2026-07-07 12:38 UTC

- **Harness git SHA**: `9fcd85270f19aeb56618b28d3329f64e9521fe90`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`deepseek-v4-pro`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**8/15 tasks passed (53.3%), mean score 0.533, 1696.2s total wall time**, 3294157 tokens, 193 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 351.1 | 506944 | 31 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 2/3 grounded, missing [libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 23.3 | 137645 | 7 | verify: FAIL: /root/readgreeting is not present/executable -- did it compile? | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 350.5 | 984616 | 25 | verify: FAIL: /root/svcstate is not present/executable -- did it compile? | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| discover-01-cpu-info | pass | 1.000 | 8.1 | 21827 | 4 | verify: PASS: cpu report matches psrinfo/kstat ground truth | manpages: 0/2 grounded, missing [psrinfo, kstat] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-02-nic-speed | pass | 1.000 | 10.3 | 22203 | 4 | verify: PASS: nic report for 'vioif0' matches dladm ground truth | manpages: 0/1 grounded, missing [dladm] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-03-zpool-create | pass | 1.000 | 16.8 | 34522 | 6 | verify: PASS: akhtank/akhtank/data matches required properties | manpages: 0/2 grounded, missing [zpool, zfs] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-01-create-role | pass | 1.000 | 17.9 | 32009 | 5 | verify: PASS: netops role exists with Network Management profile and is assigned to opuser | manpages: 0/4 grounded, missing [roleadd, usermod, profiles, roles] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-02-custom-profile | pass | 1.000 | 42.6 | 139570 | 9 | verify: PASS: auditor can read /var/adm/akh-e2e-secure.log only via the granted profile, not directly | manpages: 0/4 grounded, missing [prof_attr, exec_attr, pfexec, usermod] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-03-restrict-privileges | fail | 0.000 | 14.0 | 28575 | 5 | verify: FAIL: batchrunner account appears broken (cannot even echo) | manpages: 0/3 grounded, missing [usermod, ppriv, privileges] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 145.1 | 319347 | 31 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 338.5 | 175670 | 12 | verify: FAIL: /root/doorping.rs does not exist | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | fail | 0.000 | 157.7 | 563030 | 31 | verify: ld.so.1: xattrdemo: fatal: libgcc_s.so.1: open failed: No such file or directory | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| smf-01-clear-maintenance | pass | 1.000 | 163.3 | 258320 | 13 | verify: PASS: svc:/site/akh-relay:default is online | manpages: 0/2 grounded, missing [svcs, svcadm] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-02-enable-configure | pass | 1.000 | 22.3 | 31779 | 5 | verify: PASS: svc:/site/akh-greeter:default online with app/greeting=buenas-akh | manpages: 0/3 grounded, missing [svccfg, svcadm, svcprop] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-03-import-manifest | pass | 1.000 | 34.6 | 38100 | 5 | verify: PASS: svc:/site/akh-echo:default online and /var/tmp/akh-e2e-echo.log actively growing | manpages: 0/3 grounded, missing [svccfg, service_bundle, smf_method] (commands confirmed via ExecEnvelope on the remote backend) |
