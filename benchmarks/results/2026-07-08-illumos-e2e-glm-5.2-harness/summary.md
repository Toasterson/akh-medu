# illumos-e2e × glm-5.2-harness — 2026-07-08 17:41 UTC

- **Harness git SHA**: `f8068d2eeea3b88e33bcfd81f730eac42f704e5a`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`glm-5.2`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**8/15 tasks passed (53.3%), mean score 0.533, 570.1s total wall time**, 2401075 tokens, 178 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 16.0 | 82919 | 9 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 0/3 grounded, missing [kstat_open, kstat_lookup, libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 24.9 | 277369 | 10 | verify: FAIL: readgreeting printed 'HellofromillumosSMF!', expected 'Hellorom\illumos\SMF!' | manpages: 1/2 grounded, missing [libscf] (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 54.3 | 305966 | 12 | verify: FAIL: svcstate printed 'online', expected 'online*' | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| discover-01-cpu-info | pass | 1.000 | 8.5 | 28450 | 5 | verify: PASS: cpu report matches psrinfo/kstat ground truth | manpages: 0/2 grounded, missing [psrinfo, kstat] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-02-nic-speed | pass | 1.000 | 7.5 | 29955 | 5 | verify: PASS: nic report for 'vioif0' matches dladm ground truth | manpages: 0/1 grounded, missing [dladm] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-03-zpool-create | pass | 1.000 | 11.9 | 72220 | 8 | verify: PASS: akhtank/akhtank/data matches required properties | manpages: 1/2 grounded, missing [zfs] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-01-create-role | pass | 1.000 | 15.0 | 70949 | 8 | verify: PASS: netops role exists with Network Management profile and is assigned to opuser | manpages: 2/4 grounded, missing [profiles, roles] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-02-custom-profile | pass | 1.000 | 22.2 | 155701 | 11 | verify: PASS: auditor can read /var/adm/akh-e2e-secure.log only via the granted profile, not directly | manpages: 2/4 grounded, missing [prof_attr, pfexec] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-03-restrict-privileges | fail | 0.000 | 23.7 | 66886 | 11 | verify: FAIL: batchrunner account appears broken (cannot even echo) | manpages: 0/3 grounded, missing [usermod, ppriv, privileges] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 19.5 | 72815 | 10 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 98.4 | 147217 | 8 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | fail | 0.000 | 33.1 | 197506 | 14 | verify: FAIL: /root/xattrdemo.rs uses Linux setxattr/getxattr or the xattr crate, neither of which exists on illumos | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-01-clear-maintenance | pass | 1.000 | 163.5 | 401352 | 31 | verify: PASS: svc:/site/akh-relay:default is online | manpages: 0/2 grounded, missing [svcs, svcadm] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-02-enable-configure | pass | 1.000 | 14.3 | 58123 | 9 | verify: PASS: svc:/site/akh-greeter:default online with app/greeting=buenas-akh | manpages: 0/3 grounded, missing [svccfg, svcadm, svcprop] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-03-import-manifest | pass | 1.000 | 57.3 | 433647 | 27 | verify: PASS: svc:/site/akh-echo:default online and /var/tmp/akh-e2e-echo.log actively growing | manpages: 2/3 grounded, missing [service_bundle] (commands confirmed via ExecEnvelope on the remote backend) |
