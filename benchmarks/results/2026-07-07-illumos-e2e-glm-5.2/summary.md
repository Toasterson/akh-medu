# illumos-e2e × glm-5.2 — 2026-07-07 12:08 UTC

- **Harness git SHA**: `9fcd85270f19aeb56618b28d3329f64e9521fe90`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`glm-5.2`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**8/15 tasks passed (53.3%), mean score 0.533, 2950.0s total wall time**, 3524408 tokens, 244 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 349.7 | 158596 | 14 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 326.6 | 600529 | 26 | verify: FAIL: /root/readgreeting is not present/executable -- did it compile? | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 363.0 | 692971 | 24 | verify: FAIL: /root/svcstate.c does not exist | manpages: 2/3 grounded, missing [scf_simple_prop_get] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-01-cpu-info | pass | 1.000 | 15.7 | 21192 | 4 | verify: PASS: cpu report matches psrinfo/kstat ground truth | manpages: 0/2 grounded, missing [psrinfo, kstat] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-02-nic-speed | pass | 1.000 | 24.8 | 21073 | 4 | verify: PASS: nic report for 'vioif0' matches dladm ground truth | manpages: 0/1 grounded, missing [dladm] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-03-zpool-create | pass | 1.000 | 20.5 | 47183 | 6 | verify: PASS: akhtank/akhtank/data matches required properties | manpages: 1/2 grounded, missing [zfs] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-01-create-role | pass | 1.000 | 195.6 | 401510 | 25 | verify: PASS: netops role exists with Network Management profile and is assigned to opuser | manpages: 2/4 grounded, missing [profiles, roles] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-02-custom-profile | pass | 1.000 | 61.2 | 159709 | 14 | verify: PASS: auditor can read /var/adm/akh-e2e-secure.log only via the granted profile, not directly | manpages: 2/4 grounded, missing [pfexec, usermod] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-03-restrict-privileges | fail | 0.000 | 44.3 | 96954 | 9 | verify: FAIL: batchrunner account appears broken (cannot even echo) | manpages: 2/3 grounded, missing [privileges] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 328.0 | 210729 | 24 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 321.3 | 351058 | 31 | verify: FAIL: /root/doorping.rs does not exist | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | fail | 0.000 | 422.4 | 355472 | 29 | verify: FAIL: /root/xattrdemo.rs does not exist | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-01-clear-maintenance | pass | 1.000 | 196.1 | 194548 | 9 | verify: PASS: svc:/site/akh-relay:default is online | manpages: 0/2 grounded, missing [svcs, svcadm] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-02-enable-configure | pass | 1.000 | 96.0 | 39748 | 7 | verify: PASS: svc:/site/akh-greeter:default online with app/greeting=buenas-akh | manpages: 0/3 grounded, missing [svccfg, svcadm, svcprop] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-03-import-manifest | pass | 1.000 | 184.8 | 173136 | 18 | verify: PASS: svc:/site/akh-echo:default online and /var/tmp/akh-e2e-echo.log actively growing | manpages: 1/3 grounded, missing [service_bundle, smf_method] (commands confirmed via ExecEnvelope on the remote backend) |
