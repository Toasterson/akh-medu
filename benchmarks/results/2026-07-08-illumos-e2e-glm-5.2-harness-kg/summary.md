# illumos-e2e × glm-5.2-harness-kg — 2026-07-08 18:08 UTC

- **Harness git SHA**: `80be9c74af72dcc795b29d17fa7f2a28d3bc30dd`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`glm-5.2`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**11/15 tasks passed (73.3%), mean score 0.733, 846.8s total wall time**, 1364982 tokens, 125 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | pass | 1.000 | 76.1 | 119438 | 10 | verify: PASS: cpucount matches live cpu_info kstat instance count | manpages: 1/3 grounded, missing [kstat_open, libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | pass | 1.000 | 31.9 | 114440 | 6 | verify: PASS: readgreeting matches live config/greeting value | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 62.2 | 171005 | 7 | verify: FAIL: svcstate printed 'online', expected 'online*' | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| discover-01-cpu-info | pass | 1.000 | 26.2 | 21456 | 4 | verify: PASS: cpu report matches psrinfo/kstat ground truth | manpages: 0/2 grounded, missing [psrinfo, kstat] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-02-nic-speed | pass | 1.000 | 14.7 | 21921 | 4 | verify: PASS: nic report for 'vioif0' matches dladm ground truth | manpages: 0/1 grounded, missing [dladm] (commands confirmed via ExecEnvelope on the remote backend) |
| discover-03-zpool-create | pass | 1.000 | 15.9 | 25994 | 5 | verify: PASS: akhtank/akhtank/data matches required properties | manpages: 0/2 grounded, missing [zpool, zfs] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-01-create-role | pass | 1.000 | 29.7 | 41916 | 5 | verify: PASS: netops role exists with Network Management profile and is assigned to opuser | manpages: 2/4 grounded, missing [profiles, roles] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-02-custom-profile | pass | 1.000 | 72.2 | 135745 | 11 | verify: PASS: auditor can read /var/adm/akh-e2e-secure.log only via the granted profile, not directly | manpages: 3/4 grounded, missing [usermod] (commands confirmed via ExecEnvelope on the remote backend) |
| rbac-03-restrict-privileges | fail | 0.000 | 42.9 | 96367 | 9 | verify: FAIL: batchrunner account appears broken (cannot even echo) | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 45.9 | 79309 | 11 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 146.1 | 146423 | 12 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | pass | 1.000 | 102.5 | 130709 | 12 | verify: PASS: xattr round trip matches expected content | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-01-clear-maintenance | pass | 1.000 | 106.8 | 124464 | 12 | verify: PASS: svc:/site/akh-relay:default is online | manpages: 0/2 grounded, missing [svcs, svcadm] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-02-enable-configure | pass | 1.000 | 26.0 | 42150 | 7 | verify: PASS: svc:/site/akh-greeter:default online with app/greeting=buenas-akh | manpages: 0/3 grounded, missing [svccfg, svcadm, svcprop] (commands confirmed via ExecEnvelope on the remote backend) |
| smf-03-import-manifest | pass | 1.000 | 47.5 | 93645 | 10 | verify: PASS: svc:/site/akh-echo:default online and /var/tmp/akh-e2e-echo.log actively growing | manpages: 2/3 grounded, missing [service_bundle] (commands confirmed via ExecEnvelope on the remote backend) |
