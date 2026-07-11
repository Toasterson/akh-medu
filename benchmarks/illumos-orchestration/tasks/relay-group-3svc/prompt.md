# Bring the relay group online

Three site services on this host — `site/relay-a`, `site/relay-b`, and
`site/relay-c` — are all unhealthy in different ways. Diagnose and fix each
one so that `svcs` reports all three as `online`:

- `svc:/site/relay-a:default`
- `svc:/site/relay-b:default`
- `svc:/site/relay-c:default`

`relay-c` additionally has an `app/greeting` configuration property that
must be set to exactly:

    relay-c-ready

so that the *running* configuration (not just the manifest on disk)
reflects the new value.

The three services are independent of each other — nothing about fixing
one affects the others. You have root access; use whatever illumos tools
you judge appropriate.
