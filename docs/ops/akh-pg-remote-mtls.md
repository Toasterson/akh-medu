# Connecting a remote akh to akh-pg over mTLS

The akh truth store (`akh-pg`, CloudNativePG in the Talos cluster, ns `akh-medu`)
is reachable by akh daemons on other hosts — on the LAN, and over the Internet for
cloud machines (Archibald, an OpenIndiana zone, …). Every connection is
**mutual-TLS**: the client verifies the server cert *and* presents a client
certificate. There is no password path (the Cilium LoadBalancer SNATs source IPs,
so a subnet password rule can't be matched — mTLS is enforced for all).

## Endpoints

| From | Host | Notes |
|---|---|---|
| LAN | `192.168.1.16:5432` | Cilium LoadBalancer `akh-pg-lan` (pinned). |
| Internet | `akh-pg.wegmueller.it:5432` | Needs the network step below. Server cert SAN matches this name for `sslmode=verify-full`. |

## One-time network step (homelab side)

1. **DNS**: point `akh-pg.wegmueller.it` (public) at the homelab's public IP.
   `akh-pg.lab.home` is also a server-cert SAN for internal use (coredns-lab-home).
2. **Port-forward**: forward public TCP `5432` → `192.168.1.16:5432`.
3. (Optional hardening) restrict the forward to known source IPs; mTLS already
   blocks anyone without a client cert, but an allowlist reduces exposure.

## Client-certificate bundle

A client cert (`CN=akhomed`, signed by the cluster CA) authenticates as the
`akhomed` Postgres role. Distribute three files to each remote akh host:

- `ca.crt` — the cluster CA (server verification).
- `akhomed-client.crt` — the client certificate.
- `akhomed-client.key` — the client key (keep `chmod 600`).

Generated under `~/akh-3akh-test/pg-certs/` on hephaistos. To mint another (e.g. a
per-host cert — all use `CN=akhomed`):

```bash
cd ~/akh-3akh-test/pg-certs
openssl genrsa -out HOST-client.key 2048 && chmod 600 HOST-client.key
openssl req -new -key HOST-client.key -subj "/CN=akhomed" -out HOST-client.csr
printf 'extendedKeyUsage=clientAuth\n' > client-ext.cnf
openssl x509 -req -in HOST-client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -days 3650 -extfile client-ext.cnf -out HOST-client.crt
```

`ca.key` is the cluster CA private key — it stays on hephaistos (signing only),
`chmod 600`, never distributed or committed. (CNPG keeps it in the `akh-pg-ca`
secret; the local copy is a convenience for issuing client certs.)

## akh config on the remote host

Install the bundle (e.g. under `/etc/akh/`) and set the daemon's `[postgres].url`
(`~/.config/akh-medu/config.toml`):

```toml
[postgres]
url = "postgres://akhomed@akh-pg.wegmueller.it:5432/akhomed?sslmode=verify-full&sslrootcert=/etc/akh/ca.crt&sslcert=/etc/akh/akhomed-client.crt&sslkey=/etc/akh/akhomed-client.key"
migrate_on_start = true
```

- No password — the client cert *is* the credential.
- `sslmode=verify-full` checks the server cert against `akh-pg.wegmueller.it`
  (use `sslmode=require` on the LAN against `192.168.1.16`, which isn't a SAN).
- akhomed passes these straight to SeaORM/sqlx (verified working).

Verify before starting the daemon:

```bash
psql "host=akh-pg.wegmueller.it port=5432 user=akhomed dbname=akhomed \
  sslmode=verify-full sslrootcert=/etc/akh/ca.crt \
  sslcert=/etc/akh/akhomed-client.crt sslkey=/etc/akh/akhomed-client.key" \
  -tAc "select 'mTLS ok as '||current_user"
```

## Security model (verified)

- `hostnossl akhomed akhomed all reject` — no plaintext akhomed connection ever.
- `hostssl akhomed akhomed all cert clientcert=verify-full` — a valid client cert
  is required for every akhomed connection (LAN + Internet alike).
- Confirmed live: with the cert → `MTLS-OK as akhomed`; without → `FATAL:
  connection requires a valid client certificate`; the hephaistos daemon connects
  and writes over mTLS.
- The akh's *truth* lives on the cluster (amd64 ZFS storage tier), so a remote
  compute host (hephaistos GPU, an OpenIndiana zone) can go offline and be woken
  elsewhere — the relocatable-akh design (ADR 049/072).
