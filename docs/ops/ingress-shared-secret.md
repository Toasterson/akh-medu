# Ingress shared-secret authentication

WI-340 makes both network services fail closed behind one credential:

- Seshat requires `Authorization: Bearer <token>` on every HTTP route. A
  missing header returns 401; malformed or incorrect credentials return 403.
- akhomed requires the same bearer value in the ACP `initialize` request at
  `_meta.authorization`. Missing and invalid credentials produce ACP JSON-RPC
  errors carrying status 401 and 403 respectively. ACP is raw JSON-RPC/TCP, so
  it cannot return an HTTP status line.

The chart mounts the same Kubernetes Secret into both pods. The configured key
is a text file containing one accepted raw token per non-empty line. The
processes reload that file for every HTTP request or ACP initialization, so a
projected-Secret update takes effect without a restart. Do not put the `Bearer `
prefix in this file.

## Provisioning

Production values should reference a Secret managed by the cluster's secret
controller:

```yaml
ingressAuth:
  existingSecret: akh-medu-ingress-auth
  secretKey: tokens
```

The Secret data is equivalent to:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: akh-medu-ingress-auth
type: Opaque
stringData:
  tokens: |-
    a-long-random-current-token
```

For development only, set `existingSecret: ""` and
`ingressAuth.sharedSecret`; the chart will create a release-owned Secret.
Never commit a real token to a values file.

Seshat callers send the standard HTTP header. Anima's Seshat upstream secret
injection must therefore make the raw token available to the client code that
constructs that header. Akh clients send the same value in ACP initialize
metadata. This repository's `akh --remote` client reads
`AKHOMED_SHARED_SECRET` or the first line of `AKHOMED_SHARED_SECRET_FILE` and
constructs `_meta.authorization` automatically. The Anima-side WI-72 split
must implement the identical wire shape:

```json
{
  "method": "initialize",
  "params": {
    "_meta": { "authorization": "Bearer a-long-random-current-token" }
  }
}
```

The default liveness and readiness checks remain TCP probes. They prove that a
listener is accepting connections but cannot invoke an HTTP or ACP operation,
so there is no unauthenticated public health route to exploit.

## Zero-downtime rotation

Use an overlap window. Kubernetes Secret projection is eventually consistent,
so do not switch clients immediately after the first Secret update.

1. Generate the new token. Update the `tokens` value to contain the new token
   first and the old token second, on separate lines.
2. Wait until the projected file has updated in every akhomed and seshd pod.
   Verify both tokens succeed, and verify a request with no token and a request
   with a random token fail.
3. Update the Anima-side and all other client secret injections to the new
   token. Existing long-lived ACP connections remain valid; every new ACP
   connection must initialize with one of the currently accepted tokens.
4. After all callers use the new token, remove the old line from `tokens`.
   Wait for projection again, then verify the old token fails and the new one
   succeeds.

At no point is the accepted-token set empty, and neither service needs a pod
restart. If the mounted file becomes unreadable or empty, requests fail closed
with 503 and akhomed refuses to open its TCP listener at startup.
