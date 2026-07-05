# Put an existing daemon under SMF supervision

There is a working daemon script at `/opt/akh-e2e/echo-daemon.sh` on this
host that is not yet managed by SMF -- right now nobody restarts it if it
dies, and there's no `svcs` visibility into whether it's running.

Author a proper SMF service manifest for it, name the service instance
exactly:

    svc:/site/akh-echo:default

import it, and enable it so it is supervised and online. You have root
access; use whatever illumos tools and manual pages you judge appropriate.
