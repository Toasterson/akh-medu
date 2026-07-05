#!/bin/sh
# smf-01-clear-maintenance/setup.sh -- runs as root on exec1 before the task.
# Idempotent: installs a custom SMF service whose start method execs a
# script that does not exist yet, so svc.startd repeatedly fails the start
# method and the instance lands in the maintenance state.
set -e

METHOD_DIR=/opt/akh-e2e
MANIFEST_DIR=/var/svc/manifest/site
MANIFEST="$MANIFEST_DIR/akh-relay.xml"
FMRI="svc:/site/akh-relay:default"

mkdir -p "$METHOD_DIR"
mkdir -p "$MANIFEST_DIR"

cat > "$METHOD_DIR/akh-relay-start.sh" <<'INNER'
#!/bin/sh
# Broken on purpose: the real start script this execs does not exist.
# Fixing that (or pointing the manifest at a working start method) is the
# task; this file is safe to edit/replace as part of the fix.
exec /opt/akh-e2e/akh-relay-real-start.sh
INNER
chmod 755 "$METHOD_DIR/akh-relay-start.sh"
# Deliberately NOT creating akh-relay-real-start.sh here.

cat > "$MANIFEST" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-relay">
  <service name="site/akh-relay" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <dependency name="fs-local" grouping="require_all" restart_on="none" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e/akh-relay-start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e relay demo service</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST"
svcadm enable "$FMRI" 2>/dev/null || true

# Give svc.startd a few retries to accumulate start-method failures and
# transition the instance into maintenance.
i=0
while [ "$i" -lt 15 ]; do
    state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo unknown)
    if [ "$state" = "maintenance" ]; then
        break
    fi
    sleep 2
    i=$((i + 1))
done

echo "setup: $FMRI is now in state '$(svcs -H -o state "$FMRI" 2>/dev/null || echo unknown)'"
