#!/bin/sh
# c-03-libscf-smf-state/setup.sh -- runs as root on exec1 before the task.
# Idempotent: installs a custom service whose start method is broken on
# purpose, so it lands in maintenance -- a distinct state string the
# akh's program must report correctly (not just "online").
set -e

METHOD_DIR=/opt/akh-e2e
MANIFEST_DIR=/var/svc/manifest/site
MANIFEST="$MANIFEST_DIR/akh-guard.xml"
FMRI="svc:/site/akh-guard:default"

mkdir -p "$METHOD_DIR" "$MANIFEST_DIR"

cat > "$METHOD_DIR/akh-guard-start.sh" <<'INNER'
#!/bin/sh
exec /opt/akh-e2e/akh-guard-real-start.sh
INNER
chmod 755 "$METHOD_DIR/akh-guard-start.sh"
# Deliberately NOT creating akh-guard-real-start.sh -- this service must
# stay broken for this task (unlike smf-01, it is not meant to be fixed).

cat > "$MANIFEST" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-guard">
  <service name="site/akh-guard" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e/akh-guard-start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e guard demo service</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST"
svcadm enable "$FMRI" 2>/dev/null || true

i=0
while [ "$i" -lt 15 ]; do
    state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo unknown)
    if [ "$state" = "maintenance" ]; then
        break
    fi
    sleep 2
    i=$((i + 1))
done

echo "setup: $FMRI is now in state '$(svcs -H -o state "$FMRI" 2>/dev/null || echo unknown)' (expected: maintenance)"
