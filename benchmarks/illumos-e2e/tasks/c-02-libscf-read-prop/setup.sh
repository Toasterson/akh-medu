#!/bin/sh
# c-02-libscf-read-prop/setup.sh -- runs as root on exec1 before the task.
# Idempotent: installs a trivial custom service with a config/greeting
# property set to a fresh value each run, so the value can only be
# obtained by actually reading it (not by hardcoding a guess).
set -e

METHOD_DIR=/opt/akh-e2e
MANIFEST_DIR=/var/svc/manifest/site
MANIFEST="$MANIFEST_DIR/akh-note.xml"
FMRI="svc:/site/akh-note:default"
VALUE="orange-$$-$(date +%s | tail -c 5)"

mkdir -p "$METHOD_DIR" "$MANIFEST_DIR"

cat > "$METHOD_DIR/akh-note-start.sh" <<'INNER'
#!/bin/sh
exec /usr/bin/sleep 100000000
INNER
chmod 755 "$METHOD_DIR/akh-note-start.sh"

cat > "$MANIFEST" <<INNER
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-note">
  <service name="site/akh-note" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e/akh-note-start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <property_group name="config" type="application">
      <propval name="greeting" type="astring" value="$VALUE"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e note demo service</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST"
svcadm enable "$FMRI" 2>/dev/null || true
echo "setup: $FMRI imported with config/greeting set (value withheld from this log)"
