#!/bin/sh
# smf-02-enable-configure/setup.sh -- runs as root on exec1 before the task.
# Idempotent: installs a trivial long-running "service" (a sleep loop),
# disabled by default, with an application property group holding a
# placeholder greeting the task must change.
set -e

METHOD_DIR=/opt/akh-e2e
MANIFEST_DIR=/var/svc/manifest/site
MANIFEST="$MANIFEST_DIR/akh-greeter.xml"
FMRI="svc:/site/akh-greeter:default"

mkdir -p "$METHOD_DIR" "$MANIFEST_DIR"

cat > "$METHOD_DIR/akh-greeter-start.sh" <<'INNER'
#!/bin/sh
exec /usr/bin/sleep 100000000
INNER
chmod 755 "$METHOD_DIR/akh-greeter-start.sh"

cat > "$MANIFEST" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-greeter">
  <service name="site/akh-greeter" type="service" version="1">
    <create_default_instance enabled="false"/>
    <single_instance/>
    <dependency name="fs-local" grouping="require_all" restart_on="none" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e/akh-greeter-start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <property_group name="app" type="application">
      <propval name="greeting" type="astring" value="placeholder"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e greeter demo service</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST"
echo "setup: $FMRI imported, state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo unknown)"
