#!/bin/sh
# relay-group-3svc/setup.sh -- runs as root on exec1 before the task.
#
# Idempotent (safe to re-run against an already-partially-fixed host — the
# orchestrated arm of the W6 harness re-runs this to reset state between
# arms). Installs THREE independent, differently-broken SMF services with
# distinct FMRIs, distinct /opt/akh-e2e-relay-{a,b,c}/ method dirs, and
# distinct /var/svc/manifest/site/akh-relay-{a,b,c}.xml manifests, so N
# parallel workers each fixing one service never touch shared state:
#
#   site/relay-a -- start method execs a script that does not exist yet;
#                   svc.startd retries the start method and the instance
#                   lands in maintenance. Fix: create a working start
#                   script (or repoint the manifest's start method) so the
#                   service starts cleanly.
#   site/relay-b -- manifest imported but left disabled. Fix: svcadm
#                   enable.
#   site/relay-c -- enabled and online, but its app/greeting configuration
#                   property holds a placeholder value instead of the
#                   required "relay-c-ready". Fix: svccfg setprop + svcadm
#                   refresh (no restart needed -- the running instance
#                   picks up the new value on refresh).
set -e

MANIFEST_DIR=/var/svc/manifest/site
mkdir -p "$MANIFEST_DIR"

# ---------------------------------------------------------------------
# relay-a: broken start method -> maintenance
# ---------------------------------------------------------------------
METHOD_A=/opt/akh-e2e-relay-a
MANIFEST_A="$MANIFEST_DIR/akh-relay-a.xml"
FMRI_A="svc:/site/relay-a:default"

mkdir -p "$METHOD_A"
# Reset: nuke any fix a prior run of this task left behind (e.g. a
# real-start.sh the akh created to satisfy the broken exec pointer).
rm -f "$METHOD_A/real-start.sh"

cat > "$METHOD_A/start.sh" <<'INNER'
#!/bin/sh
# Broken on purpose: the real start script this execs does not exist.
# Fixing that (or pointing the manifest at a working start method) is the
# task; this file is safe to edit/replace as part of the fix.
exec /opt/akh-e2e-relay-a/real-start.sh
INNER
chmod 755 "$METHOD_A/start.sh"
# Deliberately NOT creating real-start.sh here.

cat > "$MANIFEST_A" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-relay-a">
  <service name="site/relay-a" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <dependency name="fs-local" grouping="require_all" restart_on="none" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e-relay-a/start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e orchestration relay-a</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST_A"
# Force a fresh start attempt so a prior fix (real-start.sh now removed,
# manifest exec pointer now reset) is re-evaluated and fails again.
svcadm disable -s "$FMRI_A" 2>/dev/null || true
svcadm enable "$FMRI_A" 2>/dev/null || true

# Give svc.startd a short window to escalate the repeated start-method
# failure to maintenance on its own -- then force it. Confirmed live on
# exec1 (2026-07-07): under startd/duration=child a start method that exits
# non-zero almost instantly (our `exec` of a missing script) can cycle
# through thousands of restart/contract attempts (restarter/contract count
# in the thousands observed) without svc.startd's fault-management ever
# escalating to maintenance within any practical setup budget -- the same
# behavior reproduces on the illumos-e2e smf-01 task's proven-identical
# pattern, so this is a guest/restarter characteristic, not a bug in this
# manifest. The underlying broken exec is still genuine (svcs -xv and the
# service log both show the real failure) and still what the akh must fix;
# `svcadm mark maintenance` here only guarantees the *starting* state is
# deterministic when natural escalation doesn't land it in time.
i=0
while [ "$i" -lt 8 ]; do
    state_a=$(svcs -H -o state "$FMRI_A" 2>/dev/null || echo unknown)
    if [ "$state_a" = "maintenance" ]; then
        break
    fi
    sleep 2
    i=$((i + 1))
done
if [ "$(svcs -H -o state "$FMRI_A" 2>/dev/null || echo unknown)" != "maintenance" ]; then
    svcadm mark maintenance "$FMRI_A" 2>/dev/null || true
    sleep 1
fi

# ---------------------------------------------------------------------
# relay-b: imported but disabled -> fix is enable
# ---------------------------------------------------------------------
METHOD_B=/opt/akh-e2e-relay-b
MANIFEST_B="$MANIFEST_DIR/akh-relay-b.xml"
FMRI_B="svc:/site/relay-b:default"

mkdir -p "$METHOD_B"

cat > "$METHOD_B/start.sh" <<'INNER'
#!/bin/sh
exec /usr/bin/sleep 100000000
INNER
chmod 755 "$METHOD_B/start.sh"

cat > "$MANIFEST_B" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-relay-b">
  <service name="site/relay-b" type="service" version="1">
    <create_default_instance enabled="false"/>
    <single_instance/>
    <dependency name="fs-local" grouping="require_all" restart_on="none" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e-relay-b/start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e orchestration relay-b</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST_B"
# Reset: SMF manifest re-import preserves site customizations (does not
# force-disable an instance the akh already enabled), so reset explicitly.
svcadm disable -s "$FMRI_B" 2>/dev/null || true

# ---------------------------------------------------------------------
# relay-c: enabled+online, wrong config property -> fix is setprop+refresh
# ---------------------------------------------------------------------
METHOD_C=/opt/akh-e2e-relay-c
MANIFEST_C="$MANIFEST_DIR/akh-relay-c.xml"
FMRI_C="svc:/site/relay-c:default"

mkdir -p "$METHOD_C"

cat > "$METHOD_C/start.sh" <<'INNER'
#!/bin/sh
exec /usr/bin/sleep 100000000
INNER
chmod 755 "$METHOD_C/start.sh"

cat > "$MANIFEST_C" <<'INNER'
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-relay-c">
  <service name="site/relay-c" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <dependency name="fs-local" grouping="require_all" restart_on="none" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <exec_method type="method" name="start"
        exec="/opt/akh-e2e-relay-c/start.sh" timeout_seconds="60"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
    </property_group>
    <property_group name="app" type="application">
      <propval name="greeting" type="astring" value="placeholder"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">akh e2e orchestration relay-c</loctext></common_name>
    </template>
  </service>
</service_bundle>
INNER

svccfg import "$MANIFEST_C"
# Reset: force the property back to the wrong placeholder value regardless
# of whether a prior run's fix already changed it (manifest re-import alone
# does not roll back a customized property value).
svccfg -s "$FMRI_C" setprop app/greeting = astring: "placeholder" 2>/dev/null || true
svcadm refresh "$FMRI_C" 2>/dev/null || true
svcadm enable "$FMRI_C" 2>/dev/null || true

# ---------------------------------------------------------------------
# Give svc.startd time to settle relay-c into online (relay-a's terminal
# state is already guaranteed by the block above).
# ---------------------------------------------------------------------
i=0
while [ "$i" -lt 15 ]; do
    state_c=$(svcs -H -o state "$FMRI_C" 2>/dev/null || echo unknown)
    if [ "$state_c" = "online" ]; then
        break
    fi
    sleep 2
    i=$((i + 1))
done

echo "setup: $FMRI_A is now state '$(svcs -H -o state "$FMRI_A" 2>/dev/null || echo unknown)' (want maintenance)"
echo "setup: $FMRI_B is now state '$(svcs -H -o state "$FMRI_B" 2>/dev/null || echo unknown)' (want disabled)"
echo "setup: $FMRI_C is now state '$(svcs -H -o state "$FMRI_C" 2>/dev/null || echo unknown)' (want online, app/greeting=placeholder)"
