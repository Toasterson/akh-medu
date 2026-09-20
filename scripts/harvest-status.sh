#!/bin/sh
# Fleet status for the transcript harvest (ADR 108). Read-only; needs the
# harvest_read client cert (default: ~/.config/akh-transcripts/certs on
# hephaistos, or set AKH_TRANSCRIPTS_READ_URL).
set -eu
D="${AKH_TRANSCRIPTS_CERTS:-$HOME/.config/akh-transcripts/certs}"
URL="${AKH_TRANSCRIPTS_READ_URL:-postgres://harvest_read@harvest-pg.lab.home:5432/transcripts?sslmode=verify-full&sslrootcert=$D/ca.crt&sslcert=$D/harvest_read.crt&sslkey=$D/harvest_read.key}"
q(){ psql "$URL" -tA -c "$1"; }
echo "== harvesters (per machine) =="
q "select rpad(machine,14)||' rows='||lpad(count(*)::text,7)||' streams='||lpad(count(distinct (source,session_id,stream_id))::text,4)||' last_harvest='||to_char(max(harvested_at) at time zone 'UTC','YYYY-MM-DD HH24:MI')||'Z age='||justify_interval(now()-max(harvested_at)) from th_raw_records group by machine order by 1"
echo "== provisioned roles with no rows yet =="
q "select coalesce(string_agg(rolname,', '),'(none)') from pg_roles r where rolname like 'harvest\_%' and rolname not in ('harvest_derive','harvest_read') and not exists (select 1 from th_raw_records t where t.machine=replace(r.rolname,'harvest_',''))"
echo "== derivation =="
q "select 'raw='||(select count(*) from th_raw_records)||' events='||(select count(*) from th_events)||' chunks='||(select count(*) from th_chunks)||' embedded='||(select count(*) from th_chunks where embedding is not null)||' sessions='||(select count(*) from th_sessions)"
q "select 'pending (anti-join, should be 0 within a worker interval): '||count(*) from th_raw_records r where not exists (select 1 from th_events e where e.derivation_version=1 and e.source=r.source and e.session_id=r.session_id and e.stream_id=r.stream_id and e.seq=r.seq and e.subseq=0)"
q "select 'worker singleton lock (key 0x414B444552495645): '||case when count(*)>0 then 'HELD by pid '||string_agg(pid::text,',') else 'NOT HELD — worker down, or running unlocked (images <= 0b5fe0f963f8 lose it after ~30 min)' end from pg_locks where locktype='advisory' and classid=1095451717 and objid=1380537925 and granted"
echo "== drift (top 5) =="
q "select rpad(source,12)||rpad(record_type,32)||' unknown='||unknown_count||' degraded='||degraded_count from th_drift_counters order by unknown_count+degraded_count desc limit 5"
