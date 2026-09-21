import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('observer',Path(__file__).with_name('observer.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DurableEvents(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'state').mkdir();(self.root/'secrets').mkdir()
        (self.root/'secrets/observer.key').write_text('fixture-token')
        (self.root/'secrets/operator.key').write_text('fixture-token')
        self.o=m.Observer(self.root)
        with self.o.db:self.o.put('role_activity',{'ok':True,'roles':{}})
    def tearDown(self):self.o.db.close();self.tmp.cleanup()
    def test_schedule_restart_keeps_exact_occurrence(self):
        with patch.object(m.time,'time',return_value=1800000000):self.o.schedules()
        before=self.o.db.execute('select id,event from outbox order by id').fetchall()
        self.o.db.close();self.o=m.Observer(self.root)
        with patch.object(m.time,'time',return_value=1800000001):self.o.schedules()
        self.assertEqual(before,self.o.db.execute('select id,event from outbox order by id').fetchall())
        self.assertEqual(len(before),3)
    def test_response_loss_replays_identical_event_before_marking_accepted(self):
        with self.o.db:self.o.event('roman-clock','schedule.janus',{'check':1},'one')
        requests=[]
        def send(url,headers,data):
            requests.append(data)
            if len(requests)==1:raise TimeoutError()
            return {'deliveries':['delivery-one']}
        with patch.object(m,'fetch',side_effect=send):
            with self.assertRaises(TimeoutError):self.o.flush()
            self.assertEqual(self.o.db.execute('select accepted from outbox').fetchone()[0],0)
            self.o.flush()
        self.assertEqual(requests[0],requests[1])
        self.assertEqual(self.o.db.execute('select state from deliveries').fetchone()[0],'pending')
    def test_initial_baseline_does_not_flood_or_infer_deletions(self):
        item={'id':'wi-one','current_phase':'backlog'}
        with self.o.db:self.o.observe_entities('work_item','project',[item])
        self.assertEqual(self.o.db.execute('select count(*) from pending').fetchone()[0],0)
        with self.o.db:self.o.observe_entities('work_item','project',[])
        self.assertEqual(self.o.get('work_item:wi-one'),item)
        with self.o.db:self.o.observe_entities('work_item','project',[dict(item,current_phase='review')])
        self.assertEqual(self.o.db.execute('select count(*) from pending').fetchone()[0],2)
    def test_watchdog_still_runs_when_event_ingress_is_down(self):
        with patch.object(self.o,'refresh_activity'), patch.object(self.o,'monitoring'), patch.object(self.o,'scan_anima'), patch.object(self.o,'schedules'), patch.object(self.o,'flush',side_effect=ConnectionError()), patch.object(self.o,'watch_deliveries'), patch.object(self.o,'notify_watchdog') as notify:
            self.o.tick()
        notify.assert_called_once()
        self.assertFalse(self.o.get('ingress_status')['ok'])
        self.assertTrue(self.o.get('board_watchdog_status')['ok'])

    def test_running_role_does_not_accumulate_schedule_wakes(self):
        with self.o.db:self.o.put('role_activity',{'ok':True,'roles':{r:{'busy':True} for r in m.ROLES}})
        with patch.object(m.time,'time',return_value=1800000000):self.o.schedules()
        self.assertEqual(self.o.db.execute('select count(*) from outbox').fetchone()[0],0)

    def test_pending_work_and_rest_period_suppress_new_wakes(self):
        now=1800000000
        with self.o.db:self.o.put('role_activity',{'ok':True,'roles':{
            'janus':{'pending':True},'minerva':{'last_finished':now-3599},'vesta':{'last_finished':now-3600}}})
        with patch.object(m.time,'time',return_value=now):self.o.schedules()
        events=[json.loads(row[0]) for row in self.o.db.execute('select event from outbox')]
        self.assertEqual([e['type'] for e in events],['schedule.vesta'])
        with self.o.db:self.o.put('role_activity',{'ok':False})
        with patch.object(m.time,'time',return_value=now+7200):self.o.schedules()
        self.assertEqual(self.o.db.execute('select count(*) from outbox').fetchone()[0],1)

    def test_archiving_completed_sessions_does_not_restart_rest(self):
        session={'id':'session-one','akh':'vesta','closed_at':None,'last_progress_at':100000,'turns':[{'id':'turn-one','status':'completed','created_at':1}]}
        with patch.object(m,'fetch',side_effect=[[],[session],session]):self.o.refresh_activity()
        session.update(closed_at=200000,last_progress_at=200000)
        with patch.object(m,'fetch',side_effect=[[],[session]]):self.o.refresh_activity()
        self.assertEqual(self.o.get('role_activity')['roles']['vesta']['last_finished'],100)

    def test_volatile_work_item_fields_do_not_wake_roles(self):
        with self.o.db:
            self.o.observe_entities('work_item','p',[{'id':'one','current_phase':'working','updated_at':'old','elapsed':1}])
            self.o.observe_entities('work_item','p',[{'id':'one','current_phase':'working','updated_at':'new','elapsed':90}])
        self.assertEqual(self.o.db.execute('select count(*) from pending').fetchone()[0],0)

    def test_alert_annotation_changes_are_not_incident_transitions(self):
        a={'grafana':{'available':True,'alerts':[{'fingerprint':'one','status':{'state':'active'},'annotations':{'description':'85%'}}]}}
        b=json.loads(json.dumps(a));b['grafana']['alerts'][0]['annotations']['description']='86%'
        self.assertEqual(m.monitoring_signature(a),m.monitoring_signature(b))
        b['grafana']['alerts'][0]['status']['state']='suppressed'
        self.assertNotEqual(m.monitoring_signature(a),m.monitoring_signature(b))

    def test_unrouted_event_is_retained_for_operator_repair(self):
        with self.o.db:self.o.event('roman-clock','schedule.janus',{})
        with patch.object(m,'fetch',return_value={'deliveries':[]}):
            with self.assertRaises(RuntimeError):self.o.flush()
        self.assertEqual(self.o.db.execute('select accepted from outbox').fetchone()[0],0)

    def outcome(self,order=1000,role='janus',status='failed',error='model returned HTTP 429',finished=100):
        return {'akh':role,'turn_id':'turn-'+str(order),'created_at':order,'finished_at':finished,
                'status':status,'error':{'code':'model_incomplete','message':error}}

    def test_provider_cooldown_survives_restart_and_allows_fresh_wake(self):
        with self.o.db:self.o.record_provider_outcome(self.outcome())
        with patch.object(m.time,'time',return_value=101):self.assertFalse(self.o.role_ready('janus'))
        self.o.db.close();self.o=m.Observer(self.root)
        with patch.object(m.time,'time',return_value=1899):self.assertFalse(self.o.role_ready('janus'))
        with patch.object(m.time,'time',return_value=1900):self.assertTrue(self.o.role_ready('janus'))
        self.assertEqual(self.o.db.execute('select count(*) from outbox').fetchone()[0],0)

    def test_repeated_failures_back_off_but_replayed_receipts_do_not(self):
        with self.o.db:
            self.o.record_provider_outcome(self.outcome())
            self.o.record_provider_outcome(self.outcome(finished=99999))
        self.assertEqual(self.o.get('provider_availability')['janus']['retry_at'],1900)
        with self.o.db:self.o.record_provider_outcome(self.outcome(order=2000,finished=2000))
        self.assertEqual(self.o.get('provider_availability')['janus']['retry_at'],5600)

    def test_success_clears_only_own_block_and_stale_failure_cannot_restore_it(self):
        with self.o.db:
            self.o.record_provider_outcome(self.outcome())
            self.o.record_provider_outcome(self.outcome(role='minerva'))
            self.o.record_provider_outcome(self.outcome(order=2000,status='completed'))
            self.o.record_provider_outcome(self.outcome(order=1500,finished=999999))
        with patch.object(m.time,'time',return_value=101):
            self.assertFalse(self.o.provider_blocked('janus'));self.assertTrue(self.o.provider_blocked('minerva'))

    def test_pending_event_retained_until_cooldown_without_changing_identity(self):
        with self.o.db:
            self.o.record_provider_outcome(self.outcome())
            self.o.event('roman-clock','schedule.janus',{'check':1},'retained')
        before=self.o.db.execute('select id,event from outbox').fetchone()
        with patch.object(m.time,'time',return_value=101),patch.object(m,'fetch') as send:
            self.o.flush();send.assert_not_called()
        with patch.object(m.time,'time',return_value=1900),patch.object(m,'fetch',return_value={'deliveries':['one']}) as send:
            self.o.flush();self.assertEqual(send.call_args.args[2],json.loads(before[1]))

    def test_auth_failure_requires_repair_and_never_expires_on_a_timer(self):
        with self.o.db:self.o.record_provider_outcome(self.outcome(error='authentication_error'))
        with patch.object(m.time,'time',return_value=999999999):self.assertTrue(self.o.provider_blocked('janus'))
        with self.o.db:self.o.record_provider_outcome(self.outcome(order=2000,status='completed'))
        self.assertFalse(self.o.provider_blocked('janus'))

    def test_task_failure_is_not_provider_quota(self):
        with self.o.db:self.o.record_provider_outcome(self.outcome(error='test suite failed'))
        self.assertFalse(self.o.provider_blocked('janus'))

    def test_manual_success_can_clear_auth_block_without_an_observer_delivery(self):
        with self.o.db:self.o.record_provider_outcome(self.outcome(error='authentication_error'))
        session={'id':'manual','akh':'janus','closed_at':None,'last_progress_at':200000,
                 'turns':[{'id':'success','status':'completed','created_at':2000}]}
        with patch.object(m,'fetch',side_effect=[[],[session],session]):self.o.refresh_activity()
        self.assertFalse(self.o.provider_blocked('janus'))

    def test_delivery_failure_defers_the_next_schedule_in_same_tick(self):
        now=1800000000
        with self.o.db:self.o.db.execute('insert into deliveries values (?,?,?)',('delivery','pending',None))
        delivery={'admission':{'session_id':'s','turn_id':'t'}}
        session={'id':'s','akh':'janus','last_progress_at':now*1000,
                 'turns':[{'id':'t','status':'failed','created_at':now*1000-1,'error':{'message':'HTTP 429'}}]}
        with patch.object(m.time,'time',return_value=now),patch.object(m,'fetch',side_effect=[delivery,session]):
            self.o.watch_deliveries();self.o.schedules()
        events=[json.loads(row[0]) for row in self.o.db.execute('select event from outbox')]
        self.assertNotIn('schedule.janus',[e['type'] for e in events])

    def test_blocked_backlog_does_not_starve_another_role(self):
        with self.o.db:
            self.o.record_provider_outcome(self.outcome())
            for i in range(32):self.o.event('roman-clock','schedule.janus',{},'blocked-'+str(i))
            self.o.event('roman-clock','schedule.minerva',{},'eligible')
        with patch.object(m.time,'time',return_value=101),patch.object(m,'fetch',return_value={'deliveries':['one']}) as send:
            self.o.flush();send.assert_called_once()
        self.assertEqual(self.o.db.execute('select id from outbox where accepted=1').fetchall(),[('eligible',)])

    def paged_inventory(self, count=205):
        calls=[]
        def mcp(name,args):
            calls.append((name,args))
            if name=='list_projects':return {'projects':[{'id':'p','name':'Project','status':'active'}]}
            if name=='list_workflows':return {'workflows':[]}
            if name=='list_plans':return {'plans':[]}
            offset=int(args.get('page_token',0))
            end=min(offset+100,count)
            return {'work_items':[{'id':str(i),'project_id':'p','current_phase':'draft'} for i in range(offset,end)],
                    'total':count,'next_page_token':str(end) if end<count else ''}
        return calls,mcp

    def test_complete_cursor_scan_survives_restart_without_backfill_flood(self):
        calls,mcp=self.paged_inventory()
        with self.o.db:self.o.put('expanded:p',True)
        with patch.object(self.o,'mcp',side_effect=mcp):self.o.scan_anima()
        coverage=self.o.get('coverage')['p']
        self.assertEqual((coverage['observed'],coverage['total'],coverage['pages']),(205,205,3))
        self.assertTrue(coverage['enumeration_complete'])
        self.assertFalse(coverage['truncated'])
        self.assertEqual([a.get('page_token') for n,a in calls if n=='list_work_items'],[None,'100','200'])
        self.assertEqual(self.o.db.execute('select count(*) from pending').fetchone()[0],0)
        self.o.db.close();self.o=m.Observer(self.root)
        _,mcp=self.paged_inventory(206)
        with patch.object(self.o,'mcp',side_effect=mcp):self.o.scan_anima()
        self.assertEqual(self.o.get('coverage')['p']['observed'],206)
        events=[json.loads(row[0]) for row in self.o.db.execute('select event from outbox')]
        self.assertEqual(len(events),2)
        self.assertTrue(all([c['id'] for c in event['data']['changes']]==['205'] for event in events))

    def test_repeated_or_missing_cursor_is_not_claimed_complete(self):
        for broken in ('repeat','missing'):
            _,normal=self.paged_inventory()
            def mcp(name,args):
                page=normal(name,args)
                if name=='list_work_items' and args.get('page_token'):
                    if broken=='repeat':page['next_page_token']=args['page_token']
                    else:page.pop('next_page_token')
                return page
            with patch.object(self.o,'mcp',side_effect=mcp):
                with self.assertRaises(RuntimeError):self.o.scan_anima()
            self.assertIsNone(self.o.get('coverage'))

    def test_legacy_cap_stays_visibly_truncated(self):
        _,normal=self.paged_inventory()
        def mcp(name,args):
            page=normal(name,args)
            if name=='list_work_items':page.pop('next_page_token')
            return page
        with patch.object(self.o,'mcp',side_effect=mcp):self.o.scan_anima()
        coverage=self.o.get('coverage')['p']
        self.assertTrue(coverage['truncated'])
        self.assertFalse(coverage['cursor_supported'])
        self.assertFalse(coverage['enumeration_complete'])

if __name__=='__main__':unittest.main()
