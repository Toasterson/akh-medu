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
        self.o=m.Observer(self.root)
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
        with patch.object(self.o,'monitoring'), patch.object(self.o,'scan_anima'), patch.object(self.o,'schedules'), patch.object(self.o,'flush',side_effect=ConnectionError()), patch.object(self.o,'watch_deliveries'), patch.object(self.o,'notify_watchdog') as notify:
            self.o.tick()
        notify.assert_called_once()
        self.assertFalse(self.o.get('ingress_status')['ok'])
        self.assertTrue(self.o.get('board_watchdog_status')['ok'])

    def test_unrouted_event_is_retained_for_operator_repair(self):
        with self.o.db:self.o.event('roman-clock','schedule.janus',{})
        with patch.object(m,'fetch',return_value={'deliveries':[]}):
            with self.assertRaises(RuntimeError):self.o.flush()
        self.assertEqual(self.o.db.execute('select accepted from outbox').fetchone()[0],0)

if __name__=='__main__':unittest.main()
