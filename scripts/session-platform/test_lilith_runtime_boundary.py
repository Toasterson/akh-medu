import unittest
from lilith_runtime_boundary import RuntimeBoundary, assert_owned_turns_idle

class Process:
    def __init__(self,pid):self.pid=pid;self.returncode=None
    def poll(self):return self.returncode

class BoundaryTest(unittest.TestCase):
    def setUp(self):
        self.states={'parent':{'turns':[{'id':'author','status':'completed'}]},
            'patch':{'turns':[{'id':'p','status':'completed'}]},
            'review':{'turns':[{'id':'r','status':'completed'}]}}
        self.row={'parent':'parent','parent_turn':'author','patch_session':'patch','patch_turn':'p'}
        self.api=lambda path:self.states[path.rsplit('/',1)[1]]

    def test_switch_waits_for_old_exit_and_records_pids(self):
        events=[]
        def stop(p):events.append(('stop',p.pid));p.returncode=0
        def start(target,sequence):events.append(('start',target,sequence));return Process(20)
        manager=RuntimeBoundary(self.api,start,stop,'vulkan');manager.attach(Process(10))
        receipt=manager.switch('hip',self.row)
        self.assertEqual(events,[('stop',10),('start','hip',1)])
        self.assertEqual((receipt['old_pid'],receipt['new_pid']),(10,20))

    def test_nonterminal_or_unknown_turn_refuses_switch_without_stop(self):
        for status in ('queued','dispatching','running','unknown'):
            with self.subTest(status=status):
                self.states['patch']['turns'][0]['status']=status
                stopped=[];manager=RuntimeBoundary(self.api,lambda *_:None,stopped.append,'vulkan')
                manager.attach(Process(10))
                with self.assertRaisesRegex(RuntimeError,'nonterminal'):manager.switch('hip',self.row)
                self.assertFalse(stopped)

    def test_same_backend_still_checks_idle_but_does_not_restart(self):
        manager=RuntimeBoundary(self.api,lambda *_:None,lambda _:self.fail('stop'),'vulkan')
        manager.attach(Process(10));receipt=manager.switch('vulkan',self.row)
        self.assertFalse(receipt['changed'])

if __name__=='__main__':unittest.main()
