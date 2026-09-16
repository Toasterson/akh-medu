import unittest
from vesta_ops import inspect_args, bounded
class AccessTests(unittest.TestCase):
    def test_cluster_mutation_and_secret_reads_are_not_exposed(self):
        for args in ({'action':'delete','resource':'pods'},{'action':'get','resource':'secrets'},{'action':'get','resource':'pods','namespace':'--kubeconfig=/private/key'},{'action':'get','resource':'pods','argv':['apply']}):
            with self.assertRaises(ValueError):inspect_args(args)
    def test_pod_logs_are_bounded(self):
        argv=inspect_args({'action':'logs','resource':'pods','name':'api-0','namespace':'apps'})
        self.assertIn('--tail=200',argv);self.assertIn('--limit-bytes=65536',argv)
    def test_output_is_drained_and_bounded(self):
        result=bounded(['/usr/bin/python3','-c','print("x"*100000)'],5)
        self.assertTrue(result['truncated']);self.assertEqual(len(result['stdout']),65536)
if __name__=='__main__':unittest.main()
