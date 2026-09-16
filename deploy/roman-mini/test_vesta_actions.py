import unittest
from vesta_actions import branch_name, write_args

class ActionTests(unittest.TestCase):
    def test_push_branch_cannot_select_options_or_refspecs(self):
        for value in ('--force','main:other','../main','a//b','a/.hidden','a.lock','main\n'):
            with self.assertRaises(ValueError):branch_name(value)
        self.assertEqual(branch_name('vesta/repair-alerts'),'vesta/repair-alerts')
    def test_mutations_are_fixed_argv_and_named(self):
        cmd,body=write_args({'action':'patch','resource':'deployments','name':'api','namespace':'apps','patch':{'spec':{'replicas':2}},'dry_run':True},'homelab')
        self.assertIn('--dry-run=server',cmd);self.assertEqual(cmd[1:3],['--context','homelab']);self.assertIsNone(body)
        for args in ({'action':'exec','resource':'pods','name':'api'},{'action':'delete','resource':'pods','name':'--all'},{'action':'delete','resource':'pods','name':'api','context':'other'},{'action':'scale','resource':'deployments','name':'api','replicas':-1}):
            with self.assertRaises(ValueError):write_args(args,'homelab')
    def test_manifest_passed_as_stdin_not_host_file_or_shell(self):
        cmd,body=write_args({'action':'apply','manifest':{'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'vesta-test'},'data':{'text':'$(id)'}}},'homelab')
        self.assertEqual(cmd[-4:],['-f','-','-o','name']);self.assertIn(b'$(id)',body)

if __name__=='__main__':unittest.main()
