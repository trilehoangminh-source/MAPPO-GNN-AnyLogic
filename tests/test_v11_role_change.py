"""No simulator needed: validate safe Routing -> all MAPPO resume."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_v11_congestion import check_minibatch_resume,check_role_resume,record_role_change

class TestV11RoleChange(unittest.TestCase):
    def setUp(self):
        self.old=dict(version='V11_CONGESTION_EDGEGAT',source_episode=304,
                      train_roles='routing',max_minibatch=64,device='cuda:0',graph_fingerprint='abc')
        self.new={**self.old,'train_roles':'all'}

    def test_reject_change_without_opt_in(self):
        with self.assertRaisesRegex(RuntimeError,'allow-role-change'):
            check_minibatch_resume(self.old,self.new)
        self.assertTrue(check_minibatch_resume(self.old,self.new,allow_role_change=True) is False)
        self.assertTrue(check_role_resume(self.old,self.new,True))

    def test_reject_other_changes_and_reverse(self):
        bad={**self.new,'source_episode':305}
        with self.assertRaisesRegex(RuntimeError,'parameters/source changed'):
            check_minibatch_resume(self.old,bad,allow_role_change=True)
        with self.assertRaisesRegex(RuntimeError,'re-freeze'):
            check_role_resume(self.new,self.old,True)

    def test_history_preserves_prior_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'experiment.json'
            file.write_text(json.dumps(self.old))
            record_role_change(file,self.old,304,325,'all')
            new=json.loads(file.read_text())
            self.assertEqual(new['train_roles'],'all')
            self.assertEqual(new['train_roles_schedule'],[
                {'from_episode':305,'train_roles':'routing'},
                {'from_episode':326,'train_roles':'all'}])
            self.assertTrue((Path(d)/'experiment_before_roles_routing_to_all_EP0326.json').exists())
            self.assertFalse(check_minibatch_resume(new,self.new))
            with self.assertRaises(FileExistsError):
                record_role_change(file,self.old,304,325,'all')

    def test_first_episode_role_change(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'experiment.json';file.write_text(json.dumps(self.old))
            record_role_change(file,self.old,304,304,'all')
            s=json.loads(file.read_text())['train_roles_schedule']
            self.assertEqual(s,[{'from_episode':305,'train_roles':'all'}])

if __name__=='__main__': unittest.main()
