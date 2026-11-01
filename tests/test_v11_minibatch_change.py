"""Offline regression tests for intentional V11 max-minibatch changes."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_v11_congestion import check_minibatch_resume, record_minibatch_change


class TestMinibatchMigration(unittest.TestCase):
    def setUp(self):
        self.old = dict(version='V11_CONGESTION_EDGEGAT', source_episode=304,
                        train_roles='routing', graph_fingerprint='fakehash', max_minibatch=32)
        self.new = {**self.old, 'max_minibatch': 64}

    def test_change_is_rejected_without_explicit_optin(self):
        with self.assertRaisesRegex(RuntimeError, 'allow-minibatch-change'):
            check_minibatch_resume(self.old, self.new)

    def test_only_minibatch_may_change(self):
        self.assertTrue(check_minibatch_resume(self.old, self.new, True))
        bad = {**self.new, 'source_episode':305}
        with self.assertRaisesRegex(RuntimeError, 'other than minibatch'):
            check_minibatch_resume(self.old, bad, True)

    def test_history_after_completed_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d)/'experiment.json'
            meta.write_text(json.dumps(self.old), encoding='utf-8')
            record_minibatch_change(meta, self.old, 304, 306, 64)
            migrated=json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(migrated['max_minibatch'],64)
            self.assertEqual(migrated['minibatch_schedule'],[
                {'from_episode':305,'max_minibatch':32},
                {'from_episode':307,'max_minibatch':64}])
            self.assertFalse(check_minibatch_resume(migrated,self.new))
            self.assertEqual(json.loads((Path(d)/'experiment_before_minibatch_32_to_64_EP0307.json').read_text())['max_minibatch'],32)

    def test_no_episode_completed_has_no_false_history(self):
        with tempfile.TemporaryDirectory() as d:
            meta=Path(d)/'experiment.json'
            meta.write_text(json.dumps(self.old),encoding='utf-8')
            record_minibatch_change(meta,self.old,304,304,64)
            migrated=json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(migrated['minibatch_schedule'],[{'from_episode':305,'max_minibatch':64}])

if __name__=='__main__': unittest.main()
