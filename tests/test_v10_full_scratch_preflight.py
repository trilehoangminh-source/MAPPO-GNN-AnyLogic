"""Fast preflight tests: do not need AnyLogic or CUDA."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Existing V10-D pack provides v10_pure_common.py.
sys.path.insert(0, str(ROOT / 'scripts'))
from train_V10_full_scratch import main as train_main, metrics_preflight


class ScratchTest(unittest.TestCase):
    def test_dry_run_without_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as p:
            src=Path(p)/'src';src.mkdir()
            cfg=dict(fleetsize=20,max_fleetsize=20,routing_encoder='residual_edge_gat',
                     dispatch_backbone='pooled',dispatch_mask_mode='operation_first',
                     ppo_v7_mode='stable',final_used_guide_strength=1.5,
                     max_events=9000,seed=43,graph_fingerprint='abc')
            (src/'run_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            # There are deliberately no V8 .pt files in this fixture.
            train_main(['--dry-run','--config-run',str(src),'--output',str(Path(p)/'output')])
            self.assertFalse((Path(p)/'output').exists())

    def test_resume_log_check(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'metrics.jsonl'
            f.write_text('\n'.join(json.dumps({'episode':i}) for i in [1,2,3])+'\n')
            metrics_preflight(f,3)
            with self.assertRaises(RuntimeError): metrics_preflight(f,4)

if __name__=='__main__':unittest.main()
