"""Safely install B2 hook into source that may already have [DISP DEBUG].

This installer is additive, preserves other instrumentation, and backups raw bytes.
Run from the project root (containing scripts/ and thesis/).
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
source = ROOT / 'thesis' / 'envs' / 'behaviors_v121.py'
module = ROOT / 'thesis' / 'envs' / 'dispatch_paper_mask_v121.py'
backup = source.with_name('behaviors_v121.py.paper_b2_backup')

# The only permitted insertion point is inside MultiDispAgent.convert_from_observation,
# after constructing the legacy dispatch mask, before temporary debug instrumentation.
ANCHOR = '        else:\n            _obs["action_mask"] = np.ones((len(self.hive.stations.keys())))\n'
HOOK = '\n        # Optional paper-inspired experiment. Legacy paths are unchanged.\n        # This is read per observation so A/B stays legacy unless explicitly enabled.\n        import os\n        if os.environ.get("MAPPO_DISPATCH_MASK_MODE", "legacy") == "operation_first":\n            from .dispatch_paper_mask_v121 import build_operation_first_dispatch_mask\n            chosen_info = translated[0] if translated else None\n            _obs["action_mask"], _paper_diag = build_operation_first_dispatch_mask(\n                chosen_info, self.part, self.hive.stations.keys()\n            )\n            if not hasattr(self, "_paper_debug_count"):\n                self._paper_debug_count = 0\n            if self._paper_debug_count < 25:\n                print(\n                    "[DISPATCH_B2] "\n                    f"caller={alpyne_obs.caller} "\n                    f"reason={_paper_diag[\'reason\']} "\n                    f"names={_paper_diag[\'station_names\']} "\n                    f"allowed={_paper_diag[\'allowed_actions\']} "\n                    f"buffer_fallback={_paper_diag[\'buffer_fallback\']}",\n                    flush=True,\n                )\n                self._paper_debug_count += 1\n\n'


def main():
    if not source.is_file():
        sys.exit(f'ERROR: Missing source: {source}')
    if not module.is_file():
        sys.exit(f'ERROR: Missing module: {module}; extract the original B2 ZIP first')
    raw = source.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    content = raw.decode('utf-8').replace('\r\n', '\n')
    if 'MAPPO_DISPATCH_MASK_MODE' in content:
        print('[OK] B2 hook appears installed already; no changes.')
        return
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        sys.exit(f'ERROR: Current source does not parse; no changes: {exc}')
    cls = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiDispAgent']
    if len(cls) != 1:
        sys.exit('ERROR: Expected exactly one MultiDispAgent class; no changes')
    methods = [node for node in cls[0].body if isinstance(node, ast.FunctionDef) and node.name == 'convert_from_observation']
    if len(methods) != 1:
        sys.exit('ERROR: Expected exactly one MultiDispAgent.convert_from_observation method; no changes')
    lines = content.splitlines(keepends=True)
    method = methods[0]
    method_text = ''.join(lines[method.lineno-1:method.end_lineno])
    if method_text.count(ANCHOR) != 1 or ('possible_station_nodes' not in method_text or 'translated' not in method_text):
        sys.exit('ERROR: Dispatch implementation differs beyond expected instrumentation; no changes. Share source for manual diff.')
    modified_method = method_text.replace(ANCHOR, ANCHOR + HOOK, 1)
    proposed = ''.join(lines[:method.lineno-1]) + modified_method + ''.join(lines[method.end_lineno:])
    try:
        ast.parse(proposed)
    except SyntaxError as exc:
        sys.exit(f'ERROR: Modified source does not parse; no changes: {exc}')
    if backup.exists():
        sys.exit(f'ERROR: Backup already exists ({backup}); no changes to avoid overwriting it')
    backup.write_bytes(raw)
    source.write_bytes(proposed.replace('\n', newline).encode('utf-8'))
    print('[OK] B2 mask hook installed while preserving existing debug code')
    print(f'[BACKUP] {backup}')
    print('[DEFAULT] A/B unchanged; B2 only with MAPPO_DISPATCH_MASK_MODE=operation_first')

if __name__ == '__main__':
    main()
