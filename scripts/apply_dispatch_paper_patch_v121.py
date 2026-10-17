"""Install the experiment-only hook into existing behaviors_v121.py.

Only one exact source anchor is changed; creates .paper_b2_backup and fails
instead of overwriting an unknown source. Run from project root once.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'thesis'/'envs'/'behaviors_v121.py'
backup=p.with_name('behaviors_v121.py.paper_b2_backup')
pattern='        else:\n            _obs["action_mask"] = np.ones((len(self.hive.stations.keys())))\n        \n        return _obs, _reward, _done, _info\n\nclass SingleDispAgent'
replacement='        else:\n            _obs["action_mask"] = np.ones((len(self.hive.stations.keys())))\n\n        # Optional paper-inspired experiment. Legacy paths are unchanged.\n        # This is read per observation so A/B stays legacy unless explicitly enabled.\n        import os\n        if os.environ.get("MAPPO_DISPATCH_MASK_MODE", "legacy") == "operation_first":\n            from .dispatch_paper_mask_v121 import build_operation_first_dispatch_mask\n            chosen_info = translated[0] if translated else None\n            _obs["action_mask"], _paper_diag = build_operation_first_dispatch_mask(\n                chosen_info, self.part, self.hive.stations.keys()\n            )\n            if not hasattr(self, "_paper_debug_count"):\n                self._paper_debug_count = 0\n            if self._paper_debug_count < 25:\n                print(\n                    "[DISPATCH_B2] "\n                    f"caller={alpyne_obs.caller} "\n                    f"reason={_paper_diag[\'reason\']} "\n                    f"names={_paper_diag[\'station_names\']} "\n                    f"allowed={_paper_diag[\'allowed_actions\']} "\n                    f"buffer_fallback={_paper_diag[\'buffer_fallback\']}",\n                    flush=True,\n                )\n                self._paper_debug_count += 1\n\n        return _obs, _reward, _done, _info\n\nclass SingleDispAgent'

def main():
    if not p.is_file():
        sys.exit(f"ERROR: file missing: {p}")
    if not (ROOT/'thesis'/'envs'/'dispatch_paper_mask_v121.py').is_file():
        sys.exit("ERROR: dispatch_paper_mask_v121.py missing: extract ZIP correctly")
    raw=p.read_bytes()  # preserve exact original bytes in backup
    decoded=raw.decode('utf-8')
    newline='\r\n' if b'\r\n' in raw else '\n'
    text=decoded.replace('\r\n','\n')
    if replacement in text:
        print("[OK] Already installed. No changes.")
        return
    if text.count(pattern)!=1:
        sys.exit("ERROR: Source differs from the confirmed v121 version; no changes made. Share this file for manual diff.")
    if backup.exists():
        sys.exit(f"ERROR: Backup already exists: {backup}; no changes made")
    backup.write_bytes(raw)
    updated=text.replace(pattern,replacement)
    if newline=='\r\n':
        updated=updated.replace('\n','\r\n')
    p.write_bytes(updated.encode('utf-8'))
    print(f"[OK] Applied optional B2 mask hook: {p}")
    print(f"[BACKUP] {backup}")
    print("[DEFAULT] A/B legacy behavior preserved unless MAPPO_DISPATCH_MASK_MODE=operation_first")

if __name__=='__main__': main()
