"""Same V12 observation/action conversion, with Dispatch choice-count diagnostics."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from v12_clean_collector import RawRoutingEventCollector


class V14EventCollector(RawRoutingEventCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choices = defaultdict(Counter)

    def _prepare(self, observation, role):
        out=super()._prepare(observation,role)
        if role in ('dispatch_front','dispatch_rear'):
            legal=int((np.asarray(out['action_mask'])>0.5).sum())
            self._choices[role][legal]+=1
        return out

    def episode(self, *args, **kwargs):
        self._choices=defaultdict(Counter)
        trajectories, meta=super().episode(*args, **kwargs)
        diag={}
        for role in ('dispatch_front','dispatch_rear'):
            counts=self._choices[role]
            total=sum(counts.values())
            multi=sum(n for choices,n in counts.items() if choices>=2)
            diag[role]=dict(decisions=total,single_choice=int(counts.get(1,0)),
                            multi_choice=int(multi),multi_choice_rate=multi/total if total else None,
                            legal_actions_histogram={int(k):int(v) for k,v in counts.items()})
        meta['dispatch_choice_diagnostics']=diag
        return trajectories, meta
