"""Plain EventCollector for V12. NO Dijkstra action prior and NO placeholder bias."""
from thesis.mappo.collector_stop_ab import EventCollector


class RawRoutingEventCollector(EventCollector):
    def _prepare(self, observation, role):
        obs = super()._prepare(observation, role)
        # Hard masks from AnyLogic remain; all path guidance code is bypassed.
        if 'routing_guide_bias' in obs:
            raise RuntimeError('Routing guidance found in V12 input')
        return obs

    def episode(self, *args, **kwargs):
        segments, meta = super().episode(*args, **kwargs)
        meta['routing_guidance'] = {'mode':'removed', 'strength':0.0,
                                    'dijkstra_guidance_calls':0,
                                    'shortest_path_flow_used':False}
        return segments, meta
