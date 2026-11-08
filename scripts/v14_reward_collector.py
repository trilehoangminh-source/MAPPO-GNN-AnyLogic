"""V14 collector with an episode-scoped environment reward shaper."""
from v14_deep_mlp_collector import V14EventCollector


class V14RewardEventCollector(V14EventCollector):
    def __init__(self, model, environment, shaper, *, stop_mask_mode='baseline'):
        if stop_mask_mode != 'baseline':
            raise ValueError('Reward experiment keeps Stop unlocked (baseline mask)')
        super().__init__(model, environment, stop_mask_mode=stop_mask_mode)
        self.shaper = shaper

    def episode(self, *args, **kwargs):
        # IMPORTANT: reset the physical-AGV history before env.reset().
        self.shaper.reset_episode()
        segments, meta = super().episode(*args, **kwargs)
        meta['reward_shaping'] = self.shaper.report()
        return segments, meta
