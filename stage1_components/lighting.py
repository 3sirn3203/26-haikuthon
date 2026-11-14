"""Lighting weights for group-level context selection."""
import math

from .model import require


def lighting_weights(policy, catalog):
    weights = policy.get('lighting_weights', {'soft_diffuse': 1})
    require(isinstance(weights, dict) and bool(weights), 'lighting_weights must be a nonempty mapping')
    require(set(weights) <= set(catalog.active['lighting']), 'lighting_weights contains inactive lighting')
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in weights.values()),
            'lighting_weights must be finite nonnegative numbers')
    require(math.isfinite(sum(weights.values())) and sum(weights.values()) > 0,
            'lighting_weights must have a positive finite sum')
    return {name: value for name, value in weights.items() if value > 0}
