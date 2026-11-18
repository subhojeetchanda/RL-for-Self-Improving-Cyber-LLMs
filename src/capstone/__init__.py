"""Capstone secret-protection task: MAPPO guard vs stochastic SFT guard.

Modules
-------
task    : canonical observation format, system prompt with a confidential token,
          and the simulated downstream agent used to measure secret leakage.
corpus  : disjoint train / dev / test corpora for the four-quadrant objective.
mappo   : corrected CTDE MAPPO trainer (real GAE, real KL anchor, real co-evolution).
"""

from .task import (  # noqa: F401
    DEFAULT_SECRET,
    PolicyOutcome,
    build_policy_input,
    build_system_prompt,
    scan_for_secret,
    simulate_downstream,
)

__all__ = [
    "DEFAULT_SECRET",
    "PolicyOutcome",
    "build_policy_input",
    "build_system_prompt",
    "scan_for_secret",
    "simulate_downstream",
]
