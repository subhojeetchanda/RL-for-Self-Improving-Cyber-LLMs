"""Single source of truth for stochastic generation settings.

All causal-LM training validation, MAPPO rollouts, and four-quadrant evaluation
must use identical sampling settings so that the SFT and MAPPO arms are
comparable. These constants are imported (not duplicated) by every caller.
"""

TEMPERATURE = 1.0
TOP_P = 1.0
DO_SAMPLE = True

GENERATION_SETTINGS = {
    "do_sample": DO_SAMPLE,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
}
