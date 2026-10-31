"""All Monte Carlo experiments. Every hazard table here is ASSUMED unless it says `measured`."""
import numpy as np

H_EQUAL = [0.001, 0.06, 0.04, 0.03]                         # per metre; both configurations alike
H_WHEEL_SENSITIVE = dict(point_foot=[0.001, 0.03, 0.02, 0.02], wheel_legged=[0.001, 0.08, 0.10, 0.06])
METHODS = ["M0", "M1", "M2", "M3", "ORACLE"]
EVAL_MAPS = list(range(200))
DEV_MAPS = list(range(100000, 100050))
REPLICATES = 5
