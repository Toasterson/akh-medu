#!/usr/bin/env python3
"""Version-pinned v2 entry point; reuses the proven RL mechanics runner."""
import runpy, sys
import lilith_rl_tasks_v2

sys.modules['lilith_rl_tasks'] = lilith_rl_tasks_v2
runpy.run_path(str(__import__('pathlib').Path(__file__).with_name('lilith_rl_probe.py')), run_name='__main__')
