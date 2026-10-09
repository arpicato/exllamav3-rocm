"""Run from repo root: python rocm_tests/check_rocm_load.py -m MODEL [-dm DRAFT]."""
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert torch.version.hip, 'ROCm required'
sys.argv += ['--cache', '8192', '--tokens', '256', '--prompts', 'prose,code', '--show']
run = runpy.run_path('rocm_tests/bench_gen.py')
prompt = 'Summarize these notes.\n' + (
    'Ocean currents move heat around Earth. Wind and salinity affect their direction.\n' * 180
)
count = run['tokenizer'].encode(run['format_prompt'](prompt), encode_special_tokens=True).shape[-1]
assert 2048 <= count < 7000, count
run['run']('long-prefill', prompt)
for _ in range(12):
    result = run['generator'].generate(
        prompt=run['format_prompt']('What is 17 * 23? Reply only with number.'), max_new_tokens=32,
    )
    assert '391' in result, result
torch.cuda.synchronize()
print('PASS: fresh load, decode, long prefill, 12 math checks')
