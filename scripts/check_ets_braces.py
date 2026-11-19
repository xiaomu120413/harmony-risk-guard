"""Crude brace-balance check for ArkTS sources before building."""
import re
import sys
from pathlib import Path

FILES = [
    'entry/src/main/ets/pages/AssessView.ets',
    'entry/src/main/ets/pages/Index.ets',
    'entry/src/main/ets/pages/HistoryView.ets',
    'entry/src/main/ets/pages/AboutView.ets',
    'entry/src/main/ets/pages/Diagnostics.ets',
    'entry/src/main/ets/services/RiskGuardService.ets',
    'entry/src/main/ets/services/HistoryStore.ets',
]

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
ok = True
for f in FILES:
    s = (root / f).read_text(encoding='utf-8')
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    bal = 0
    minbal = 0
    for ch in s:
        if ch == '{':
            bal += 1
        elif ch == '}':
            bal -= 1
            minbal = min(minbal, bal)
    status = 'OK' if bal == 0 and minbal >= 0 else 'BAD'
    if status == 'BAD':
        ok = False
    print(f'{status} {f} balance={bal} min={minbal}')
sys.exit(0 if ok else 1)
