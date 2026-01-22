"""Quick test for MTF fractal positions."""
from nexus_core.temporal_engine.encoders import get_mtf_positions, get_mtf_feat_names

# Test at different times to show the fractal structure
test_times = [
    (8, 0, '08:00 - H4 start'),
    (9, 0, '09:00 - H4-2'),
    (10, 30, '10:30 - H4-3 (expansion)'),
    (11, 45, '11:45 - H4-4 (distribution)'),
    (14, 15, '14:15 - NY Killzone'),
]

print('=' * 70)
print('MTF FRACTAL POSITION TEST')
print('=' * 70)

for h, m, desc in test_times:
    pos = get_mtf_positions(h, m)
    feat = get_mtf_feat_names(pos)
    print(f'\n{desc}:')
    print(f'  M15: pos={pos["m15_position"]} ({feat.get("m15_feat", "?")})')
    print(f'  H1:  pos={pos["h1_position"]} ({feat.get("h1_feat", "?")})')
    print(f'  H4:  pos={pos["h4_position"]} ({feat.get("h4_feat", "?")})')

print('\n' + '=' * 70)
print('FEAT CYCLE VERIFICATION')
print('=' * 70)
print('H4-1 (00:00-03:59) = ACCUMULATION')
print('H4-2 (04:00-07:59) = ACCUMULATION') 
print('H4-3 (08:00-11:59) = MANIPULATION')
print('H4-4 (12:00-15:59) = EXPANSION <-- NY Killzone')
print('H4-5 (16:00-19:59) = EXPANSION')
print('H4-6 (20:00-23:59) = DISTRIBUTION')
