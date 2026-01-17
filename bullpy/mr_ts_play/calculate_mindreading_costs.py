#!/usr/bin/env python3
"""
Calculate LLM costs for MindReading/Emotions dataset.
"""

# Pricing
pricing = {
    'gpt-4o-mini': {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    'gpt-4o': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000}
}

# Current EU-Emotion
eu_videos = 54

# MindReading/Emotions
mr_total_videos = 5801
mr_test_10pct = int(mr_total_videos * 0.1)
mr_test_20pct = int(mr_total_videos * 0.2)

def calc_cost(videos, model='gpt-4o-mini', frames=8):
    input_tokens = videos * frames * 85
    output_tokens = videos * 100
    p = pricing[model]
    input_cost = input_tokens * p['input']
    output_cost = output_tokens * p['output']
    return input_cost + output_cost, input_tokens, output_tokens

print('='*70)
print('LLM COST ESTIMATES FOR MINDREADING/EMOTIONS DATASET')
print('='*70)
print()

scenarios = [
    ('EU-Emotion (current)', eu_videos),
    ('MindReading 10% test', mr_test_10pct),
    ('MindReading 20% test', mr_test_20pct),
    ('MindReading full', mr_total_videos),
]

print('gpt-4o-mini (Recommended):')
print('-'*70)
for name, videos in scenarios:
    cost, inp, out = calc_cost(videos, 'gpt-4o-mini')
    print(f'{name:25s} {videos:5d} videos -> ${cost:6.2f} (input: {inp/1000:.0f}k, output: {out/1000:.0f}k tokens)')

print()
print('gpt-4o (Higher Quality):')
print('-'*70)
for name, videos in scenarios:
    cost, inp, out = calc_cost(videos, 'gpt-4o')
    print(f'{name:25s} {videos:5d} videos -> ${cost:6.2f} (input: {inp/1000:.0f}k, output: {out/1000:.0f}k tokens)')

print()
print('Cost with 4 frames (instead of 8):')
print('-'*70)
for name, videos in scenarios:
    cost, _, _ = calc_cost(videos, 'gpt-4o-mini', frames=4)
    print(f'{name:25s} {videos:5d} videos -> ${cost:6.2f} (50% reduction)')
