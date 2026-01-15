#!/usr/bin/env python3
"""
Visualization script to compare fish trajectories with and without pitch/roll corrections.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
reference_file = '/home/david-james/Desktop/01-Marlin-Tracking/MarlinTracking/0_Notebooks/02-inference-output/BB36_mob_m1/02-gps-coordinates/world-trajectory.csv'
pitch_roll_file = '/tmp/test_world_trajectory_with_pitch_roll.csv'

# Load data
df_original = pd.read_csv(reference_file)
df_corrected = pd.read_csv(pitch_roll_file)

# Select the 3 fish IDs with the longest trajectories
trajectory_lengths = df_original.groupby('target_id').size().sort_values(ascending=False)
selected_ids = trajectory_lengths.head(3).index.tolist()
print(f'Selected fish IDs with longest trajectories: {selected_ids}')
print(f'Trajectory lengths: {trajectory_lengths.head(3).values}')

# Create figure with 3 subplots (one per fish)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Fish Trajectory Comparison: Original (Pitch=0, Roll=0) vs Corrected (Real Pitch & Roll)',
             fontsize=14, fontweight='bold')

# Plot each fish in its own subplot
for idx, fish_id in enumerate(selected_ids):
    ax = axes[idx]

    # Get data for this fish
    fish_orig = df_original[df_original['target_id'] == fish_id]
    fish_corr = df_corrected[df_corrected['target_id'] == fish_id]

    # Plot original trajectory (blue, solid)
    ax.plot(fish_orig['avg_pos_x'], fish_orig['avg_pos_y'],
            color='#1f77b4', linewidth=2.5, alpha=0.7, linestyle='-',
            label='Original (no pitch/roll)', zorder=1)
    # Start/end markers for original
    ax.scatter(fish_orig['avg_pos_x'].iloc[0], fish_orig['avg_pos_y'].iloc[0],
              color='#1f77b4', s=120, marker='o', edgecolors='black', linewidth=2, zorder=3)
    ax.scatter(fish_orig['avg_pos_x'].iloc[-1], fish_orig['avg_pos_y'].iloc[-1],
              color='#1f77b4', s=120, marker='s', edgecolors='black', linewidth=2, zorder=3)

    # Plot corrected trajectory (red, solid)
    ax.plot(fish_corr['avg_pos_x'], fish_corr['avg_pos_y'],
            color='#d62728', linewidth=2.5, alpha=0.8, linestyle='-',
            label='Corrected (with pitch/roll)', zorder=2)
    # Start/end markers for corrected
    ax.scatter(fish_corr['avg_pos_x'].iloc[0], fish_corr['avg_pos_y'].iloc[0],
              color='#d62728', s=120, marker='o', edgecolors='black', linewidth=2, zorder=4)
    ax.scatter(fish_corr['avg_pos_x'].iloc[-1], fish_corr['avg_pos_y'].iloc[-1],
              color='#d62728', s=120, marker='s', edgecolors='black', linewidth=2, zorder=4)

    # Compute displacement stats for this fish
    orig_pts = fish_orig[['avg_pos_x', 'avg_pos_y']].values
    corr_pts = fish_corr[['avg_pos_x', 'avg_pos_y']].values
    displacement = np.sqrt(np.sum((orig_pts - corr_pts)**2, axis=1))
    mean_disp = np.mean(displacement)

    ax.set_xlabel('X Position (meters)', fontsize=10)
    ax.set_ylabel('Y Position (meters)', fontsize=10)
    ax.set_title(f'Fish {fish_id} ({len(fish_orig)} points)\nMean displacement: {mean_disp:.2f}m',
                fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

# Add legend explanation
fig.text(0.5, 0.02, 'Circle (●) = Start, Square (■) = End',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save figure
output_file = '/tmp/trajectory_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'✅ Plot saved to: {output_file}')

# Compute and display statistics
print('\n' + '='*60)
print('TRAJECTORY DISPLACEMENT STATISTICS')
print('='*60)

for fish_id in selected_ids:
    orig = df_original[df_original['target_id'] == fish_id][['avg_pos_x', 'avg_pos_y']].values
    corr = df_corrected[df_corrected['target_id'] == fish_id][['avg_pos_x', 'avg_pos_y']].values

    # Compute point-by-point displacement
    displacement = np.sqrt(np.sum((orig - corr)**2, axis=1))

    print(f'\nFish {fish_id}:')
    print(f'  Mean displacement: {np.mean(displacement):.2f} meters')
    print(f'  Max displacement:  {np.max(displacement):.2f} meters')
    print(f'  Min displacement:  {np.min(displacement):.2f} meters')
    print(f'  X-shift range:     [{np.min(orig[:,0] - corr[:,0]):.2f}, {np.max(orig[:,0] - corr[:,0]):.2f}] meters')

plt.show()
