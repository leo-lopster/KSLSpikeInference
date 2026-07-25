from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

folder = Path('LT_DRG_leg_angles/_model_output/0_deg')
trace_path = folder / 'Averaged_traces_trace.csv'
series = [
    ('GCaMP6f_mouse_30Hz_smoothing200ms_partial_noise2', folder / 'GCaMP6f_mouse_30Hz_smoothing200ms_partial_noise2_spikes.csv'),
    ('GC8_EXC_40Hz_smoothing30ms_high_noise', folder / 'GC8_EXC_40Hz_smoothing30ms_high_noise_spikes.csv'),
    # ('Spinal_cord_excitatory_2.5Hz_smoothing400ms', folder / 'Spinal_cord_excitatory_2.5Hz_smoothing400ms_spikes.csv'),
    ('Spinal_cord_excitatory_30Hz_smoothing50ms', folder / 'Spinal_cord_excitatory_30Hz_smoothing50ms_spikes.csv'),
    # ('Spinal_cord_excitatory_3Hz_smoothing400ms_high_noise', folder / 'Spinal_cord_excitatory_3Hz_smoothing400ms_high_noise_spikes.csv'),
]
rois = [1, 2, 8, 16, 20, 32]

trace_df = pd.read_csv(trace_path)
outputs = []
for label, path in series:
    if not path.exists():
        raise SystemExit(f'Missing input: {path}')
    outputs.append((label, pd.read_csv(path)))

time = trace_df['Time'].to_numpy()
fig, axes = plt.subplots(len(rois), 1, sharex=True, figsize=(15.5, 2.8 * len(rois)))
colors = plt.get_cmap('tab10')
line_handles = []
line_labels = []
trace_handle = None

for row, roi in enumerate(rois):
    ax = axes[row]
    trace_ax = ax.twinx()
    spike_col = f'Spike_ROI{roi}'
    trace_col = f'Trace_ROI{roi}'

    for idx, (label, df) in enumerate(outputs):
        if spike_col not in df.columns:
            continue
        line, = ax.plot(
            time,
            df[spike_col].to_numpy(),
            label=label,
            linestyle='solid',
            linewidth=1.25,
            color=colors(idx % 10),
        )
        if row == 0:
            line_handles.append(line)
            line_labels.append(label)

    if trace_col in trace_df.columns:
        trace_handle, = trace_ax.plot(
            time,
            trace_df[trace_col].to_numpy(),
            color='black',
            linestyle='dotted',
            linewidth=1.0,
            alpha=0.85,
            label='Calcium trace',
        )

    ax.set_title(f'ROI {roi}', fontsize=10)
    ax.set_ylabel('Spike probability')
    trace_ax.set_ylabel('Calcium Trace (dF/F)')
    ax.grid(True, linestyle='--', alpha=0.3)
    trace_ax.grid(False)

    if row == 0:
        handles = line_handles + ([trace_handle] if trace_handle is not None else [])
        labels = line_labels + (['Calcium trace'] if trace_handle is not None else [])
        ax.legend(handles, labels, loc='upper right', fontsize=8, ncol=1)

axes[-1].set_xlabel('Time (s)')
fig.suptitle('0_deg model comparison with calcium traces', fontsize=12)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
out = folder / 'model_comparison_rois_GCaMP6f_GC8_40Hz30ms_Spinal_with_trace.png'
fig.savefig(out, dpi=150)
plt.close(fig)
print(out)