"""Render the trained H=30 Prefix-KV BetaNav architecture; no model weights loaded."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
META = Path('/mnt/cfs/Models/beta_nav/r2r_fm_only_h30_prefix_kv_beta1_1p5/checkpoints/step-00003699/metadata.json')
metadata = json.loads(META.read_text())
cfg = metadata['beta_config']
assert cfg['expert_architecture'] == 'prefix_kv_v1'
assert (cfg['horizon'], cfg['hidden_size'], cfg['num_layers']) == (30, 512, 6)

C = dict(ink='#15273D', muted='#5A6E85', line='#DCE4ED', blue='#3074AD',
         bluebg='#EDF5FC', orange='#C76B18', orangebg='#FFF3E6', purple='#7950AD',
         purplebg='#F4EEFC', green='#21876C', greenbg='#EAF7F1', graybg='#F6F8FB')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(18, 13.6), dpi=170)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 1800), ylim=(1360, 0))
ax.axis('off')
fig.patch.set_facecolor('white')

def text(x, y, label, size=13, color=None, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, label, fontsize=size, color=color or C['ink'], weight=weight,
                   ha=ha, va=va, linespacing=1.45, **kw)

def box(x, y, w, h, label='', fill='graybg', edge='line', size=13, weight='normal'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=11',
                      facecolor=C.get(fill, fill), edgecolor=C.get(edge, edge), linewidth=1.3)
    ax.add_patch(p)
    if label:
        text(x+w/2, y+h/2, label, size=size, weight=weight, ha='center')
    return p

def arrow(x1, y1, x2, y2, color='muted', dashed=False, style='-|>', lw=1.7):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=C.get(color, color),
                        linestyle='--' if dashed else '-', shrinkA=3, shrinkB=3)
    ax.add_patch(p)

text(55, 51, 'BetaNav  |  Frozen VLM + Prefix-KV Flow Action Expert', 24, weight='bold')
text(55, 92, 'Trained architecture  /  30 future actions  /  Beta(1, 1.5)  /  1.0 × FM loss',
     14, color=C['muted'])
box(1360, 34, 385, 60, 'ACTION EXPERT   60.86M', fill='orangebg', edge='orange', size=15, weight='bold')

# Three columns: condition encoder, six action layers, one layer expanded.
box(45, 128, 490, 808, fill='bluebg', edge='line')
box(635, 128, 555, 808, fill='orangebg', edge='line')
box(1230, 128, 525, 808, fill='graybg', edge='line')
text(68, 160, '01  FROZEN CONDITION ENCODER', 14, color=C['blue'], weight='bold')
text(658, 160, '02  TRAINABLE ACTION EXPERT', 14, color=C['orange'], weight='bold')
text(1253, 160, '03  INSIDE EACH ACTION BLOCK', 14, color=C['ink'], weight='bold')

box(72, 190, 436, 68, 'Instruction + image history through t\nSlow–Fast frame sampling / spatial pooling',
    fill='white', edge='blue', size=12.2)
arrow(290, 258, 290, 280, color='blue')
box(72, 280, 436, 76, 'Original LightNav / Qwen3-VL\nVision encoder + 36 language layers',
    fill='white', edge='blue', size=13, weight='bold')
text(290, 378, 'eval()  ·  no_grad()  ·  4.448B frozen parameters', 11.3,
     ha='center', color=C['blue'])

box(662, 190, 500, 62, r'Noisy actions  $x_\tau$ : [B, 30, 3]', fill='white', edge='orange', size=14)
arrow(912, 252, 912, 274, color='orange')
box(662, 274, 500, 68, 'Linear 3 → 512 + learned step embedding\nHidden actions: [B, 30, 512]',
    fill='white', edge='orange', size=12.6)
text(912, 378, '6 blocks  ·  residual width 512  ·  SwiGLU width 2048',
     11.5, ha='center', color=C['orange'])

levels = [6, 12, 18, 24, 30, 36]
for j, level in enumerate(levels):
    y = 410 + j * 66
    if j:
        arrow(290, y-18, 290, y, color='blue')
        arrow(912, y-18, 912, y, color='orange')
    else:
        arrow(290, 388, 290, y, color='blue')
        arrow(912, 388, 912, y, color='orange')
    box(92, y, 396, 48, f'VLM layer {level}  →  native K / V', fill='white', edge='blue', size=13)
    box(672, y, 450, 48, f'Action block {j+1}   ·   KV from layer {level}',
        fill='white', edge='orange', size=13)
    arrow(488, y+24, 672, y+24, color='blue', dashed=True)
text(579, 804, 'detach', 11, color=C['blue'], ha='center')
text(290, 839, 'Each selected K and V: [B, 8, N, 128]\nNative K normalization + multimodal RoPE\nN = padded visual / language prefix length',
     11.5, ha='center', color=C['blue'])
text(290, 906, 'Capture only the 6 selected KV pairs', 11.5, ha='center', weight='bold')

arrow(912, 788, 912, 812, color='orange')
box(662, 812, 500, 46, 'Final time AdaRMSNorm → Linear 512 → 3',
    fill='white', edge='orange', size=12.2)
arrow(912, 858, 912, 878, color='orange')
box(662, 878, 500, 42, r'Flow field  $v_\theta(x_\tau, \tau, C)$ : [B, 30, 3]',
    fill='white', edge='orange', size=13)

# Detailed action block: time modulates attention, FFN and final output norm.
box(1258, 194, 470, 72, 'Time τ → sinusoidal embedding (512)\n→ shared MLP 512 → 512 → 512',
    fill='purplebg', edge='purple', size=12.2)
text(1493, 290, 'Time controls scale, shift and residual gates', 11, color=C['purple'], ha='center')
box(1280, 320, 400, 48, 'Time AdaRMSNorm (attention)', fill='purplebg', edge='purple', size=12.5)
arrow(1490, 299, 1490, 320, color='purple', dashed=True)
arrow(1480, 368, 1480, 388, color='orange')
box(1280, 388, 400, 96, 'Action Q: 32 × 128 = 4096\nAction K / V: 8 × 128 = 1024\nQ / K normalization + action RoPE',
    fill='white', edge='orange', size=12.1)
arrow(1480, 484, 1480, 504, color='orange')
box(1280, 504, 400, 106, 'ONE joint attention / ONE softmax\nQ = action queries\nK / V = [frozen prefix ; noisy actions]',
    fill='white', edge='orange', size=12.1, weight='bold')
text(1480, 631, 'All 30 actions attend bidirectionally', 11.5, color=C['muted'], ha='center')
arrow(1480, 643, 1480, 665, color='orange')
box(1280, 665, 400, 47, 'Output 4096 → 512 · gated residual', fill='white', edge='orange', size=12)
arrow(1480, 712, 1480, 732, color='orange')
box(1280, 732, 400, 47, 'Time AdaRMSNorm (FFN)', fill='purplebg', edge='purple', size=12.5)
arrow(1480, 779, 1480, 799, color='orange')
box(1280, 799, 400, 56, 'SwiGLU 512 → 2048 → 512\n+ gated residual',
    fill='white', edge='orange', size=12.2)
text(1480, 897, '9,964,800 parameters per block', 12, ha='center', weight='bold')
# Purple side bus supplies both branch modulators, without crossing attention labels.
ax.plot([1710,1710],[284,755], color=C['purple'], lw=1.3, linestyle='--')
arrow(1710,344,1680,344,color='purple',dashed=True)
arrow(1710,755,1680,755,color='purple',dashed=True)

# Training and inference contracts.
box(45, 967, 838, 277, fill='greenbg', edge='line')
box(913, 967, 842, 277, fill='graybg', edge='line')
text(70, 1001, 'TRAINING  /  FM loss updates only the orange expert', 13.7, color=C['green'], weight='bold')
text(70, 1047, r'Normalized GT $a$: 30 local SE(2) increments;  $\epsilon \sim \mathcal{N}(0,I)$', 13)
text(70, 1090, r'$\tau \sim \mathrm{Beta}(1,1.5),\qquad x_\tau=(1-\tau)\epsilon+\tau a$', 17)
text(70, 1138, r'$\mathcal{L}_{\mathrm{FM}}=\mathrm{MSE}_{\mathrm{valid}}\!\left(v_\theta,\ a-\epsilon\right)$', 19)
text(70, 1193, 'τ = 0: noise  /  τ = 1: clean actions\nMask invalid GT tail steps in the loss. No CE or RVQ targets.',
     12.2, color=C['muted'])
text(938, 1001, 'INFERENCE  /  reuse the same prefix KV', 13.7, weight='bold')
text(938, 1047, 'Encode the current instruction / history once → cache 6 KV pairs', 12.3)
text(938, 1090, r'$x_0\sim\mathcal{N}(0,I),\quad x_{k+1}=x_k+\frac{1}{10}v_\theta(x_k,k/10,C)$', 16.5)
text(938, 1140, '10 Euler iterations → denormalize → 30 × 3 local increments', 13, weight='bold')
text(938, 1193, 'Compose increments into waypoints.\n30 future action steps and 10 flow iterations are independent.', 12.2, color=C['muted'])
text(55, 1286, 'Trainable: 60,858,371 parameters (1.35% of the full 4.509B model)', 14, weight='bold')
text(55, 1322, 'Source: completed step 3699 metadata + current prefix_kv.py / pi_action_expert.py / beta_model.py',
     11.5, color=C['muted'])
text(1740, 1322, '2026-09-21', 11.5, color=C['muted'], ha='right')

for suffix in ('png','svg','pdf'):
    fig.savefig(ROOT / f'betanav_prefix_kv_architecture.{suffix}', facecolor='white', dpi=170)
plt.close(fig)

paths = [META, REPO/'src/lightnav/inference/beta_model.py', REPO/'src/lightnav/inference/prefix_kv.py',
         REPO/'src/lightnav/inference/pi_action_expert.py',
         REPO/'runs/betanav_launch/20260920T081142Z_fm_only_h30_prefix_kv_beta1_1p5_8gpu/train_config.yaml']
sources = dict(checkpoint=str(META.parent), beta_config=cfg,
               sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(ROOT/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
print(ROOT/'betanav_prefix_kv_architecture.png')
