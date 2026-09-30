"""Paper figures: trained six-layer model or proposed all-layer variant."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle

ALL_LAYERS = '--all-layers' in sys.argv
OUT = Path(__file__).resolve().parent / ('paper_all36_proposal' if ALL_LAYERS else 'paper_v2')
OUT.mkdir(exist_ok=True)
cfg = json.loads((OUT.parent/'sources.json').read_text())['beta_config']
assert cfg['prefix_layers'] == [6, 12, 18, 24, 30, 36] and cfg['horizon'] == 30
assert json.loads((OUT.parent/'parameter_counts.json').read_text())['total_parameters'] == 60858371

plt.rcParams.update({'font.family':'DejaVu Sans', 'mathtext.fontset':'stix',
                     'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(7.2, 4.08))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 1800), ylim=(1020, 0))
ax.axis('off')
I='#243246'; M='#637083'; L='#D9DEE7'
B='#427DAD'; BF='#EDF4FA'; O='#BA7538'; OF='#FCF0E3'; P='#8962A3'; PF='#F3EDF7'
G='#428675'; GF='#EAF4EF'

def label(x,y,s,fs=8,ha='center',va='center',color=I,bold=False,**kw):
    return ax.text(x,y,s,fontsize=fs,ha=ha,va=va,color=color,
                   weight='bold' if bold else 'normal',linespacing=1.3,zorder=8,**kw)

def rect(x,y,w,h,fc='white',ec=L,r=9,lw=.65,ls='-',z=2):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    fc=fc,ec=ec,lw=lw,linestyle=ls,zorder=z)
    ax.add_patch(p)
    return p

def node(x,y,w,h,s,fc='white',ec=L,fs=8,bold=False):
    rect(x,y,w,h,fc,ec)
    label(x+w/2,y+h/2,s,fs=fs,bold=bold)

def line(points,c=M,lw=.7,ls='-',z=1):
    x,y=zip(*points)
    ax.plot(x,y,color=c,lw=lw,linestyle=ls,zorder=z,solid_capstyle='round')

def arr(x1,y1,x2,y2,c=M,lw=.75,ls='-',z=3):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=5.2,
                     lw=lw,color=c,linestyle=ls,shrinkA=1.2,shrinkB=1.2,zorder=z)
    ax.add_patch(p)

def elbow(points,c=M,lw=.75,ls='-'):
    line(points[:-1],c,lw,ls)
    arr(*points[-2],*points[-1],c,lw,ls)

def op(x,y,s):
    ax.add_patch(Circle((x,y),12.5,fc='white',ec=I,lw=.7,zorder=5))
    label(x,y,s,fs=8.2)

def lock(x,y,scale=1,c=B):
    rect(x-6*scale,y,12*scale,11*scale,fc='white',ec=c,r=2,lw=.6,z=4)
    t=np.linspace(0,np.pi,30)
    ax.plot(x+4*scale*np.cos(t),y-5*scale*np.sin(t),color=c,lw=.6,zorder=4)

def room(x,y,w=152,h=106,variant=0):
    """Schematic observation card, not a sampled dataset image."""
    rect(x,y,w,h,'white','#AEBCCD',r=3,lw=.6,z=3)
    pad=5
    x0,y0=x+pad,y+pad; w0,h0=w-2*pad,h-2*pad
    ax.add_patch(Rectangle((x0,y0),w0,h0,fc='#E6E9EA',ec='none',zorder=3))
    cx=x0+w0*(.56-.06*variant); cy=y0+h0*.56
    polys=[([(x0,y0),(x0+w0,y0),(cx+19,cy-22),(cx-25,cy-22)],'#F3F2ED'),
           ([(x0,y0),(cx-25,cy-22),(cx-25,cy+18),(x0,y0+h0)],'#D5DDDF'),
           ([(x0+w0,y0),(cx+19,cy-22),(cx+19,cy+18),(x0+w0,y0+h0)],'#E2E7E7'),
           ([(x0,y0+h0),(cx-25,cy+18),(cx+19,cy+18),(x0+w0,y0+h0)],'#C5D0D3')]
    for verts,fc in polys: ax.add_patch(Polygon(verts,fc=fc,ec='none',zorder=4))
    ax.add_patch(Rectangle((cx-11,cy-18),20,36,fc='#9FAFB5',ec='#8296A1',lw=.25,zorder=5))
    ax.add_patch(Polygon([(x0+7,y0+h0*.6),(cx-32,cy+2),(cx-32,cy+14),(x0+7,y0+h0*.82)],
                         fc='#9DB7BA',ec='none',zorder=5))
    line([(cx,cy+19),(x0+w0*.7,y0+h0)],'#A0B2BB',.3,z=5)

# Panel headings and compact color semantics.
label(43,39,'(a)  All-layer visual–language conditioning' if ALL_LAYERS else '(a)  Layer-wise visual–language conditioning',9.4,ha='left',bold=True)
label(1219,39,'(b)  Time-adaptive action block',9.4,ha='left',bold=True)
line([(1188,73),(1188,796)],L,.6)
rect(46,75,17,17,BF,B,r=2); label(76,84,'Frozen',6.7,ha='left',color=M)
rect(219,75,17,17,OF,O,r=2); label(249,84,'Trainable',6.7,ha='left',color=M)
line([(435,84),(477,84)],P,.7,'--'); label(492,84,'Time modulation',6.7,ha='left',color=M)
if ALL_LAYERS:
    label(923,84,'PROPOSED · 36 ↔ 36',6.6,color=O,bold=True)

# Input observations and instruction.
label(173,151,'Language instruction',6.8,bold=True)
node(46,175,255,79,'“Turn left past\nthe piano …”',fc='#F7F8FA',fs=7.7)
label(173,309,'Observation history',6.8,bold=True)
room(62,340,variant=0)
room(100,362,variant=1)
room(138,384,variant=2)
label(174,514,r'$I_{t-k},\ldots,I_t$',8.8,color=M)
arr(173,532,173,563,B)
node(46,568,255,77,'Slow–Fast history\nVisual encoder',BF,B,fs=6.9)
lock(316,580,1.05)

# Frozen multimodal language backbone: retained native KV at six layers.
rect(371,124,338,557,fc='#F8FBFD',ec='#ADC5D8',r=14,lw=.6,ls='--',z=0)
label(530,148,'Frozen LightNav',7.8,bold=True,color=B)
lock(680,143,1.0)
node(398,180,284,67,'Multimodal prefix',BF,B,fs=8.0)
elbow([(302,214),(354,214),(398,214)],B)
elbow([(302,607),(342,607),(342,235),(398,235)],B)
label(539,277,'36 language layers\nALL 36 KV pairs' if ALL_LAYERS else '36 language layers\n6 selected KV taps',6.1,color=M)
arr(539,300,539,312,B)

# Noisy action sequence and expert input embedding.
rect(812,124,316,615,fc='#FFFCF8',ec='#D9B793',r=14,lw=.6,ls='--',z=0)
label(970,148,'Flow action expert',7.4,bold=True,color=O)
node(830,180,280,58,r'$x_\tau\in\mathbb{R}^{30\times3}$',OF,O,fs=10)
arr(970,238,970,256,O)
node(830,257,280,39,'Action embedding',OF,O,fs=7.4)
arr(970,296,970,312,O)
label(752,285,'Native KV',6.7,color=B)

shown_layers = [1,2,3,34,35,36] if ALL_LAYERS else cfg['prefix_layers']
shown_y = [311,360,409,505,554,603] if ALL_LAYERS else [313+j*58 for j in range(6)]
for j,(layer,y) in enumerate(zip(shown_layers, shown_y)):
    node(398,y,284,43,f'Layer {layer}',BF,B,fs=7.7)
    node(830,y,280,43,f'Action block {layer if ALL_LAYERS else j+1}',OF,O,fs=7.7)
    arr(684,y+21.5,827,y+21.5,B,lw=.7,ls='--')
    if j<5 and not (ALL_LAYERS and j==2):
        arr(539,y+43,539,shown_y[j+1],B,lw=.6)
        arr(970,y+43,970,shown_y[j+1],O,lw=.6)
if ALL_LAYERS:
    label(539,477,r'$\vdots$',11,color=B)
    label(970,477,r'$\vdots$',11,color=O)
    label(752,474,'4–33: identical links',5.5,color=B)
    arr(684,487,827,487,B,lw=.6,ls='--')

label(752,666,'stop-grad',6.5,color=B)
label(539,706,r'$K_p^{(\ell)},V_p^{(\ell)}\in\mathbb{R}^{8\times N\times128}$',8.2,color=B)
arr(970,646,970,669,O)
node(830,670,280,45,'AdaRMSNorm → Linear',OF,O,fs=6.4)
arr(970,715,970,749,O)
label(970,771,r'$v_\theta(x_\tau,\tau,C)\in\mathbb{R}^{30\times3}$',9.3)

# Shared time signal to all expert layers and final normalization.
ax.add_patch(Circle((1153,256),18,fc=PF,ec=P,lw=.7,zorder=4))
label(1153,256,r'$\tau$',10,color=P)
line([(1153,277),(1153,694)],P,.7,'--')
for y in [v+21.5 for v in shown_y]+[692.5]:
    arr(1153,y,1110,y,P,lw=.55,ls='--')

label(47,709,'36 blocks · d = 512' if ALL_LAYERS else '6 blocks · d = 512',7.0,ha='left',bold=True)
label(47,744,'359.80M trainable' if ALL_LAYERS else '60.86M trainable',7.0,ha='left',color=O)
label(47,778,'Frozen prefix reused at every flow step.',6.4,ha='left',color=M)

# Expanded block, drawn as a real residual computational graph.
label(1448,126,r'$h_j$',10.5)
arr(1448,141,1448,208)
elbow([(1448,170),(1248,170),(1248,441),(1432,441)],I,lw=.6)
node(1293,210,310,54,'AdaRMSNorm',PF,P,fs=8)
arr(1448,264,1448,287)
rect(1293,288,310,90,OF,O)
label(1448,310,'Joint attention',8,bold=True)
label(1448,349,r'$Q_a\ ;\ [K_p;K_a],\ [V_p;V_a]$',8.6)
arr(1448,378,1448,391)
op(1448,405,'×')
arr(1448,419,1448,427)
op(1448,441,'+')
arr(1448,455,1448,477)
elbow([(1448,465),(1248,465),(1248,695),(1432,695)],I,lw=.6)
node(1293,479,310,54,'AdaRMSNorm',PF,P,fs=8)
arr(1448,533,1448,556)
node(1293,558,310,61,'SwiGLU FFN',OF,O,fs=8)
arr(1448,619,1448,635)
op(1448,649,'×')
arr(1448,663,1448,681)
op(1448,695,'+')
arr(1448,709,1448,742)
label(1448,764,r'$h_{j+1}$',10.5)

# Native prefix enters attention once; time generates both AdaNorms and gates.
label(1668,316,r'$K_p^{(j)}, V_p^{(j)}$' if ALL_LAYERS else r'$K_p, V_p$',7.5 if ALL_LAYERS else 8.0,color=B)
arr(1688,343,1605,343,B)
label(1664,95,r'$\tau$',10.5,color=P)
arr(1664,108,1664,137,P,ls='--')
node(1582,139,164,59,'Time MLP',PF,P,fs=7.3)
line([(1664,198),(1664,214),(1742,214),(1742,649)],P,.7,'--')
for y in (237,506):
    arr(1742,y,1603,y,P,lw=.65,ls='--')
for y in (405,649):
    arr(1742,y,1463,y,P,lw=.65,ls='--')
label(1634,389,r'$g_{\mathrm{attn}}(\tau)$',7.2,color=P)
label(1634,633,r'$g_{\mathrm{ffn}}(\tau)$',7.2,color=P)
label(1450,799,'GQA: 32 Q / 8 KV heads · head dimension 128',6.7,color=M)

# A compact, visual flow-matching strip, suitable for an external paper caption.
line([(43,827),(1753,827)],L,.6)
label(44,855,'(c)  Conditional flow matching',9.0,ha='left',bold=True)
label(54,908,r'$x_\tau=(1-\tau)\epsilon+\tau a,\quad \tau\sim\mathrm{Beta}(1,1.5)$',9.3,ha='left')
label(54,963,r'$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{\tau,\epsilon}\!\left[\mathrm{MSE}_{\mathrm{valid}}\!\left(v_\theta,\,a-\epsilon\right)\right]$',
      9.2,ha='left')
label(1063,855,'Sampling',8.2,ha='left',bold=True)
label(1610,864,'Denormalize + compose',6.4,color=M)
rng=np.random.default_rng(28)
xy=rng.normal(size=(30,2))*[28,23]+[1085,932]
ax.scatter(xy[:,0],xy[:,1],s=2.3,c='#A6ACB7',alpha=.85,zorder=3,linewidths=0)
label(1085,994,r'$x_0=\epsilon$',7.6,color=M)
arr(1137,932,1189,932,G)
node(1194,901,233,62,'10 Euler steps',GF,G,fs=8)
arr(1430,932,1484,932,G)
t=np.linspace(0,1,30)
px=1510+170*t; py=962-69*t-24*np.sin(t*np.pi*1.4)
ax.plot(px,py,c=G,lw=.85,zorder=4)
ax.scatter(px[::2],py[::2],s=2.5,c=G,edgecolors='white',linewidths=.2,zorder=5)
arr(px[-2],py[-2],px[-1]+8,py[-1]-2,G,lw=.8)
label(1610,994,'30-step trajectory',7.2,color=M)
label(1310,991,'reuse prefix KV',6.5,color=M)

for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'betanav_paper_architecture.{ext}',dpi=500,facecolor='white')
plt.close(fig)
print(OUT/'betanav_paper_architecture.png')

if ALL_LAYERS:
    # Companion drawing explicitly expands every one-to-one connection.
    # The main paper figure compresses repeated layers using an ellipsis.
    fig, ax = plt.subplots(figsize=(7.2, 9.4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,1800),ylim=(2350,0))
    ax.axis('off')
    label(75,55,'All-layer Prefix-KV action expert',13,ha='left',bold=True)
    label(75,108,'Proposed architecture · every VLM layer has a corresponding action block',7.3,ha='left',color=M)
    label(465,175,'Frozen LightNav',10,color=B,bold=True)
    label(1355,175,'Trainable action expert',10,color=O,bold=True)
    node(200,221,530,83,'Instruction + observation prefix',BF,B,fs=8.7)
    node(1090,221,530,83,r'Noisy actions $x_\tau\in\mathbb{R}^{30\times3}$',OF,O,fs=9.2)
    label(905,270,r'$K_p^{(j)},V_p^{(j)}$',9.4,color=B)
    label(905,312,'native KV · stop-gradient',7.2,color=B)
    arr(465,307,465,345,B)
    arr(1355,307,1355,345,O)
    # Six visual bands group rows for readability, but no layer is omitted.
    for group in range(6):
        y=344+group*288
        rect(122,y-9,1556,282,fc='#F8FAFC' if group%2==0 else 'white',ec=L,r=9,lw=.4,z=0)
        label(78,y+122,f'{group*6+1}–{group*6+6}',6.8,color=M,rotation=90)
        for offset in range(6):
            j=group*6+offset+1
            yy=y+offset*46
            node(200,yy,530,36,f'VLM layer {j:02d}',BF,B,fs=7.8)
            node(1090,yy,530,36,f'Action block {j:02d}',OF,O,fs=7.8)
            arr(735,yy+18,1085,yy+18,B,lw=.8,ls='--')
            label(905,yy+12,fr'$K_{{{j}}}, V_{{{j}}}$',6.8,color=B,
                  bbox=dict(facecolor='white',edgecolor='none',pad=.8))
            if j<36:
                next_y = yy+46 if offset<5 else y+288
                arr(465,yy+36,465,next_y,B,lw=.55)
                arr(1355,yy+36,1355,next_y,O,lw=.55)
            arr(1715,yy+18,1624,yy+18,P,lw=.55,ls='--')
    line([(1715,308),(1715,2154)],P,.7,'--')
    ax.add_patch(Circle((1715,278),25,fc=PF,ec=P,lw=.7,zorder=4))
    label(1715,278,r'$\tau$',11,color=P)
    arr(1355,2050,1355,2103,O)
    node(1090,2105,530,82,'Time AdaRMSNorm → Linear\n30 × 3 flow field',OF,O,fs=8.6)
    arr(1715,2154,1624,2154,P,lw=.6,ls='--')
    label(199,2120,'ONE-TO-ONE AT EVERY LAYER',8,ha='left',bold=True)
    label(199,2170,r'$j\in\{1,\ldots,36\}:\quad(K_p^{(j)},V_p^{(j)})\longrightarrow\mathrm{Block}_j$',9.3,ha='left')
    line([(75,2222),(1725,2222)],L,.6)
    label(75,2268,'36 blocks · width 512 · 359,802,371 trainable parameters',8.9,ha='left',bold=True)
    label(75,2320,'Each block reads its matching VLM layer. The trained 6-layer checkpoint remains a separate model.',7.0,ha='left',color=M)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'betanav_all36_expanded.{ext}',dpi=400,facecolor='white')
    plt.close(fig)
    print(OUT/'betanav_all36_expanded.png')
