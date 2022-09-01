import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

fontsize = 8
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize

mmtoinch = 1 / 25.4

fw = 190 * mmtoinch
fh = 150 * mmtoinch

fig = plt.figure(figsize=(fw, fh))
fig.subplots_adjust(left=0.08, right=0.97,
                    bottom=0.08, top=0.97,
                    hspace=0.06, wspace=0.20)

gs1 = gridspec.GridSpec(nrows=4, ncols=2)
gs2 = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=(30, 13))

ax11 = fig.add_subplot(gs1[0, 0], aspect='equal')
ax12 = fig.add_subplot(gs1[1, 0], aspect='equal')
ax13 = fig.add_subplot(gs1[2, 0], aspect='equal')
ax14 = fig.add_subplot(gs1[3, 0], aspect='equal')

A45_v20_L14 = np.load('../reduced_data/A45_v20_L14_slices.npz')
xx = A45_v20_L14['x']
zz = A45_v20_L14['z']

def plot_step(ax, t):

    dd = A45_v20_L14['d{}'.format(t)]
    mm = A45_v20_L14['m{}'.format(t)]
    Den = np.ma.masked_array(dd, mask=mm)
    p = ax.pcolormesh(xx, zz, Den,
                      vmin=2000, vmax=4000, cmap='magma',
                      zorder=-1) 
    
    ax.set_xlim(-150, 80)
    ax.set_ylim(-55, 40)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_ylabel('z [km]')
    ax.set_rasterization_zorder(0) 

    ax.text(x=0.02, y=0.88,
            s='$t = {:3.0f}$s'.format(A45_v20_L14['t'][t]),
            transform=ax.transAxes)

    ax.label_outer()

    return p

for t, ax in enumerate((ax11, ax12, ax13, ax14)):
    p = plot_step(ax, t)

ax14.set_xlabel('x [km]')

xoff = -0.16
ax11.text(x=xoff, y=1, s='a)', transform=ax11.transAxes,
          va='top')
ax12.text(x=xoff, y=1, s='b)', transform=ax12.transAxes,
          va='top')
ax13.text(x=xoff, y=1, s='c)', transform=ax13.transAxes,
          va='top')
ax14.text(x=xoff, y=1, s='d)', transform=ax14.transAxes,
          va='top')

cax = fig.add_axes([0.30, 0.25, 0.17, 0.01])
cb = fig.colorbar(p, cax=cax, orientation='horizontal',
                  ticks=[2000, 4000])
cb.set_label('Density [kg/m$^3$]')
cax.xaxis.set_label_position('top')
for t, ha in zip(cb.ax.get_xticklabels(),
                 ('left', 'right')):
    t.set_ha(ha)
cb.update_ticks()
cb.ax.xaxis.set_tick_params(pad=1)

ax11.annotate(xytext=(0, 7), xy=(-21, -10.5),
              text='', color='k',
              arrowprops={'arrowstyle': 'simple',
                          'shrinkA': 0, 'shrinkB': 0,
                          'facecolor': 'k',
                          'edgecolor': 'w', 
                          'linewidth': 0.5}
             )

ax11.text(x=0.02, y=0.47, s='Granite crust',
         color='w', transform=ax11.transAxes)
ax11.text(x=0.02, y=0.05, s='Dunite mantle',
         color='k', transform=ax11.transAxes)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

angles = [30, 40, 45, 50, 60, 75, 90]

axT = fig.add_subplot(gs2[0, 1])
axF = fig.add_subplot(gs2[1, 1], sharex=axT)

axT.set_ylim(0, 300)
axF.set_ylim(0, 65)
for ax in (axT, axF):
    ax.set_xlim(-140, 120)
    ax.set_ylabel('z [km]')

axT.yaxis.set_major_locator(ticker.MultipleLocator(50))
axF.yaxis.set_major_locator(ticker.MultipleLocator(25))

axF.set_xlabel('x [km]')

axT.set_xticklabels([])
axT.set_aspect('equal')
axF.set_aspect(2)

def loadfile(stage):

    files = np.load('../reduced_data/v20_L14_profiles_{}.npz'.format(stage), allow_pickle=True)

    topos = files['topo'][()]
    xs = files['x'][()]

    return topos, xs

topos_tr, xs = loadfile('trans')
topos_fn, _ = loadfile('final')

for j, i in enumerate(angles[::-1]):

    col = plt.cm.magma(0.12+j/9.)
    ii = '{}'.format(i)
    minpt = 0
   
    Toff = (j + 1) * 38
    Foff = (j + 1) * 8
    axT.axhline(Toff, color='#AAAAAA', lw=1)
    axF.axhline(Foff, color='#AAAAAA', lw=1)

    axT.plot((xs[ii] - minpt)/1e3, Toff + topos_tr[ii][:, 0]/1e3, '-',
             c=col, label='$\\theta={}^\circ$'.format(i))
    axF.plot((xs[ii] - minpt)/1e3, Foff + topos_fn[ii][:, 0]/1e3, '-',
             c=col, label='$\\theta=${}'.format(i))

    axT.text(120, 2 + Toff, '$\\theta={}^\circ$'.format(i), ha='right', va='bottom', color=col)


axT.text(0.50, 0.9826, 'Transient crater', ha='center', va='top', transform=axT.transAxes)
axF.text(0.50, 0.96, 'Final crater', ha='center', va='top', transform=axF.transAxes)

axT.text(x=xoff, y=1, s='e)', ha='left', va='top', transform=axT.transAxes)
axF.text(x=xoff, y=1, s='f)', ha='left', va='top', transform=axF.transAxes)

fig.savefig('figure1.png', dpi=300)
fig.savefig('figure1.pdf', dpi=300)
