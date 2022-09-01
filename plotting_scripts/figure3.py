import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import scipy.optimize as opti

### Turn off annoying message about substituting symbol \perp from STIXGeneral
import logging
logger = logging.getLogger('matplotlib.mathtext')
logger.setLevel(logging.ERROR)

def fitpower(x,a,b):
    return a*x**b

def fit(xdata, ydata, ff, p0=np.array([0., 0.])):
    sigma = np.ones_like(xdata)
    
    popt, pcov = opti.curve_fit(ff, xdata, ydata, p0, sigma)
    
    return popt

def rsquared(xdata,ydata,fitconst,ff):
    f=ff(xdata,*fitconst)
    yminusf2=(ydata-f)**2
    sserr=sum(yminusf2)
    mean=float(sum(ydata))/float(len(ydata))
    yminusmean2=(ydata-mean)**2
    sstot=sum(yminusmean2)
    return 1.-(sserr/sstot)

fontsize = 8
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize

dfEarth = pd.read_csv('../reduced_data/craterStats_Earth.txt',
                      delim_whitespace=True, header=0, comment='#').drop(
        ['topo', 'tind1', 'tind2', 'fsind', 'feind'], axis=1)  # remove some unnecessary columns

dfMoon = pd.read_csv('../reduced_data/craterStats_Moon.txt',
                     delim_whitespace=True, header=0, comment='#').drop(
        ['topo', 'tind1', 'tind2', 'fsind', 'feind'], axis=1)  # remove some unnecessary columns

# Add columns for pi2 and D/L
def pi2(L, v, g=9.81):
    return 1.61 * g * (L * 1000) / (v * 1000)**2

def pi2a(L, v, a, g=9.81):
    return 1.61 * g * (L * 1000) / (v * 1000 * np.sin(np.radians(a)))**2

for df, g in zip((dfEarth, dfMoon), (9.81, 1.63)):
    df.insert(3, 'pi2', pi2(df['L'], df['U'], g))
    df.insert(4, 'pi2a', pi2a(df['L'], df['U'], df['ang'], g))
    df.insert(5, 'D/L', df['final_eqrad'] * 2. / df['L'])  # D/L using equivalent final radius
    df.insert(6, 'Dt/L', df['trans_eqrad'] * 2. / df['L'])  # D/L using equivalent transient radius
    df.insert(7, 'Dr/L', df['rimrad'] * 2. / df['L'])
    df.insert(8, 'piD', df['D/L'] * (6 / np.pi)**(1/3))
    df.insert(9, 'piDt', df['Dt/L'] * (6 / np.pi)**(1/3))
    df.insert(10, 'piDr', df['Dr/L'] * (6 / np.pi)**(1/3))

mmtoinch = 1. / 25.4

fw = 95 * mmtoinch
fh = 120 * mmtoinch

# Create the Figure and Axes
nrows=2
fig, (ax1, ax2) = plt.subplots(nrows=nrows, figsize=(fw, fh))

fig.subplots_adjust(right=0.96, hspace=0.32, top=0.88, bottom=0.09, left=0.175)

abc = 'abcdefghijklmopqrstuvwxyz'

for ax, letter in zip(fig.axes, abc[:nrows]):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

    ax.text(-0.22, 1, '{})'.format(letter), ha='left', va='top',transform=ax.transAxes)

# Pairs of velocities and impactor diameters for Earth-based models
earth_vLlist = list(zip((10., 15., 20., 30., 20., 25., 20., 30.),
                        (14., 14., 14., 14., 8.96, 8.96, 6.22, 6.22)))
# And for the Moon
moon_vLlist = list(zip(( 5., 15., 20., 30., 20., 25., 20., 30.),
                        (14., 14., 14., 14., 8.96, 8.96, 6.22, 6.22)))

# Markers to use for each impactor size or velocity
mdict = {'14': 'o', '09': 's', '06': 'D'}
vdict = {'05': '^', '10': '+', '15': 'x', '20': 'o', '25': 's', '30': 'D'}
marker = vdict

xt, yt = [], []
xfe, yfe = [], []
xfm, yfm = [], []

ang_thres = 45

for df, vLlist, col in zip((dfEarth, dfMoon), (earth_vLlist, moon_vLlist), ('navy', 'skyblue')):

    for v, L in vLlist:

        if marker == mdict:

            thismarker = mdict['{:02.0f}'.format(L)]

        elif marker == vdict:

            thismarker = vdict['{:02.0f}'.format(v)]

        suite = df[(df['U'] == v) & (df['L'] == L)]

        angmask = suite['ang'] >= ang_thres
        ax1.plot(suite['pi2'][angmask], suite['piDt'][angmask],
                 marker=thismarker, mec=col, mfc='None', ms=3.5, ls='None', zorder=20)
        ax1.plot(suite['pi2'][~angmask], suite['piDt'][~angmask],
                 marker=thismarker, mec='#CCCCCC', mfc='None', ms=3.5, ls='None', zorder=10)
        ax2.plot(suite['trans_eqrad']*2, suite['rimrad']*2,
                 marker=thismarker, mec=col, mfc='None', ms=3.5, ls='None')

        for xx in suite['pi2'][angmask]:
            xt.append(xx)
        for yy in suite['piDt'][angmask]:
            yt.append(yy)

        for xx in suite['trans_eqrad']:  #[angmask]:
            if col == 'navy':
                xfe.append(xx*2)
            else:
                xfm.append(xx*2)

        for yy in suite['rimrad']:  #[angmask]:
            if col == 'navy':
                yfe.append(yy*2)
            else:
                yfm.append(yy*2)

ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel('$\pi_2 = 1.61 g L / u^2$')
ax2.set_xlabel('Transient diameter, $D_{tr,eq}$ [km]')

ax1.set_ylabel('$\pi_D = D_{tr,eq}\ (\\rho / m)^{1/3}$')
ax2.set_ylabel('Final rim-to-rim\ndiameter, $D_{f,rr}$ [km]')

ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, loc: '{:g}'.format(x)))
ax1.yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))

ax1.set_ylim(4.5, 18)

pt = fit(xt, yt, fitpower, np.array([1.6, -0.22]))
print("CD, beta = ", pt)
print("R^2 = ", rsquared(np.array(xt), np.array(yt), pt, fitpower))
xlim = ax1.get_xlim()
xr = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 10)
l, = ax1.plot(xr, fitpower(xr, *pt), 'k-', label = '$\pi_D = {:.2f} \pi_2^{{{:.2f}}}$'.format(*pt))

leg1 = ax1.legend(handles=[l], loc=3)
ax1.add_artist(leg1)

if marker == vdict:

    vlines = []

    for vv, mm in vdict.items():

        vlines.append(mlines.Line2D([], [], mec='#000000', mfc='None',
                                    marker=vdict[vv], markersize=5,
                                    linestyle='None',
                                    label='$u = {}$ km s$^{{-1}}$'.format(int(vv))))

    vlines.append(mlines.Line2D([], [], color='navy', mec='None', marker='s',
                                markersize=6, linestyle='None', label='Earth'))
    vlines.append(mlines.Line2D([], [], color='skyblue', mec='None', marker='s',
                                markersize=6, linestyle='None', label='Moon'))

    ax1.legend(handles=vlines, handletextpad=0.5, fontsize=6, ncol=3,
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               mode="expand", borderaxespad=0., borderpad=0.8)

def trans_final(Dt, a, eta, c):

    return a * Dt**(1+eta) / c**eta

dsce = 4.
dscm = 20.
pe = fit(xfe, yfe, lambda x, a, b: trans_final(x, a, b, dsce), np.array([1.17, 1.13]))
pm = fit(xfm, yfm, lambda x, a, b: trans_final(x, a, b, dscm), np.array([1.17, 1.13]))

xr2 = np.arange(25, 150, 1)
ax2.plot(xr2, trans_final(xr2, pe[0], pe[1], dsce), '-', color='navy',
        label='$D_f = {:.2f} D_t^{{{:.2f}}} / {:.0f}^{{{:.2f}}} \ \ (\gamma={:.2f})$'.format(
            pe[0], pe[1]+1, dsce, pe[1], pe[0]**(1 / (1 + pe[1]))))
ax2.plot(xr2, trans_final(xr2, pm[0], pm[1], dscm), '-', color='skyblue',
        label='$D_f = {:.2f} D_t^{{{:.2f}}} / {:.0f}^{{{:.2f}}} \ \ (\gamma={:.2f})$'.format(
            pm[0], pm[1]+1, dscm, pm[1], pm[0]**(1 / (1 + pm[1]))))

ax2.legend()

fig.savefig('figure3.pdf')
fig.savefig('figure3.png', dpi=300)

