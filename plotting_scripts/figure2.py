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

dfEarth = pd.read_csv('../reduced_data/craterStats_Earth.txt',
                      delim_whitespace=True, header=0, comment='#').drop(
        ['topo', 'tind1', 'tind2', 'fsind', 'feind'], axis=1)  # remove some unnecessary columns

dfMoon = pd.read_csv('../reduced_data/craterStats_Moon.txt',
                     delim_whitespace=True, header=0, comment='#').drop(
        ['topo', 'tind1', 'tind2', 'fsind', 'feind'], axis=1)  # remove some unnecessary columns

# Add columns for pi2 and D/L
def pi2(L, v, g=9.81):
    return g * (L * 1000) / (v * 1000)**2

def pi2a(L, v, a, g=9.81):
    return g * (L * 1000) / (v * 1000 * np.sin(np.radians(a)))**2

for df, g in zip((dfEarth, dfMoon), (9.81, 1.63)):
    df.insert(3, 'pi2', pi2(df['L'], df['U'], g))
    df.insert(4, 'pi2a', pi2a(df['L'], df['U'], df['ang'], g))
    df.insert(5, 'D/L', df['final_eqrad'] * 2. / df['L'])  # D/L using equivalent final radius
    df.insert(6, 'Dt/L', df['trans_eqrad'] * 2. / df['L'])  # D/L using equivalent transient radius

norms = {'D/L': colors.Normalize(vmin=7, vmax=13.5),
         'Dt/L': colors.Normalize(vmin=4, vmax=15),
         'pi2': colors.LogNorm(vmin=1e-5, vmax=1e-3),
         'U': colors.Normalize(vmin=10, vmax=40)}

cmaps = {'D/L': cm.viridis,
         'Dt/L': cm.magma,
         'pi2': cm.magma,
         'U': cm.magma}

cblabels = {'D/L': '$D_{f, eq, \perp}/L$',
            'Dt/L': '$D_{tr, eq, \perp}/L$',
            'pi2': '$\pi_2$',
            'U': 'Velocity [km/s]'}

titles = {'trans_drrad': 'Transient diameter, downrange',
          'trans_xrrad': 'Transient diameter, crossrange',
          'trans_eqrad': 'Transient diameter, equivalent',
          'trans_depth': 'Transient depth',
          'trans_volum': 'Transient volume',
          'trans_ellip': 'Transient ellipticity',
          'final_drrad': 'Final diameter, downrange',
          'final_xrrad': 'Final diameter, crossrange',
          'final_eqrad': 'Final diameter, equivalent',
          'final_depth': 'Final depth',
          'final_volum': 'Final volume',
          'final_ellip': 'Final ellipticity',
          'rimrad': 'Final rim-to-rim diameter'}

ylabels = {'trans_drrad': '$D_{tr, dr, \\theta}/D_{tr, dr, \perp}$',
           'trans_xrrad': '$D_{tr, xr, \\theta}/D_{tr, xr, \perp}$',
           'trans_eqrad': '$D_{tr, eq, \\theta}/D_{tr, eq, \perp}$',
           'trans_depth': '$d_{tr, \\theta}/d_{tr, \perp}$',
           'trans_volum': '$V_{tr, \\theta}/V_{tr, \perp}$',
           'trans_ellip': '$e_{tr, \\theta}/e_{tr, \perp}$',
           'final_drrad': '$D_{f, dr, \\theta}/D_{f, dr, \perp}$',
           'final_xrrad': '$D_{f, xr, \\theta}/D_{f, xr, \perp}$',
           'final_eqrad': '$D_{f, eq, \\theta}/D_{f, eq, \perp}$',
           'final_depth': '$d_{f, \\theta}/d_{f, \perp}$',
           'final_volum': '$V_{f, \\theta}/V_{f, \perp}$',
           'final_ellip': '$e_{f, \\theta}/e_{f, \perp}$',
           'D/L': '$D/L$',
           'Dt/L': '$D_t/L$',
           'rimrad': '$D_{f, rr, \\theta}/D_{f, rr, \perp}$'}

xlabels = {'ang': 'Angle [$^\circ$]', 
           'pi2': '$\pi_2 = g L / u^2$',
           'pi2a': '$pi_2^\star = g L / (u \sin\\theta)**2$',
           'trans_eqrad': 'Transient diameter'}

# Markers to use for each impactor size
#mdict = {'14': '.', '09': 'x', '06': '+'}
mdict = {'14': 'o', '09': 's', '06': 'D'}
vdict = {'05': '^', '10': '+', '15': 'x', '20': 'o', '25': 's', '30': 'D'}

# Pairs of velocities and impactor diameters for Earth-based models
earth_vLlist = list(zip((10., 15., 20., 30., 20., 25., 20., 30.),
                        (14., 14., 14., 14., 8.96, 8.96, 6.22, 6.22)))
# And for the Moon
moon_vLlist = list(zip(( 5., 15., 20., 30., 20., 25., 20., 30.),
                        (14., 14., 14., 14., 8.96, 8.96, 6.22, 6.22)))

def axplotter(ax, sdata, xfield, yfield, marker, label, cscheme, jitter=False, ms=5, mfc='None'):
    '''
    Plot a suite of models on an Axes.
    
    Parameters
    ----------
    ax : Axes
        A matplotlib Axes instance on which to plot the data
        
    sdata : DataFrame
        A pandas data frame with the data from the suite to be plotted
        
    xfield : string
        The column name to plot on the x axis
        
    yfield : string
        The column name to plot on the y axis
        
    marker : string
        The matplotlib marker type
        
    label : string
        The label to put in the legend for this suite
        
    cscheme : string
        The colour scheme to use on this plot. Should be one of "D/L", "pi2" or "U"
        
    Returns
    -------
    None
    '''

    # Normalise by the normal incidence impact in this suite
    a90 = sdata['ang'] == 90.
    
    if yfield == 'Dt/L':
        ynorm = sdata[yfield][sdata['ang']>=45]
        xx = sdata[xfield][sdata['ang']>=45].to_numpy().astype(float)
    elif yfield == 'D/L':
        ynorm = sdata[yfield]
        xx = (sdata[xfield] * np.sin(np.radians(sdata['ang']))).to_numpy().astype(float)
    elif xfield == 'ang':
        ynorm = sdata[yfield] / sdata[yfield][a90].iloc[0]
        xx = sdata[xfield].to_numpy().astype(float)
    else:
        ynorm = sdata[yfield]
        xx = sdata[xfield].to_numpy().astype(float)

    if cscheme in ['D/L', 'Dt/L', 'pi2', 'U']:
        # Colour the points according to the chosen scheme and 
        # for the normal incidence impact
        cnorm = sdata[cscheme][a90].iloc[0]
  
        # Grab the appropriate colour map and norm instances
        cmap = cmaps[cscheme]
        norm = norms[cscheme]
    
        # Convert to a colour value
        colour = cmap(norm(cnorm))
        
    elif colors.is_color_like(cscheme):
        
        colour = cscheme
        
    else:
        
        raise Exception
        
    if mfc == 'same':
        mfc = colour
    else:
        mfc = 'None'
    # Plot data on ax
    
    if jitter:
        xx += np.random.rand(len(xx))
    
    ax.plot(xx, ynorm, linestyle='None', alpha=0.8,
            marker=marker, mec=colour, label=label, markersize=ms, mfc=mfc)

    return xx, ynorm


def makeFigureBase(nrows=3):

    fontsize = 8
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize

    mmtoinch = 1. / 25.4
    
    fw = 95 * mmtoinch
    fh = 115 * mmtoinch

    if nrows > 3:
        fh *= nrows / 3

    # Create the Figure and Axes
    fig, axes = plt.subplots(nrows=nrows, figsize=(fw, fh))
    
    if nrows >= 3:
        fig.subplots_adjust(right=0.96, hspace=0.05, top=0.98, bottom=0.10, left=0.155)
    else:
        fig.subplots_adjust(right=0.96, hspace=0.05, top=0.98, bottom=0.10, left=0.155)


    abc = 'abcdefghijklmopqrstuvwxyz'

    for ax, letter in zip(axes, abc[:nrows]):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

        ax.text(-0.16, 1, '{})'.format(letter), ha='left', va='top',transform=ax.transAxes)


    return fig, axes

def addLines(ax, yfield):

    ax.set_ylabel(ylabels[yfield])
    
    if yfield[-3:] == 'rad' or yfield[-5:] == 'depth':
        xr = np.arange(0, 90, 1)
        if yfield[:5] == 'trans':
            l1, = ax.plot(xr, np.sin(np.radians(xr))**0.33, 'k--', label='$\sin^{0.33}\\theta$, GW78')
            l2, = ax.plot(xr, np.sin(np.radians(xr))**0.44, 'k:', label='$\sin^{0.44}\\theta$, CM86')
        else:
            l1, = ax.plot(xr, np.sin(np.radians(xr))**0.38, 'k-.', label='$\sin^{0.38}\\theta$, J16')
            l2 = None
        if yfield[-3:] == 'rad': 
            ax.set_ylim(0.65, 1.05)
        if yfield[-5:] == 'depth':
            ax.set_ylim(0.45, 1.05)

    if yfield[-5:] == 'volum':
        xr = np.arange(0, 90, 1)
        l1, = ax.plot(xr, np.sin(np.radians(xr))**1, 'k--', label='$\sin\\theta$, GW78')
        l2, = ax.plot(xr, np.sin(np.radians(xr))**1.3, 'k:', label='$\sin^{1.3}\\theta$, CM86')

        ax.set_ylim(0.35, 1.05)

    ax.axhline(1, alpha=0.5, color='#606060')

    return [l1, l2]

def addLegend(ax, legtype='L', markers=None, cschemes=None, labels=None):

    if legtype == 'L':

        L14 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=mdict['14'],
                            markersize=6, linestyle='None', label='L=14km')
        L09 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=mdict['09'],
                            markersize=6, linestyle='None', label='L=9km')
        L06 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=mdict['06'],
                            markersize=6, linestyle='None', label='L=6km')
    
        p1 = mlines.Line2D([], [], color=cschemes[0], mec='None', marker='s',
                           markersize=6, linestyle='None', label=labels[0])
        p2 = mlines.Line2D([], [], color=cschemes[1], mec='None', marker='s',
                           markersize=6, linestyle='None', label=labels[1])
    
        ax.legend(handles=[p1, p2, L06, L09, L14], handletextpad=0.5)

    if legtype == 'v':

        vlines = []

        for vv, mm in vdict.items():

            vlines.append(mlines.Line2D([], [], mec='#888888', mfc='None',
                                        marker=vdict[vv], markersize=5,
                                        linestyle='None',
                                        label='$u = {}$ km s$^{{-1}}$'.format(int(vv))))
    
        vlines.append(mlines.Line2D([], [], color=cschemes[0], mec='None', marker='s',
                                    markersize=6, linestyle='None', label=labels[0]))
        vlines.append(mlines.Line2D([], [], color=cschemes[1], mec='None', marker='s',
                                    markersize=6, linestyle='None', label=labels[1]))
    
        ax.legend(handles=vlines, handletextpad=0.5, fontsize=6, ncol=3)

    if legtype == 'rad_measure':

        h, _ = ax.get_legend_handles_labels()

        l = ['Equivalent diameter', 'Downrange diameter', 'Crossrange diameter']

        ax.legend(handles=h, labels=l, handletextpad=0.5)

    if legtype == 'diam':
        
        h, _ = ax.get_legend_handles_labels()

        l = ['$L = 14$ km', '$L = 8.96$ km', '$L = 6.22$ km']

        ax.legend(handles=h, labels=l, handletextpad=0.5)

    if legtype == 'vel_E':
        
        h, _ = ax.get_legend_handles_labels()

        l = ['$u = 15$ km/s', '$u = 20$ km/s', '$u = 30$ km/s']

        ax.legend(handles=h, labels=l, handletextpad=0.5)
    
    if legtype == 'vel_M':
        
        h, _ = ax.get_legend_handles_labels()

        l = ['$u = 5$ km/s', '$u = 15$ km/s', '$u = 20$ km/s', '$u = 30$ km/s']

        ax.legend(handles=h, labels=l, handletextpad=0.5)
    
    if legtype == 'all_E':

        L14 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=markers[0],
                            markersize=3.5, linestyle='None', label='$L = 14$ km')
        L09 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=markers[1],
                            markersize=3.5, linestyle='None', label='$L = 9$ km')
        L06 = mlines.Line2D([], [], mec='#888888', mfc='None', marker=markers[2],
                            markersize=3.5, linestyle='None', label='$L = 6$ km')

        cmap = cmaps[cschemes]
        norm = norms[cschemes]

        v15 = mlines.Line2D([], [], color=cmap(norm(15)), mec='None', marker='s',
                            markersize=3.5, linestyle='None', label='$u = 15$ km/s') 
        v20 = mlines.Line2D([], [], color=cmap(norm(20)), mec='None', marker='s',
                            markersize=3.5, linestyle='None', label='$u = 20$ km/s') 
        v25 = mlines.Line2D([], [], color=cmap(norm(25)), mec='None', marker='s',
                            markersize=3.5, linestyle='None', label='$u = 25$ km/s') 
        v30 = mlines.Line2D([], [], color=cmap(norm(30)), mec='None', marker='s',
                            markersize=3.5, linestyle='None', label='$u = 30$ km/s') 

        ax.legend(handles=[v15, v20, v25, v30, L14, L09, L06], ncol=2, handletextpad=0.2, columnspacing=0.7)



def addColorbar():

    sm = plt.cm.ScalarMappable(cmap=cmaps[cscheme], norm=norms[cscheme])
    sm.set_array([])

    cax = fig.add_axes([0.15, 0.92, 0.77, 0.01])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cax.xaxis.set_label_position('top')

    cb.set_label(cblabels[cscheme])

    return cax, cb

def makePaperFig(data, yfields, vLlists, cschemes, labels,
                 xfield='ang', legend=None, markers='o', markersize=4,
                 extrayfields=None, extraymarkers=None, extracs=None,
                 lineleg=False, fitfinal=False, fittrans=False):
    '''
    A function to make a matplotlib figure for the paper
    
    Parameters
    ----------
    data : DataFrame
        A pandas dataframe with the crater statistics data to be plotted
    
    yfield : string
        The column name to be plotted on the y axis
    
    xfield : string
        The column name to be plotted on the x axis (default='ang')

    cscheme : string
        The colour scheme to use on this plot. Should be one of "D/L", "pi2" or "U"
        
    Returns
    -------
    fig : Figure
        A matplotlib Figure instance
    '''

    fig, axes = makeFigureBase(len(yfields))

    if type(xfield) == str:
        xfields = [xfield] * len(yfields)

    else:
        xfields = xfield

    for df, label, vLlist, cscheme, marker in zip(data, labels, vLlists, cschemes, markers):

        for (v, L) in vLlist:

            if marker == mdict:

                thismarker = mdict['{:02.0f}'.format(L)]

            elif marker == vdict:

                thismarker = vdict['{:02.0f}'.format(v)]

            else:

                thismarker = marker

            # Grab just the data related to this suite of models
            suite = df[(df['U'] == v) & (df['L'] == L)]

            if fitfinal:
                xdat = np.array([])
                ydat = np.array([])

            if fittrans:
                xdat0 = np.array([])
                ydat0 = np.array([])

            for ax, yfield, xfield in zip(fig.axes, yfields, xfields):
                # Plot the suite on the Axes
                if (suite[suite['ang']==90][yfield] > 0).bool():
                    xx, yy = axplotter(
                              ax=ax, sdata=suite,
                              xfield=xfield, yfield=yfield,
                              marker=thismarker, ms=markersize, #mdict['{:02.0f}'.format(L)],
                              label='v={}, L={}'.format(v, L),
                              cscheme=cscheme, jitter=False
                              )
                if fitfinal and yfield == 'rimrad':
                    xdat = np.concatenate((xdat, xx))
                    ydat = np.concatenate((ydat, yy))
                if fittrans and yfield == 'trans_eqrad':
                    xdat0 = np.concatenate((xdat0, xx))
                    ydat0 = np.concatenate((ydat0, yy))
            
            if extrayfields is not None:
                for yf, ym, cs in zip(extrayfields, extraymarkers, extracs):
                    
                    xx, yy = axplotter(axes[0], sdata=suite, 
                                       xfield=xfields[0], yfield=yf,
                                       marker=ym, ms=markersize, 
                                       label='a', cscheme=cs)

    if len(set(xfields)) > 1:

        for ax, xf in zip(axes, xfields):

            ax.set_xlabel(xlabels[xfield])

    elif xfields[0] == 'ang' and len(set(xfields)) == 1:
        axes[-1].set_xlabel('Angle [$^\circ$]')
        
        for ax in axes:
            ax.set_xlim(25, 90)

    else:
        axes[-1].set_xlabel(xfield)

    for ax, xf in zip(axes, xfields):
        if xf[:3] == 'pi2':
            ax.set_xscale('log')
            ax.set_yscale('log')

    # Add some lines
    for ax, yfield in zip(fig.axes, yfields):

        if xfield == 'ang':
            if yfield not in ['D/L', 'Dt/L']:
                lines = addLines(ax, yfield)
                if fitfinal and yfield == 'rimrad':
                    ff = lambda x, a: np.sin(np.radians(x))**a
            
                    sigma = np.ones_like(xdat)
                    p0 = np.array([0.38])
                    popt, pcov = opti.curve_fit(ff, xdat, ydat, p0, sigma) 
            
                    print("FNAL POPT=", popt)

                    print("R^2=", rsquared(xdat, ydat, popt, ff))
                    xr = np.arange(0, 90, 1)
                    l3, = ax.plot(xr, ff(xr, popt), '-', alpha=0.7, color='#404040',
                                  label='$\sin^{{{:0.2f}}}\\theta$; This work'.format(popt[0]))
                    lines.append(l3)
                if fittrans and yfield == 'trans_eqrad':
                    ff = lambda x, a: np.sin(np.radians(x))**a
            
                    sigma = np.ones_like(xdat0)
                    p0 = np.array([0.33])
                    popt, pcov = opti.curve_fit(ff, xdat0, ydat0, p0, sigma) 
            
                    print("TRNS POPT=", popt)

                    print("R^2=", rsquared(xdat0, ydat0, popt, ff))
                    xr = np.arange(0, 90, 1)
                    l4, = ax.plot(xr, ff(xr, popt), '-', alpha=0.7, color='#404040',
                                  label='$\sin^{{{:0.2f}}}\\theta$; This work'.format(popt[0]))
                    lines.append(l4)
                if lineleg:
                    leg = ax.legend(handles=[l for l in lines if l is not None],
                                    fontsize=6, loc=5, handlelength=2.4)
                    ax.add_artist(leg)

    # Legend
    if legend == 'all_E':

        addLegend(axes[0], legend, markers, cschemes[0])

    elif legend == 'v4':

        addLegend(axes[-1], legend[0], cschemes=cschemes, labels=labels)

    elif legend is not None:

        addLegend(axes[0], legend, cschemes=cschemes, labels=labels)

    else:

        pass

    # Tick labels
    for ax in axes[:-1]:  #(ax1, ax2):
        ax.set_xticklabels([])
    
    return fig

def rsquared(xdata,ydata,fitconst,ff):
    f=ff(xdata,*fitconst)
    yminusf2=(ydata-f)**2
    sserr=sum(yminusf2)
    mean=float(sum(ydata))/float(len(ydata))
    yminusmean2=(ydata-mean)**2
    sstot=sum(yminusmean2)
    return 1.-(sserr/sstot)


paperFig = makePaperFig(data=(dfEarth, dfMoon), 
                         vLlists=(earth_vLlist, moon_vLlist),
                         cschemes=('navy', 'skyblue'),
                         labels=('Earth', 'Moon'), 
                         markers=(vdict, vdict), markersize=3.5,
                         xfield='ang', yfields=['trans_eqrad', 'trans_depth', 'trans_volum', 'rimrad'],
                         legend='v', lineleg=True,
                         fitfinal=True)

paperFig.savefig('figure2.pdf')
paperFig.savefig('figure2.png', dpi=300)
