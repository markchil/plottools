# Copyright 2018 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose
# License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.gridspec import GridSpec
import scipy
import scipy.stats
from scipy.ndimage.filters import gaussian_filter
from itertools import cycle
import wkde


def get_color_10_cycle():
    """Returns the "color10" cycle, in string form.
    """
    return cycle(['C{:d}'.format(i) for i in range(10)])


def register_all_w_color(colors, **kwargs):
    for c in colors:
        register_wc_colormap(c, **kwargs)


def register_all_w_color_10(**kwargs):
    """Register white-color colormaps for each of the "color10" colors.
    """
    register_all_w_color(['C{:d}'.format(i) for i in range(10)], **kwargs)


def register_wck_colormap(color, name=None):
    """Create and register a colormap which fades linearly from white to a
    given color to black. It also defines the reverse version, with '_r'
    appended to the name.

    Parmameters
    -----------
    color : Matplotlib color descriptor (str, tuple, etc.)
        The color to use in the middle of the colormap.
    name : str, optional
        The name to register the colormap under. If `color` is a Matplotlib
        named color string, this defaults to 'w_NAME_k'. Otherwise, this
        defaults to 'custom'.

    Returns
    -------
    cm, cm_r : LinearSegmentedColormap
        The actual colormap object, and the reversed version.
    """
    if name is None:
        if isinstance(color, str):
            name = 'w_' + color + '_k'
        else:
            name = 'custom'
    color = to_rgba(color)
    cdict = {
        'red': [
            (0.0, 1.0, 1.0),
            (0.5, color[0], color[0]),
            (1.0, 0.0, 0.0)
        ],
        'green': [
            (0.0, 1.0, 1.0),
            (0.5, color[1], color[1]),
            (1.0, 0.0, 0.0)
        ],
        'blue': [
            (0.0, 1.0, 1.0),
            (0.5, color[2], color[2]),
            (1.0, 0.0, 0.0)
        ]
    }
    cm = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=cm)
    cdict_r = {
        'red': [
            (0.0, 0.0, 0.0),
            (0.5, color[0], color[0]),
            (1.0, 1.0, 1.0)
        ],
        'green': [
            (0.0, 0.0, 0.0),
            (0.5, color[1], color[1]),
            (1.0, 1.0, 1.0)
        ],
        'blue': [
            (0.0, 0.0, 0.0),
            (0.5, color[2], color[2]),
            (1.0, 1.0, 1.0)
        ]
    }
    cm_r = LinearSegmentedColormap(name + '_r', cdict_r)
    plt.register_cmap(cmap=cm_r)
    return cm, cm_r


def register_wc_colormap(color, name=None, alpha=False):
    """Create and register a colormap which fades linearly from white to a
    given color. It also defines the reverse version, with '_r'
    appended to the name.

    Parmameters
    -----------
    color : Matplotlib color descriptor (str, tuple, etc.)
        The color to use at the end of the colormap.
    name : str, optional
        The name to register the colormap under. If `color` is a Matplotlib
        named color string, this defaults to 'w_NAME'. Otherwise, this
        defaults to 'custom'.
    alpha : bool, optional
        If True, have alpha fade from 0 to 1. Default is False.

    Returns
    -------
    cm, cm_r : LinearSegmentedColormap
        The actual colormap object, and the reversed version.
    """
    if name is None:
        if isinstance(color, str):
            name = 'w_' + color
        else:
            name = 'custom'
        if alpha:
            name += '_a'
    color = to_rgba(color)
    cdict = {
        'red': [
            (0.0, 1.0, 1.0),
            (1.0, color[0], color[0]),
        ],
        'green': [
            (0.0, 1.0, 1.0),
            (1.0, color[1], color[1]),
        ],
        'blue': [
            (0.0, 1.0, 1.0),
            (1.0, color[2], color[2]),
        ]
    }
    if alpha:
        cdict['alpha'] = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0)
        ]
    cm = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=cm)
    cdict_r = {
        'red': [
            (0.0, color[0], color[0]),
            (1.0, 1.0, 1.0)
        ],
        'green': [
            (0.0, color[1], color[1]),
            (1.0, 1.0, 1.0)
        ],
        'blue': [
            (0.0, color[2], color[2]),
            (1.0, 1.0, 1.0)
        ]
    }
    if alpha:
        cdict_r['alpha'] = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0)
        ]
    cm_r = LinearSegmentedColormap(name + '_r', cdict_r)
    plt.register_cmap(cmap=cm_r)
    return cm, cm_r


def register_ck_colormap(color, name=None):
    """Create and register a colormap which fades linearly from a given color
    to black. It also defines the reverse version, with '_r'
    appended to the name.

    Parmameters
    -----------
    color : Matplotlib color descriptor (str, tuple, etc.)
        The color to use at the end of the colormap.
    name : str, optional
        The name to register the colormap under. If `color` is a Matplotlib
        named color string, this defaults to 'NAME_k'. Otherwise, this
        defaults to 'custom'.

    Returns
    -------
    cm, cm_r : LinearSegmentedColormap
        The actual colormap object, and the reversed version.
    """
    if name is None:
        if isinstance(color, str):
            name = color + '_k'
        else:
            name = 'custom'
    color = to_rgba(color)
    cdict = {
        'red': [
            (0.0, color[0], color[0]),
            (1.0, 0.0, 0.0)
        ],
        'green': [
            (0.0, color[1], color[1]),
            (1.0, 0.0, 0.0)
        ],
        'blue': [
            (0.0, color[2], color[2]),
            (1.0, 0.0, 0.0)
        ]
    }
    cm = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=cm)
    cdict_r = {
        'red': [
            (0.0, 0.0, 0.0),
            (1.0, color[0], color[0]),
        ],
        'green': [
            (0.0, 0.0, 0.0),
            (1.0, color[1], color[1]),
        ],
        'blue': [
            (0.0, 0.0, 0.0),
            (1.0, color[2], color[2]),
        ]
    }
    cm_r = LinearSegmentedColormap(name + '_r', cdict_r)
    plt.register_cmap(cmap=cm_r)
    return cm, cm_r


def hist2d_contour(
    a, x, y, w=None, plot_heatmap=False, plot_levels_filled=False,
    plot_levels=True, plot_points=False, filter_contour=True,
    filter_heatmap=False, hist_kwargs={}, pcolor_kwargs={}, contour_kwargs={},
    scatter_kwargs={}, filter_kwargs={}, filter_sigma=1.0, plot_ci=None,
    ci_kwargs={}, scatter_fraction=1.0,
    use_kde=False, kde_bw0=None, kde_bw1=None
):
    """Make combined 2d histogram, contour and/or scatter plot.

    Parameters
    ----------
    a : axes
        The axes to plot on.
    x : 1d array
        The x data.
    y : 1d array
        The y data.
    w : 1d array, optional
        The weights
    plot_heatmap : bool, optional
        If True, plot the heatmap of the histogram. Default is False.
    plot_levels_filled : bool, optional
        If True, plot the filled contours of the histogram. Default is False.
    plot_levels : bool, optional
        If True, plot the contours of the histogram. Default is True.
    plot_points : bool, optional
        If True, make a scatterplot of the points. Default is False.
    filter_contour : bool, optional
        If True, filter the histogram before plotting contours. Default is
        True.
    filter_heatmap : bool, optional
        If True, filter the histogram before plotting heatmap. Default is
        False.
    hist_kwargs : dict, optional
        Keyword arguments for scipy.histogram2d.
    pcolor_kwargs : dict, optional
        Keyword arguments for pcolormesh when plotting heatmap.
    contour_kwargs : dict, optional
        Keyword arguments for contour and contourf when plotting contours. To
        specify the number of contours, use the key 'N'. To use specific
        contour levels, use the key 'V'.
    scatter_kwargs : dict, optional
        Keyword arguments for scatterplot when plotting points.
    filter_kwargs : dict, optional
        Keyword arguments for filtering of histogram.
    filter_sigma : float, optional
        The standard deviation for the Gaussian filter used to smooth the
        histogram. Default is 2 bins.
    plot_ci : float or 1d array, optional
        If this is a float, the contour containing this much probability mass
        is drawn. Default is None (don't draw contour).
    ci_kwargs : dict, optional
        Keyword arguments for drawing the confidence interval.
    scatter_fraction : float, optional
        Fraction of points to include in the scatterplot. Default is 1.0 (use
        all points).
    use_kde : bool, optional
        If True, use a kernel density estimator (KDE) in place of bivariate
        histogram when `plot_heatmap=True`. Default is False.
    kde_bw0 : float, optional
        KDE bandwidth for the zeroth dimension.
    kde_bw1 : float, optional
        KDE bandwidth for the first dimension.
    """
    if 'bins' not in hist_kwargs:
        hist_kwargs['bins'] = (100, 101)
    if 'normed' not in hist_kwargs:
        hist_kwargs['normed'] = True
    # Only compute histogram if needed:
    if (
        plot_heatmap or plot_levels or plot_levels_filled
        or (plot_ci is not None)
    ):
        if use_kde:
            xgrid = scipy.linspace(x.min(), x.max(), hist_kwargs['bins'][0])
            ygrid = scipy.linspace(y.min(), y.max(), hist_kwargs['bins'][1])
            H = wkde.bivariate_weighted_kde(
                x,
                y,
                w if w is not None else scipy.ones(len(x)),
                xgrid,
                ygrid,
                kde_bw0,
                kde_bw1
            )
            # Match pcolormesh's silly format:
            dx = xgrid[1] - xgrid[0]
            xedges = (xgrid[1:] + xgrid[:-1]) / 2.0
            xedges = scipy.concatenate(
                ([xedges[0] - dx], xedges, [xedges[-1] + dx])
            )
            dy = ygrid[1] - ygrid[0]
            yedges = (ygrid[1:] + ygrid[:-1]) / 2.0
            yedges = scipy.concatenate(
                ([yedges[0] - dy], yedges, [yedges[-1] + dy])
            )
            # Doesn't make sense to filter KDE...
            Hf = H
        else:
            H, xedges, yedges = scipy.histogram2d(
                x, y, weights=w, **hist_kwargs
            )
            if filter_contour or filter_heatmap:
                Hf = gaussian_filter(H, filter_sigma, **filter_kwargs)
    if plot_heatmap:
        XX, YY = scipy.meshgrid(xedges, yedges)
        a.pcolormesh(XX, YY, Hf.T if filter_heatmap else H.T, **pcolor_kwargs)
    if plot_levels or plot_levels_filled or (plot_ci is not None):
        # Convert to bin centers:
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        XX, YY = scipy.meshgrid(xcenters, ycenters)
        args = []
        if 'V' in contour_kwargs:
            args += [scipy.atleast_1d(contour_kwargs.pop('V')), ]
        elif 'N' in contour_kwargs:
            args += [contour_kwargs.pop('N'), ]
        if plot_levels_filled:
            a.contourf(
                XX, YY, Hf.T if filter_contour else H.T,
                *args, **contour_kwargs
            )
        if plot_levels:
            a.contour(
                XX, YY, Hf.T if filter_contour else H.T,
                *args, **contour_kwargs
            )
        if plot_ci is not None:
            V = prob_contour(H, xedges, yedges, p=plot_ci)
            if 'vmin' not in ci_kwargs:
                ci_kwargs['vmin'] = 0.0
            if 'vmax' not in ci_kwargs:
                ci_kwargs['vmax'] = H.max()
            a.contour(
                XX,
                YY,
                Hf.T if filter_contour else H.T,
                scipy.unique(scipy.atleast_1d(V)),
                **ci_kwargs
            )
    if plot_points:
        # Make the markersize not ridiculous by default:
        if 'ms' not in scatter_kwargs and 'markersize' not in scatter_kwargs:
            scatter_kwargs['ms'] = 0.1
        # Make points transparent by default (approximates heatmap...):
        if 'alpha' not in scatter_kwargs:
            scatter_kwargs['alpha'] = 0.5
        # Do not connect with lines by default:
        if 'ls' not in scatter_kwargs and 'linestyle' not in scatter_kwargs:
            scatter_kwargs['ls'] = ''
        # Plot with circles by default:
        if 'marker' not in scatter_kwargs:
            scatter_kwargs['marker'] = 'o'
        if scatter_fraction != 1.0:
            N = int(round(scatter_fraction * len(x)))
            indices = scipy.random.choice(
                range(len(x)),
                size=N,
                replace=False
            )
        else:
            indices = range(len(x))
        a.plot(x[indices], y[indices], **scatter_kwargs)


def prob_contour(H, xedges, yedges, p=0.95):
    """Compute PDF value enclosing desired probability mass.

    The contour corresponding to the returned PDF value will contain
    (approximately) p integrated probability mass.

    Parameters
    ----------
    H : 2d array, (n_x, n_y)
        Normalized (as PDF) histogram.
    xedges : 1d array, (n_x + 1,)
        X edges of histogram bins.
    yedges: 1d array, (n_y + 1,)
        Y edges of histogram bins.
    p : float, optional
        Probability to find contour of. Default is 0.95
    """
    # Plan: Find highest value, add. Repeat until target probability reached,
    # return value of H at last point added. This should be the contour which
    # encloses the desired fraction of probability mass.
    dx = scipy.atleast_2d(scipy.diff(xedges)).T
    dy = scipy.atleast_2d(scipy.diff(yedges))
    PM = (H * dx * dy).ravel()
    H = H.ravel()
    # Sort into order of decreasing probability mass:
    srtidx = PM.argsort()[::-1]
    # Find cumulative sum:
    PM_sum = PM[srtidx].cumsum()
    # Find first point where PM_sum >= p:
    mask = PM_sum >= scipy.atleast_2d(p).T
    out = scipy.zeros(mask.shape[0])
    for i in range(mask.shape[0]):
        idx, = scipy.nonzero(mask[i, :])
        out[i] = H[srtidx[idx[0]]]
    return out


def grouped_plot_matrix(
    X, Y=None, w=None, feature_labels=None, class_labels=None,
    show_legend=True, colors=None, fixed_height=None, fixed_width=None, l=0.1,
    r=0.9, t=0.9, b=0.1, ax_space=0.1, rotate_last_hist=None, hist1d_kwargs={},
    cmap=None, legend_kwargs={}, autocolor=True,

    use_kde=False, kde_bws='scott', cutoff_w=None, kde1d_kwargs={},

    **kwargs
):
    """Plot the results of MCMC sampler (posterior and chains).

    Loosely based on triangle.py. Provides extensive options to format the
    plot.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Features.
    Y : 1d array, (num_samp,), optional
        Labels. Default is to treat all data as being from same class.
    w : 1d array, (num_samp,), optional
        Weights of the samples. Default is to treat all samples as
        equally-weighted.
    feature_labels : list of str, optional
        The labels to use for each of the features.
    class_labels : list of str, optional
        The labels to use for each class.
    show_legend : bool, optional
        If True, draw a legend. A legend is never drawn for single-class data.
        Default is True.
    colors : list of color strings, optional
        The colors to use for each class. Using strings (either name or hex)
        lets the function automatically pick the corresponding colormaps.
    fixed_height : float, optional
        The desired figure height (in inches). Default is to automatically
        adjust based on `fixed_width` to make the subplots square.
    fixed_width : float, optional
        The desired figure width (in inches). Default is `figure.figsize[0]`.
    l : float, optional
        The location (in relative figure units) of the left margin. Default is
        0.1.
    r : float, optional
        The location (in relative figure units) of the right margin. Default is
        0.9.
    t : float, optional
        The location (in relative figure units) of the top of the grid of
        histograms. Default is 0.9.
    b : float, optional
        The location (in relative figure units) of the bottom of the grid of
        histograms. Default is 0.1.
    ax_space : float, optional
        The `w_space` and `h_space` to use (in relative figure units). Default
        is 0.1.
    rotate_last_hist : bool, optional
        If True, rotate the bottom right histogram. Default is to only do this
        for bivariate data.
    hist1d_kwargs : dict, optional
        Extra keyword arguments for the 1d histograms.
    cmap : str, optional
        Colormap to use, overriding the default color cycle. Probably only
        useful for one-class data.
    autocolor : bool, optional
        If True, automatically assign colors and colormaps. Otherwise, just
        use what is in the optional keywords. Default is True.
    use_kde : bool, optional
        If True, use a kernel density estimator (KDE) for the univariate and
        bivariate (when `plot_heatmap=True`) histograms. Default is False.
    kde_bws : {'scott', 'silverman', array of float (num_dim,) or (2, num_dim)}
        Which rule to use to estimate KDE bandwidths, a bandwidth to use for
        all dimensions, or a list of bandwidths for each dimension. If an array
        of bandwidths is provided, it can either be 1d (same bandwidth for 1d
        and 2d KDEs), or 2d (row 0 is 1d bandwidth, row 1 is 2d bandwidth).
        Default is 'scott' (use Scott's rule of thumb). Note that the labels
        `Y` are ignored when computing the bandwidths, and the same bandwidth
        is used for each label. Therefore, you should compute the proper
        bandwidth externally when plotting KDEs for each label.
    cutoff_w : float, optional
        If `w` and `cutoff_w` are present, points with `w < cutoff_w * w.max()`
        will be excluded. Default is to plot all points.
    kde1d_kwargs : dict, optional
        Extra keyword arguments for the `plot` command when plotting the 1d
        KDE.
    kwargs : optional keywords
        All extra keyword arguments are passed to `hist2d_contour`.
    """
    # TODO: kwarg handling when using KDE is too convoluted!
    # TODO: add proper handling of KDE bandwidth for multiclass data!

    # Mask out points with low weights:
    if w is not None and cutoff_w is not None:
        mask = w >= (cutoff_w * w.max())
        X = X[mask, :]
        if Y is not None:
            Y = Y[mask]
        w = w[mask]

    # Compute kernel bandwidths:
    if use_kde:
        if isinstance(kde_bws, str):
            kde_bw_method = kde_bws
            kde_bws = scipy.zeros((2, X.shape[1]))
            # Handle both weighted and unweighted case:
            if w is None:
                wtmp = scipy.ones(X.shape[0])
            else:
                wtmp = w
            if kde_bw_method == 'scott':
                kde_bws[0, :] = wkde.scott_bw(X, wtmp, 1)
                kde_bws[1, :] = wkde.scott_bw(X, wtmp, 2)
            elif kde_bw_method == 'silverman':
                kde_bws[0, :] = wkde.silverman_bw(X, wtmp, 1)
                kde_bws[1, :] = wkde.silverman_bw(X, wtmp, 2)
            else:
                raise ValueError(
                    'Invalid KDE bandwidth estimation method "{:s}"!'.format(
                        kde_bw_method
                    )
                )
        else:
            kde_bws = scipy.asarray(kde_bws, dtype=float)
            if kde_bws.ndim == 0:
                kde_bws = kde_bws * scipy.ones((2, X.shape[1]))
            elif kde_bws.ndim == 1:
                kde_bws = scipy.tile(kde_bws, (2, 1))
            if kde_bws.shape != (2, X.shape[1]):
                raise ValueError('Invalid number of KDE bandwidths!')

    # Number of features:
    k = X.shape[1]

    # Unique class labels:
    if Y is None:
        Y = scipy.ones(X.shape[0])
        uY = [1.0, ]
    else:
        uY = scipy.unique(Y)

    # Default to plot heatmap for one-class data, contours for multi-class
    # data:
    if len(uY) == 1:
        if 'plot_heatmap' not in kwargs:
            kwargs['plot_heatmap'] = True
        if 'plot_levels_filled' not in kwargs:
            kwargs['plot_levels_filled'] = False
        if 'plot_levels' not in kwargs:
            kwargs['plot_levels'] = False
        if 'plot_points' not in kwargs:
            kwargs['plot_points'] = False
        if cmap is None:
            cmap = mpl.rcParams['image.cmap']
    else:
        if 'plot_heatmap' not in kwargs:
            kwargs['plot_heatmap'] = False
        if 'plot_levels_filled' not in kwargs:
            kwargs['plot_levels_filled'] = False
        if 'plot_levels' not in kwargs:
            kwargs['plot_levels'] = True
        if 'plot_points' not in kwargs:
            kwargs['plot_points'] = False
        if 'plot_ci' not in kwargs:
            kwargs['plot_ci'] = 0.95

    # Defaults for 1d histograms:
    if 'density' not in hist1d_kwargs:
        hist1d_kwargs['density'] = True
    if 'histtype' not in hist1d_kwargs:
        hist1d_kwargs['histtype'] = 'stepfilled'
    if 'bins' not in hist1d_kwargs:
        if w is None and not use_kde:
            hist1d_kwargs['bins'] = 'auto'
        else:
            # TODO: Come up with better default!
            hist1d_kwargs['bins'] = 50
    if 'alpha' not in hist1d_kwargs:
        hist1d_kwargs['alpha'] = 0.75 if len(uY) > 1 else 1

    # Set color order:
    if colors is None:
        cc = get_color_10_cycle()
        colors = [next(cc) for i in range(0, len(uY))]

    # Handle rotation of bottom right histogram:
    if rotate_last_hist is None:
        rotate_last_hist = k == 2

    # Default labels for features and classes:
    if feature_labels is None:
        feature_labels = ['{:d}'.format(kv) for kv in range(0, k)]
    if class_labels is None:
        class_labels = ['{:d}'.format(int(yv)) for yv in uY]

    # Set up geometry:
    if fixed_height is None:
        if fixed_width is None:
            # Default: use matplotlib's default width, handle remaining
            # parameters with the fixed width case below:
            fixed_width = mpl.rcParams['figure.figsize'][0]
        fixed_height = fixed_width * (r - l) / (t - b)
    elif fixed_width is None:
        # Only height specified, compute width to yield square histograms
        fixed_width = fixed_height * (t - b) / (r - l)
    # Otherwise width and height are fixed, and we may not have square
    # histograms, at the user's discretion.

    wspace = ax_space
    hspace = ax_space

    f = plt.figure(figsize=(fixed_width, fixed_height))
    gs = GridSpec(k, k)
    gs.update(bottom=b, top=t, left=l, right=r, wspace=wspace, hspace=hspace)
    axes = []
    # j is the row, i is the column.
    for j in range(k):
        row = []
        for i in range(k):
            if i > j:
                row.append(None)
            else:
                if rotate_last_hist and i == j and i == k - 1:
                    sharey = row[-1]
                else:
                    sharey = row[-1] if i > 0 and i < j and j < k else None
                sharex = axes[-1][i] if j > i and j < k else \
                    (row[-1] if i > 0 and j == k else None)
                row.append(
                    f.add_subplot(gs[j, i], sharey=sharey, sharex=sharex)
                )
        axes.append(row)
    axes = scipy.asarray(axes)

    # Update axes with the data:
    # j is the row, i is the column.
    for i in range(0, k):
        if rotate_last_hist and i == k - 1:
            orientation = 'horizontal'
        else:
            orientation = 'vertical'
        for ic, yv in enumerate(uY):
            mask = Y == yv
            if autocolor:
                hist1d_kwargs['color'] = colors[ic]
            if use_kde:
                kde_grid = scipy.linspace(
                    X[mask, i].min(),
                    X[mask, i].max(),
                    hist1d_kwargs['bins']
                )
                kde = wkde.univariate_weighted_kde(
                    X[mask, i],
                    w[mask] if w is not None else scipy.ones(mask.sum()),
                    kde_grid,
                    kde_bws[0, i]
                )
                if orientation == 'vertical':
                    axes[i, i].plot(
                        kde_grid,
                        kde,
                        color=hist1d_kwargs['color'],
                        **kde1d_kwargs
                    )
                    if hist1d_kwargs['alpha'] > 0:
                        axes[i, i].fill(
                            kde_grid,
                            kde,
                            color=hist1d_kwargs['color'],
                            alpha=hist1d_kwargs['alpha']
                        )
                    axes[i, i].set_ylim(bottom=0.0)
                else:
                    axes[i, i].plot(
                        kde,
                        kde_grid,
                        color=hist1d_kwargs['color'],
                        **kde1d_kwargs
                    )
                    if hist1d_kwargs['alpha'] > 0:
                        axes[i, i].fill(
                            kde,
                            kde_grid,
                            color=hist1d_kwargs['color'],
                            alpha=hist1d_kwargs['alpha']
                        )
                    axes[i, i].set_xlim(left=0.0)
            else:
                axes[i, i].hist(
                    X[mask, i],
                    weights=w[mask] if w is not None else None,
                    orientation=orientation,
                    **hist1d_kwargs
                )
        if i < k - 1 or (rotate_last_hist and i == k - 1):
            plt.setp(axes[i, i].get_xticklabels(), visible=False)
        else:
            axes[i, i].set_xlabel(feature_labels[i])
        plt.setp(axes[i, i].get_yticklabels(), visible=False)
        for j in range(i + 1, k):
            for ic, yv in enumerate(uY):
                # TODO: more control over coloring!
                mask = Y == yv
                c = colors[ic]
                cm = 'w_' + c + '_a' if cmap is None else cmap

                if autocolor:
                    pcolor_kwargs = kwargs.get('pcolor_kwargs', {})
                    pcolor_kwargs['cmap'] = cm
                    kwargs['pcolor_kwargs'] = pcolor_kwargs

                    contour_kwargs = kwargs.get('contour_kwargs', {})
                    if 'colors' in contour_kwargs:
                        contour_kwargs['colors'] = c
                    else:
                        contour_kwargs['cmap'] = cm
                    kwargs['contour_kwargs'] = contour_kwargs

                    scatter_kwargs = kwargs.get('scatter_kwargs', {})
                    scatter_kwargs['color'] = c
                    kwargs['scatter_kwargs'] = scatter_kwargs

                plot_ci = kwargs.get('plot_ci', None)
                if plot_ci is not None and autocolor:
                    ci_kwargs = kwargs.get('ci_kwargs', {})
                    plot_ci = scipy.atleast_1d(plot_ci)
                    if 'colors' in ci_kwargs or len(plot_ci) == 1:
                        ci_kwargs['colors'] = c
                    else:
                        ci_kwargs['cmap'] = cm
                    kwargs['ci_kwargs'] = ci_kwargs
                hist2d_contour(
                    axes[j, i],
                    X[mask, i],
                    X[mask, j],
                    w=w,
                    use_kde=use_kde,
                    kde_bw0=kde_bws[1, i] if use_kde else None,
                    kde_bw1=kde_bws[1, j] if use_kde else None,
                    **kwargs
                )
            if j < k - 1:
                plt.setp(axes[j, i].get_xticklabels(), visible=False)
            else:
                axes[j, i].set_xlabel(feature_labels[i])
            if i != 0:
                plt.setp(axes[j, i].get_yticklabels(), visible=False)
            else:
                axes[j, i].set_ylabel(feature_labels[j])
    # Draw legend:
    if show_legend and len(uY) > 1:
        handles = [
            Patch(color=clr, label=lbl, alpha=hist1d_kwargs['alpha'])
            for clr, lbl in zip(colors, class_labels)
        ]
        if 'loc' not in legend_kwargs:
            legend_kwargs['loc'] = 'upper right'
        if 'bbox_to_anchor' not in legend_kwargs:
            legend_kwargs['bbox_to_anchor'] = (r, t)
        if 'bbox_transform' not in legend_kwargs:
            legend_kwargs['bbox_transform'] = f.transFigure
        f.legend(
            handles,
            [h.get_label() for h in handles],
            **legend_kwargs
        )

    return f, axes


def add_points(
    a, points, Sigma=None, ci=0.95, colors=None, linestyles=None, markers=None,
    point_kwargs={}, line_kwargs={}
):
    """Add point(s) to axis array from `grouped_plot_matrix`.

    Parameters
    ----------
    a : 2d array of axis, (`num_dim`, `num_dim`)
        Axis to plot on.
    points : 1d or 2d array of float, (`num_pt`, `num_dim`)
        Points to plot.
    Sigma : 3d array of float, (`num_pt`, `num_dim`, `num_dim`), optional
        Covariance matrix associated with each point. To not draw for a given
        point, set the corresponding entry to `None`.
    ci : float, optional
        The confidence interval(s) to plot. Default is 0.95.
    colors : list of color specifications, optional
        Colors for each point. Default is to use matplotlib color cycle.
    linestyles : list of str, optional
        Line specifications for the vertical lines in the univariate
        histograms. Default is all solid.
    markers : list of str, optional
        Marker specifications for the points in the bivariate histograms.
        Default is all circle.
    point_kwargs : dict, optional
        Keyword arguments for plotting the points.
    line_kwargs : dict, optional
        Keyword arguments for plotting the lines and edges of the covariance
        ellipses.
    """
    # TODO: Better argument handling!
    # TODO: Add support for rotate_last_hist!
    points = scipy.atleast_2d(points)
    k = a.shape[1]
    np = points.shape[0]
    if colors is None:
        c = mpl.rcParams['axes.prop_cycle']()
        colors = [next(c)['color'] for i in range(np)]
    if linestyles is None:
        linestyles = ['-', ] * np
    if markers is None:
        markers = ['o', ] * np
    if Sigma is None:
        Sigma = [None, ] * np
    # j is the row, i is the column:
    for i in range(k):
        for ip, p in enumerate(points):
            a[i, i].axvline(
                p[i],
                color=colors[ip],
                ls=linestyles[ip],
                **line_kwargs
            )
            if Sigma[ip] is not None:
                xl = a[i, i].get_xlim()
                grid = scipy.linspace(xl[0], xl[1], int(1e3))
                a[i, i].plot(
                    grid,
                    scipy.stats.norm.pdf(
                        grid,
                        loc=p[i],
                        scale=scipy.sqrt(Sigma[ip][i, i])
                    ),
                    color=colors[ip],
                    ls=linestyles[ip],
                    **line_kwargs
                )

        for j in range(i + 1, k):
            for ip, p in enumerate(points):
                a[j, i].plot(
                    p[i],
                    p[j],
                    color=colors[ip],
                    marker=markers[ip],
                    ls='',
                    **point_kwargs
                )
                if Sigma[ip] is not None:
                    cov = scipy.asarray(
                        [[Sigma[ip][i, i], Sigma[ip][i, j]],
                         [Sigma[ip][j, i], Sigma[ip][j, j]]],
                        dtype=float
                    )
                    aa, bb, ang = compute_ellipse_params(cov, ci=ci)
                    plot_ellipse(
                        a[j, i],
                        p[i],
                        p[j],
                        aa,
                        bb,
                        ang,
                        edgecolor=colors[ip],
                        facecolor='none',
                        ls=linestyles[ip],
                        **line_kwargs
                    )


def compute_ellipse_params(Sigma, ci=0.95):
    """Compute the parameters of the confidence ellipse for the bivariate
    normal distribution with the given covariance matrix.

    Parameters
    ----------
    Sigma : 2d array, (2, 2)
        Covariance matrix of the bivariate normal.
    ci : float or 1d array, optional
        Confidence interval(s) to compute. Default is 0.95.

    Returns
    -------
    a : float or 1d array
        Major axes for each element in `ci`.
    b : float or 1d array
        Minor axes for each element in `ci`.
    ang : float
        Angle of ellipse, in radians.
    """
    ci = scipy.atleast_1d(ci)
    lam, v = scipy.linalg.eigh(Sigma)
    chi2 = [-scipy.log(1.0 - cival) * 2.0 for cival in ci]
    a = [2.0 * scipy.sqrt(chi2val * lam[-1]) for chi2val in chi2]
    b = [2.0 * scipy.sqrt(chi2val * lam[-2]) for chi2val in chi2]
    ang = scipy.arctan2(v[1, -1], v[0, -1])
    return a, b, ang


def plot_ellipse(ax, x, y, a, b, ang, **kwargs):
    """Plot ellipse(s) on the specified axis.

    Parameters
    ----------
    ax : axis
        Axis to plot on.
    x : float
        X coordinate of center. All ellipses have same center.
    y : float
        Y coordinate of center. All ellipses have same center.
    a : float or 1d array
        Major axis of ellipse(s).
    b : float or 1d array
        Minor axis of ellipse(s).
    ang : float
        Angle of ellipse(s), in radian. All ellipses have same angle.
    """
    for av, bv in zip(a, b):
        ell = Ellipse(
            [x, y],
            av,
            bv,
            angle=scipy.degrees(ang),
            **kwargs
        )
        ax.add_artist(ell)


# Map letters/numbers to Morse code:
CODE = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.'
}


def str2morse(
    s, dash=3.0, dot=1.0, tone_spacing=1.0, letter_spacing=3.0,
    word_spacing=7.0, end_with_space=None, scale=1.0
):
    """Make a matplotlib dash pattern which consists of the indicated string in
    Morse code.

    Pass the resulting object to the `dashes` kwarg of `plt.plot()`.

    You may want to play with the `handlelength` argument of `plt.legend()` to
    ensure the entire pattern shows up in the legend.

    Parameters
    ----------
    s : str
        The string to encode. This can contain letters, numbers, and spaces.
    dash : float, optional
        The length of the dash. For proper Morse code, this should be three
        times the length of the dot. Default is 3.
    dot : float, optional
        The length of the dot. For proper Morse code, this should be one third
        the length of the dash. Default is 1.
    tone_spacing : float, optional
        Scale factor for how long to make the space between individual
        dots/dashes. The dot length is multiplied by this. Default is 1 (for
        conventional Morse code).
    letter_spacing : float, optional
        Scale factor for how long to make the space between individual letters.
        The tone spacing is multiplied by this. Default is 3 (for conventional
        Morse code).
    word_spacing : float, optional
        Scale factor for how long to make spaces between words. The tone
        spacing is multiplied by this. Default is 7 (for conventional Morse
        code).
    end_with_space : bool, optional
        Set to True if you want to add a space at the end of the phrase. Set to
        False to never append a space. By default, a space is automatically
        appended if the phrase has multiple words (i.e., a space appears in the
        middle).
    scale : float, optional
        Factor to scale every length by. Default is 1.0.
    """
    # One dot space between tones.
    # Three dot spaces between letters.
    # Seven dot spaces between words.
    scale = float(scale)
    dash *= scale
    dot *= scale
    tone_spacing *= scale
    letter_spacing *= scale
    word_spacing *= scale

    space_map = {'.': dot, '-': dash}
    pattern = []
    for c in s:
        if c == ' ':
            pattern[-1] *= word_spacing / letter_spacing
            if end_with_space is None:
                end_with_space = True
        else:
            morse = CODE[c.upper()]
            for mc in morse:
                pattern += [space_map[mc], dot * tone_spacing]
            pattern[-1] *= letter_spacing
    if end_with_space and s[-1] != ' ':
        pattern[-1] *= word_spacing / letter_spacing
    return pattern


def dashdot(num_dot, dash_length=4.8, dot_length=0.8, space_length=1.2):
    """Create a matplotlib dash pattern consisting of a single dash and zero
    or more dots.

    Parameters
    ----------
    num_dot : non-negative int
        The number of dots to put after the dash.
    dash_length : float, optional
        The length of the initial dash. Default is 4.8 points, which is based
        on the default matplotlib 2.0 style.
    dot_length : float, optional
        The length of the dots. Default is 0.8 points, which is based on the
        default matplotlib 2.0 style.
    space_length : float, optional
        The length of the spaces. Default is 1.2, which is based on the
        default matplotlib 2.0 style.
    """
    return (
        [dash_length, space_length] + [dot_length, space_length] * int(num_dot)
    )


def make_pcolor_grid(grid):
    """Make grids with the correct spacing for pcolor/pcolormesh.

    Parameters
    ----------
    grid : 1d array of float
        The center coordinates of each point.
    """
    grid = scipy.asarray(grid, dtype=float)
    pgrid = (grid[1:] + grid[:-1]) / 2.0
    pgrid = scipy.concatenate((
        [grid[0] - (pgrid[0] - grid[0])],
        pgrid,
        [grid[-1] + (grid[-1] - pgrid[-1])]
    ))
    return pgrid


# TODO: Legend handler for contour plots, and ellipses.
# TODO: Turn off internal ticks if ticks are not on both sides.
# TODO: Function to turn off all ticks, spines, labels, etc.


register_all_w_color_10(alpha=True)
