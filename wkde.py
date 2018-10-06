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
import scipy


def univariate_weighted_kde(X, w, d, grid, bw):
    """Compute 1d weighted KDE along the specified dimension.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.
    d : int
        The dimension to operate along.
    grid : 1d array, (num_pts,)
        The grid to evaluate the KDE on.
    bw : float
        The kernel bandwidth to use.

    Returns
    -------
    kde : 1d array, (num_pts,)
        The 1d weighted KDE.
    """
    return 1.0 / (scipy.sqrt(2.0 * scipy.pi) * bw * scipy.sum(w)) * scipy.sum(
        w[None, :] * scipy.exp(
            -0.5 * (X[:, d] - grid[:, None]) ** 2 / bw ** 2
        ),
        axis=1
    )


def bivariate_weighted_kde(X, w, d0, d1, grid0, grid1, bw0, bw1):
    """Compute 1d weighted KDE along the specified dimension.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.
    d0 : int
        The first dimension to operate along.
    d1 : int
        The second dimension to operate along.
    grid0 : 1d array, (num_pts_0,)
        The grid for the first dimension to evaluate the KDE on.
    grid1 : 1d array, (num_pts_1,)
    bw0 : float
        The kernel bandwidth to use for the first dimension.
    bw1 : float
        The kernel bandwidth to use for the second dimension.

    Returns
    -------
    kde : 1d array, (num_pts_0, num_pts_1)
        The 1d weighted KDE.
    """
    return 1.0 / (2.0 * scipy.pi * bw0 * bw1 * scipy.sum(w)) * scipy.sum(
        w[None, None, :] * scipy.exp(
            -0.5 * (
                (
                    X[:, d0][None, None, :] - grid0[:, None, None]
                ) ** 2 / bw0 ** 2
                + (
                    X[:, d1][None, None, :] - grid1[None, :, None]
                ) ** 2 / bw1 ** 2
            )
        ),
        axis=2
    )


def silverman_bw(X, w, d):
    """Compute the bandwidths for KDEs along each dimension.

    Uses Silverman's rule.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.
    d : int
        Number of dimensions the KDE uses.

    Returns
    -------
    bw : 1d array, (num_dim,)
        The bandwidth for each dimension.
    """
    sigmas = scipy.sqrt(varw(X, w))
    n_eff = scipy.sum(w) ** 2 / scipy.sum(w ** 2)
    return (
        (4.0 / (d + 2.0)) ** (1 / (d + 4.0))
        * n_eff ** (-1.0 / (d + 4.0)) * sigmas
    )


def scott_bw(X, w, d):
    """Compute the bandwidths for KDEs along each dimension.

    Uses Scott's rule.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.
    d : int
        Number of dimensions the KDE uses.

    Returns
    -------
    bw : 1d array, (num_dim,)
        The bandwidth for each dimension.
    """
    sigmas = scipy.sqrt(varw(X, w))
    n_eff = scipy.sum(w) ** 2 / scipy.sum(w ** 2)
    return n_eff ** (-1.0 / (d + 4.0)) * sigmas


def meanw(X, w):
    """Find the weighted mean along each dimension.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.

    Returns
    -------
    meanw : 1d array, (num_dim,)
        The weighted mean.
    """
    X = scipy.asarray(X, dtype=float)
    w = scipy.asarray(w, dtype=float)
    return scipy.squeeze(scipy.sum(w[:, None] * X, axis=0)) / scipy.sum(w)


def varw(X, w):
    """Find the unbiased, weighted variance along each dimension.

    Parameters
    ----------
    X : 2d array, (num_samp, num_dim)
        Data.
    w : 1d array, (num_samp,)
        Weights.

    Returns
    -------
    varw : 1d array, (num_dim,)
        The weighted variance.
    """
    X = scipy.asarray(X, dtype=float)
    w = scipy.asarray(w, dtype=float)
    mu = meanw(X, w)
    V1 = scipy.sum(w)
    V2 = scipy.sum(w ** 2)
    return scipy.squeeze(meanw((X - mu) ** 2, w) / (V1 - V2 / V1))
