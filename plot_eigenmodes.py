# %%
import numpy as np
from scipy.special import sph_harm
from matplotlib import pyplot as plt
import starry
starry.config.lazy = False

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)


def get_flux_maps(output_file, wave_idx=0, N_lat=181, N_lon=361):
    file = np.load(output_file, encoding='latin1', allow_pickle=True)
    harmonic_weights = np.asarray(file["arr_0"][()]["ecoeffList"])
    signed_weights = np.einsum("abc->acb", harmonic_weights)
    weights = signed_weights[..., 1::2] - signed_weights[..., ::2]
    signed_curves = np.asarray(file["arr_0"][()]["escoreList"])
    eigencurves = signed_curves[wave_idx, :np.shape(signed_curves)[1]//2, ...]

    las = np.linspace(0, np.pi, N_lat)
    los = np.linspace(-np.pi, np.pi, N_lon)

    sph_l = np.concatenate([np.tile(l, l+1) for l in range(1, degree)])
    sph_m = np.concatenate([np.arange(l+1) for l in range(1, degree)])

    base_harmonics = sph_harm(np.tile(sph_m, (N_lon, N_lat, 1)).T,
                              np.tile(sph_l, (N_lon, N_lat, 1)).T,
                              *np.meshgrid(los, las))

    fluxes = np.sum([
                np.einsum('m...vu,m->...vu', np.array(
                          [np.einsum('...x,xvu->...vu',
                                     weights[...,
                                             l*(l+1)-1 + np.array([m, -m])],
                                     np.array([base_harmonics[l*(l-1)+m].real,
                                               base_harmonics[l*(l-1)+m].imag
                                               ]))
                           for m in range(l+1)]),
                          [1] + l * [((-1)**l)*np.sqrt(2)])
                for l in range(1, degree)], axis=0)

    # Here we convert to (-pi, pi) in longitude, and (-pi/2, pi/2) in latitude,
    # and multiply by the factor that normalizes the harmonics.
    eigenmaps = np.real(2*np.sqrt(np.pi) *
                        np.flip(fluxes, axis=-2))

    return eigenmaps, eigencurves


def get_flux_curves(flux_maps, degree, wave_idx, N_points=1000):
    phases = np.linspace(-180, 180, 1000)
    curves = []
    for flux in flux_maps[:(degree**2-1), ...]:
        map = starry.Map(ydeg=degree-1)
        map.load(flux)
        curves.append(map.flux(theta=phases))

    return curves


def make_eigenplots(output_file, degree, wave_idx, save_name=None):
    all_eigenmaps, eigencurves = get_flux_maps(output_file)
    eigenmaps = all_eigenmaps[wave_idx]

    map_min = np.min(eigenmaps)
    map_max = np.max(eigenmaps)

    flip_curves = {"1": np.array([1, 1, 1]),
                   "2": np.array([-1, 1, 1, 1, 1, 1, 1, 1])}
    plotted_curves = flip_curves[str(degree-1)][:, np.newaxis] * eigencurves
    curve_min = np.min(plotted_curves)
    curve_max = np.max(plotted_curves)

    map_colors = {"1": "Blues_r",
                  "2": "Oranges_r",
                  "3": "Greens_r"}
    cmap = map_colors[str(degree-1)]

    fig, axes = plt.subplots(3, degree**2-1, figsize=(6*(degree**2-1), 9))
    for i, (map, ortho_ax, rect_ax) in enumerate(zip(eigenmaps,
                                                     axes[1, :].flatten(),
                                                     axes[2, :].flatten())):
        eigenmap = starry.Map(ydeg=degree-1)
        eigenmap.load(map)
        eigenmap.show(ax=ortho_ax, cmap=cmap)
        if i > 0:
            rect_ax.yaxis.set_ticklabels([])
        rect_ax.imshow(map, vmin=map_min, vmax=map_max,
                       cmap=cmap,
                       extent=(-180, 180, -90, 90))
        rect_ax.xaxis.tick_top()

    phases = np.linspace(-180, 180, np.shape(plotted_curves)[-1])
    for curve, ax in zip(plotted_curves,
                         axes[0, :].flatten()):
        ax.plot(phases, curve, linewidth=2, c="C{}".format(degree-2))
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(-180, 180)
        ax.set_ylim(curve_min - 0.05*(curve_max-curve_min),
                    curve_max + 0.05*(curve_max-curve_min))

    axes[2, 0].set_ylabel(r"Latitude $\left(^\circ\right)$", fontsize=18)
    fig.suptitle(r"Offset Hotspot Map, $\ell_\mathrm{{max}}={}$".format(
            degree-1), fontsize=22)
    fig.add_subplot(311, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel(r"Phase from Eclipse $\left(^\circ\right)$", fontsize=18)

    # The subplot bounds are values that produce nicely-aligned
    # plots, tweaked by hand.
    fig.add_subplot(312, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel(r"Longitude from Sub-stellar $\left(^\circ\right)$",
               fontsize=18)

    plt.figure(fig.number)
    plot_dimensions = {"1": {"top": 0.95,
                             "bottom": 0.01,
                             "left": 0.05,
                             "right": 0.99,
                             "hspace": 0.54,
                             "wspace": 0.05},
                       "2": {"top": 0.93,
                             "bottom": 0.,
                             "left": 0.045,
                             "right": 0.985,
                             "hspace": 0.16,
                             "wspace": 0.085}}
    if str(degree-1) in plot_dimensions.keys():
        plt.subplots_adjust(**plot_dimensions[str(degree-1)])

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, dpi=fig.dpi)

    return fig, axes

  
map_problem = 1
degree = 3
wave_idx = 0
output_file = "data/sph_harmonic_coefficients_full_samples/good/mystery{}/spherearray_deg_{}.npz".format(map_problem, degree)
fig, axes = make_eigenplots(output_file, degree, wave_idx)
