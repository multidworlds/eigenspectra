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


def get_flux_maps(output_file, N_lat=181, N_lon=361):
    file = np.load(output_file, encoding='latin1', allow_pickle=True)
    harmonic_weights = np.asarray(file["arr_0"][()]["ecoeffList"])
    weights = np.einsum("abc->acb", harmonic_weights)

    las = np.linspace(0, np.pi, N_lat)
    los = np.linspace(-np.pi, np.pi, N_lon)

    sph_l = np.concatenate([np.tile(l, l+1) for l in range(1, degree)])
    sph_m = np.concatenate([np.arange(l+1) for l in range(1, degree)])

    base_harmonics = sph_harm(np.tile(sph_m, (N_lon, N_lat, 1)).T,
                              np.tile(sph_l, (N_lon, N_lat, 1)).T,
                              *np.meshgrid(los, las))

    pos_fluxes = np.sum([
                np.einsum('m...vu,m->...vu', np.array(
                        [np.einsum('...x,xvu->...vu',
                                   weights[..., 1::2][...,
                                                      l*(l+1)-1 +
                                                      np.array([m, -m])
                                                      ],
                                   np.array([base_harmonics[l*(l-1)+m].real,
                                             base_harmonics[l*(l-1)+m].imag
                                             ]))
                            for m in range(l+1)]),
                        [1] + l * [((-1)**l)*np.sqrt(2)])
                for l in range(1, degree)], axis=0)

    neg_fluxes = np.sum([
                np.einsum('m...vu,m->...vu', np.array(
                        [np.einsum('...x,xvu->...vu',
                                   weights[..., ::2][...,
                                                     l*(l+1)-1 +
                                                     np.array([m, -m])
                                                     ],
                                   np.array([base_harmonics[l*(l-1)+m].real,
                                             base_harmonics[l*(l-1)+m].imag
                                             ]))
                            for m in range(l+1)]),
                        [1] + l * [((-1)**l)*np.sqrt(2)])
                for l in range(1, degree)], axis=0)

    # Here we convert to (-pi, pi) in longitude, and (-pi/2, pi/2) in latitude,
    # and multiply by the factor that normalizes the harmonics.
    fluxes = np.real(2*np.sqrt(np.pi) *
                     np.flip(pos_fluxes-neg_fluxes, axis=-2))

    return fluxes


def get_flux_curves(flux_maps, degree, wave_idx, N_points=1000):
    phases = np.linspace(-180, 180, 1000)
    curves = []
    for flux in flux_maps[:(degree**2-1), ...]:
        map = starry.Map(ydeg=degree-1)
        map.load(flux)
        curves.append(map.flux(theta=phases))

    return curves


def make_eigenplots(output_file, degree, wave_idx, save_name=None):
    eigenmaps = get_flux_maps(output_file)[wave_idx]
    eigencurves = get_flux_curves(eigenmaps, degree, wave_idx)

    map_min = np.min(eigenmaps)
    map_max = np.max(eigenmaps)
    curve_min = np.min(eigencurves)
    curve_max = np.max(eigencurves)

    map_colors = {"1": "Blues_r",
                  "2": "Oranges_r",
                  "3": "Greens_r"}

    fig, axes = plt.subplots(2, degree**2-1, figsize=(6*(degree**2-1), 6))
    for i, (map, ax) in enumerate(zip(eigenmaps, axes[1, :].flatten())):
        if i > 0:
            ax.yaxis.set_ticklabels([])
        ax.imshow(map, vmin=map_min, vmax=map_max,
                  cmap=map_colors[str(degree-1)],
                  extent=(-180, 180, -90, 90))
    phases = np.linspace(-180, 180, np.shape(eigencurves)[-1])
    for curve, ax in zip(eigencurves, axes[0, :].flatten()):
        ax.plot(phases, curve, linewidth=2, c="C{}".format(degree-2))
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(-180, 180)
        ax.set_ylim(curve_min - 0.05*(curve_max-curve_min),
                    curve_max + 0.05*(curve_max-curve_min))

    axes[1, 0].set_ylabel(r"Latitude $\left(^\circ\right)$", fontsize=18)
    fig.suptitle(r"Offset Hotspot Map, $\ell_\mathrm{{max}}={}$".format(
            degree-1), fontsize=22)
    curve_figure = fig.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel(r"Phase from Eclipse $\left(^\circ\right)$", fontsize=18)

    # The offset and dimension values are values that produce nicely-aligned
    # plots, tweaked by hand.
    label_offset = {"1": 0.515,
                    "2": 0.5215}
    fig.text(label_offset[str(degree-1)], 0.04,
             r"Longitude from Sub-stellar $\left(^\circ\right)$",
             ha="center", fontsize=18)

    plot_dimensions = {"1": {"top": 0.91,
                             "bottom": 0.13,
                             "left": 0.04,
                             "right": 0.985,
                             "hspace": 0.384,
                             "wspace": 0.054},
                       "2": {"top": 0.93,
                             "bottom": 0.09,
                             "left": 0.045,
                             "right": 0.99,
                             "hspace": 0.284,
                             "wspace": 0.089}}
    if str(degree-1) in plot_dimensions.keys():
        plt.subplots_adjust(**plot_dimensions[str(degree-1)])

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, dpi=300)

    return fig, axes


map_problem = 1
degree = 3
wave_idx = 0
output_file = "data/sph_harmonic_coefficients_full_samples/good/mystery{}/spherearray_deg_{}.npz".format(map_problem, degree)
fig, axes = make_eigenplots(output_file, degree, wave_idx)
