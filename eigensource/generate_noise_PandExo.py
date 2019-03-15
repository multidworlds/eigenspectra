# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pandexo.engine.justdoit import load_exo_dict, run_pandexo
from pandeia.engine.perform_calculation import perform_calculation


def sum_spectrum(img, ap=4, center=None):
    '''
    Simply sums the spectrum along spatial direction.
    '''
    if center is None:
        center = img.shape[0]/2
    subImg = img[(center-ap):(center+ap), :]

    return np.sum(subImg, axis=0)


def pandexo_dict(planet):
    '''
    Construct the dictionary for PandExo using system-specific inputs.
    '''
    exo_dict = {'observation': {
                    'sat_level': 80,
                    'sat_unit': '%',
                    'noccultations': 2,
                    'R': None,
                    'baseline_unit': 'total',
                    'baseline': 4.0*60.0*60.0,
                    'noise_floor': 0
                },
                'planet': planet.PandExo_planet,
                'star': planet.PandExo_star
                }

    return exo_dict


def make_snr_spectrum(planet):
    '''
    Run PandExo, extract a single integration, and return the noise.
    '''
    exo_dict = load_exo_dict()
    exo_dict = {**exo_dict, **pandexo_dict(planet)}
    result = run_pandexo(exo_dict, ['NIRCam F322W2'])
    texp = result['timing']['Time/Integration incl reset (sec)']
    newDictInput = deepcopy(result['PandeiaOutTrans']['input'])
    newDictInput['configuration']['detector']['nint'] = 1
    PandeiaResult = perform_calculation(newDictInput)
    noise = PandeiaResult['1d']['extracted_noise']

    fig, ax = plt.subplots()

    plt.plot(*noise)
    plt.ylim(300, 900)

    fig.savefig('data/instrument/snr_spectrum.pdf')

    return {'wave': noise[0], 'sigma_ppm': noise[1], 'texp': texp}
