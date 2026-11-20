import argparse
import os
import re

import awkward as ak
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import vector
from matplotlib import rcParams

hep.style.use(hep.style.ROOT)
vector.register_awkward()
parser = argparse.ArgumentParser()
parser.add_argument(
    'root_files', metavar='root-files', 
    help='\',\' delimited list of ROOT files (including glob syntax) to use'
)
parser.add_argument(
    'plot_dir', metavar='plot-dir', 
    help='Path to output directory for plots (if it doesn\'t exist it will be created)'
)
parser.add_argument(
    '--load-method', default='needed-at-once', type=str,
    help='Method describing how to load root files. Default loads all columns of all files at once'
)
######################################################

NEVTS = 0
CUTIDX = 0
EVENT_CUTFLOW = []
OBJECT_CUTFLOW = {}
REQUIRED_FIELDS = ['event', 'L1GTgmtTkMuon_*', 'hltGeneralTrack_*', 'GenPart_*', 'L1PV_*', 'hltPrimaryVertex4D_*']
######################################################

def field_regex(regfunc, sequence):
    return [item for item in sequence if regfunc(item)]

def load_uproot(files: list[str], method: str):
    if method == 'all-at-once':
        return uproot.concatenate({file: 'Events' for file in files})
    elif method == 'needed-at-once':
        return uproot.concatenate({file: 'Events' for file in files}, filter_name=REQUIRED_FIELDS)
    else: raise NotImplementedError(f"Requested method for file-loading ({method}) isn't implemented")


def add_event_cut(prev_mask, new_mask, label):
    global CUTIDX, EVENT_CUTFLOW
    and_mask = np.logical_and(prev_mask, new_mask)
    if CUTIDX <= 1:
        EVENT_CUTFLOW.append(f'{CUTIDX}) {label}: {np.sum(and_mask)}, {100*np.sum(and_mask)/NEVTS:.2f}%')
    else:
        EVENT_CUTFLOW.append(f'{CUTIDX}) {label.replace("requiring", "requiring previous and")}: {np.sum(and_mask)}, {100*np.sum(and_mask)/NEVTS:.2f}%')
        EVENT_CUTFLOW.append(f'    --> {label}: {np.sum(new_mask)}, {100*np.sum(new_mask)/NEVTS:.2f}%')
    CUTIDX += 1
    return and_mask

def print_cutflow(obj: list[str], title: str=''):
    if title != '':
        print(title);print('-'*60)
    for statement in obj: print(statement)


@nb.njit
def nb_reshape(data_array: ak.Array, shape_array: ak.Array, out_array: ak.ArrayBuilder):
    for data_event, shape_event in zip(data_array, shape_array):
        out_array.begin_list()
        for _ in shape_event:
            out_array.append(data_event)
        out_array.end_list()
    return out_array
        

@nb.njit
def nb_DR(array1: ak.Array, array2: ak.Array, outDR: ak.ArrayBuilder):
    for array1_event, array2_event in zip(array1, array2):
        outDR.begin_list()
        for vec1 in array1_event:
            outDR.begin_list()
            DRs = [0.0 for _ in range(0)]
            for vec2 in array2_event:
                outDR.append(((vec1.eta - vec2.eta)**2 + (vec1.phi - vec2.phi)**2)**0.5)
            outDR.end_list()
        outDR.end_list()
    return outDR
@nb.njit
def nb_var_at_minDR(array1: ak.Array, array2: ak.Array, array2var: ak.Array, outDR: ak.ArrayBuilder):
    for array1_event, array2_event, array2var_event in zip(array1, array2, array2var):
        outDR.begin_list()
        for vec1 in array1_event:
            minDR, var = 999., -999.
            for vec2, var2 in zip(array2_event, array2var_event):
                vec1vec2DR = ((vec1.eta - vec2.eta)**2 + (vec1.phi - vec2.phi)**2)**0.5
                if vec1vec2DR < minDR:
                    minDR = vec1vec2DR
                    var = var2
            outDR.append(var)
        outDR.end_list()
    return outDR

@nb.njit
def nb_Dz(array1: ak.Array, array2: ak.Array, outDz: ak.ArrayBuilder):
    for array1_event, array2_event in zip(array1, array2):
        outDz.begin_list()
        for vec1 in array1_event:
            outDz.begin_list()
            for vec2 in array2_event:
                outDz.append(((vec1.z - vec2.z)**2)**0.5)
            outDz.end_list()
        outDz.end_list()
    return outDz
@nb.njit
def nb_var_at_minDz(array1: ak.Array, array2: ak.Array, array2var: ak.Array, outDz: ak.ArrayBuilder):
    for array1_event, array2_event, array2var_event in zip(array1, array2, array2var):
        outDz.begin_list()
        for vec1 in array1_event:
            minDz, var = 999., -999.
            for vec2, var2 in zip(array2_event, array2var_event):
                vec1vec2Dz = ((vec1.z - vec2.z)**2)**0.5
                if vec1vec2Dz < minDz:
                    minDz = vec1vec2Dz
                    var = var2
            outDz.append(var)
        outDz.end_list()
    return outDz
    

def main(files: list[str], plot_dir: str, load_method: str):
    events = load_uproot(files, load_method)
    global NEVTS; NEVTS = ak.num(events['event'], axis=0)
    event_mask = np.ones(NEVTS, dtype=bool)
    event_mask = add_event_cut(event_mask, event_mask, 'N. events before selections')

    ############################
    ## OBJECTS ##
    # L1 Global Tracker Muons
    l1mu = ak.zip({
        field.replace('L1GTgmtTkMuon_', ''): events[field]
        for field in field_regex(lambda field: re.match('L1GTgmtTkMuon_', field) is not None, events.fields)
    }, with_name='Momentum3D')
    # HLT Tracks
    hlttk = ak.zip({
        field.replace('hltGeneralTrack_', ''): events[field]
        for field in field_regex(lambda field: re.match('hltGeneralTrack_', field) is not None, events.fields)
    }, with_name='Momentum4D')
    hlttk['t'] = hlttk['t0safe'] * 1_000  # convert from ns to ps
    hlttk['sigmat'] = hlttk['sigmat0safe'] * 1_000  # convert from ns to ps
    # All (interesting) generator particles
    genpart = ak.zip({
        field.replace('GenPart_', ''): events[field]
        for field in field_regex(lambda field: re.match('GenPart_', field) is not None, events.fields)
    }, with_name='Momentum4D')
    # Generator Muons from W/Gamma/Z
    genmu = genpart[
        (np.abs(genpart['pdgId']) == 13) & (                                 # isMuon
            (np.abs(genpart[genpart['genPartIdxMother']]['pdgId']) == 24)    # W
            # (np.abs(genpart[genpart['genPartIdxMother']]['pdgId']) == 22) |  # photon
            # (np.abs(genpart[genpart['genPartIdxMother']]['pdgId']) == 23)    # Z
        ) & ((genpart['statusFlags'] & 128) == 128)                          # isHardProcess
    ]
    # Generator Muons NOT from W/Gamma/Z
    genmu_notpv = genpart[
        (np.abs(genpart['pdgId']) == 13) &                                   # isMuon
        ((genpart['statusFlags'] & 128) != 128)                              # NOT isHardProcess
    ]
    # L1 Primary Vertices (PVs)
    l1vtx = ak.zip({
        field.replace('L1PV_', ''): events[field]
        for field in field_regex(lambda field: re.match('L1PV_', field) is not None, events.fields)
    })
    l1vtx['z'] = l1vtx['z0']
    # HLT Vertices
    hltvtx = ak.zip({
        field.replace('hltPrimaryVertex4D_', ''): events[field]
        for field in field_regex(lambda field: re.match('hltPrimaryVertex4D_', field) is not None, events.fields)
    }, with_name='Momentum4D')
    hltvtx['t'] = hltvtx['t'] * 1_000  # convert from ns to ps
    hltvtx['sigmat'] = hltvtx['tError'] * 1_000  # convert from ns to ps

    ############################
    ## OBJECT SELECTIONS ##
    good_l1mu = l1mu[(l1mu.pt > 10) & (np.abs(l1mu.eta) < 2.4)]
    good_hlttk = hlttk[(hlttk['sigmat0safe'] > 0) & (hlttk['mtdQualMVA'] > 0.5)]
    good_l1vtx = l1vtx[l1vtx.z0 > -32]
    good_hltvtx = ak.with_field(hltvtx[hltvtx.isGood > 0], ak.local_index(hltvtx[hltvtx.isGood > 0], axis=1), 'idx')

    ############################
    ## EVENT SELECTIONS ##
    cut_sequence = [
        (ak.num(good_l1vtx, axis=1) > 0, 'N. events after requiring at least 1 good L1 vertex'),
        (ak.num(good_hltvtx, axis=1) > 0, 'N. events after requiring at least 1 good HLT vertex'),
        (ak.num(good_hlttk, axis=1) > 0, 'N. events after requiring at least 1 good HLT track'),
        (ak.num(good_l1mu, axis=1) > 0, 'N. events after requiring at least 1 good L1-mu'),
        (ak.num(genmu, axis=1) > 0, 'N. events after requiring at least 1 Gen-mu from gamma/Z/W'),
    ]
    for cut_mask, label in cut_sequence:
        event_mask = add_event_cut(event_mask, cut_mask, label)

    # Find matched L1 muons and HLT tracks (require at least 1 per-event)
    l1mu_hlttk_good_matches = (ak.firsts(ak.sort(nb_DR(good_l1mu, good_hlttk, ak.ArrayBuilder()).snapshot(), axis=-1), axis=-1) < 0.1)
    l1mu_hlttk_matched_muons = ak.with_field(good_l1mu, nb_var_at_minDR(good_l1mu, good_hlttk, good_hlttk['t'], ak.ArrayBuilder()).snapshot(), 't')[l1mu_hlttk_good_matches]
    l1mu_hlttk_matched_muons = ak.with_field(l1mu_hlttk_matched_muons, nb_var_at_minDR(l1mu_hlttk_matched_muons, good_hlttk, good_hlttk['sigmat'], ak.ArrayBuilder()).snapshot(), 'sigmat')
    event_mask = add_event_cut(
        event_mask, ak.num(l1mu_hlttk_good_matches, axis=1) > 0, 
        'N. events after requiring at least 1 L1 muon matched to a good HLT track'
    )

    # Find matched L1 vertices and HLT vertices (require at least 1 per-event)
    l1vtx_hltvtx_good_matches = (ak.firsts(ak.sort(nb_Dz(good_l1vtx, good_hltvtx, ak.ArrayBuilder()).snapshot(), axis=-1), axis=-1) < 0.2)
    l1vtx_hltvtx_matched_vtx = ak.with_field(good_l1vtx, nb_var_at_minDz(good_l1vtx, good_hltvtx, good_hltvtx['t'], ak.ArrayBuilder()).snapshot(), 't')[l1vtx_hltvtx_good_matches]
    l1vtx_hltvtx_matched_vtx = ak.with_field(l1vtx_hltvtx_matched_vtx, nb_var_at_minDz(l1vtx_hltvtx_matched_vtx, good_hltvtx, good_hltvtx['x'], ak.ArrayBuilder()).snapshot(), 'x')
    l1vtx_hltvtx_matched_vtx = ak.with_field(l1vtx_hltvtx_matched_vtx, nb_var_at_minDz(l1vtx_hltvtx_matched_vtx, good_hltvtx, good_hltvtx['y'], ak.ArrayBuilder()).snapshot(), 'y')
    l1vtx_hltvtx_matched_vtx = ak.with_field(l1vtx_hltvtx_matched_vtx, nb_var_at_minDz(l1vtx_hltvtx_matched_vtx, good_hltvtx, good_hltvtx['sigmat'], ak.ArrayBuilder()).snapshot(), 'sigmat')
    l1vtx_hltvtx_matched_vtx = ak.with_field(l1vtx_hltvtx_matched_vtx, nb_var_at_minDz(l1vtx_hltvtx_matched_vtx, good_hltvtx, good_hltvtx['idx'], ak.ArrayBuilder()).snapshot(), 'hltidx')
    event_mask = add_event_cut(
        event_mask, ak.num(l1vtx_hltvtx_good_matches, axis=1) > 0, 
        'N. events after requiring at least 1 good L1 vertex matched to any good HLT vertex'
    )
    l1vtx_hltvtx_matched_vtx = l1vtx_hltvtx_matched_vtx[l1vtx_hltvtx_matched_vtx['hltidx'] == 0]    
    event_mask = add_event_cut(
        event_mask, ak.num(l1vtx_hltvtx_matched_vtx, axis=1) > 0, 
        'N. events after requiring at least 1 good L1 vertex matched to the primary HLT vertex'
    )

    # Find real (gen-matched) and fake muons
    good_muons = l1mu_hlttk_matched_muons[event_mask]
    good_vertices = l1vtx_hltvtx_matched_vtx[event_mask]
    real_muon_mask = (ak.firsts(ak.sort(nb_DR(good_muons, genmu[event_mask], ak.ArrayBuilder()).snapshot(), axis=-1), axis=-1) < 0.1)
    real_notPVmuon_mask = ak.fill_none(ak.firsts(ak.sort(nb_DR(good_muons, genmu_notpv[event_mask], ak.ArrayBuilder()).snapshot(), axis=-1), axis=-1) < 0.1, False)
    real_muons = ak.cartesian({'mu': good_muons[real_muon_mask], 'vtx': good_vertices})
    real_muons['deltat'] = real_muons['mu'].t - real_muons['vtx'].t
    real_notPVmuons = ak.cartesian({'mu': good_muons[real_notPVmuon_mask], 'vtx': good_vertices})
    real_notPVmuons['deltat'] = real_notPVmuons['mu'].t - real_notPVmuons['vtx'].t
    fake_muons = ak.cartesian({'mu': good_muons[~(real_muon_mask | real_notPVmuon_mask)], 'vtx': good_vertices})
    fake_muons['deltat'] = fake_muons['mu'].t - fake_muons['vtx'].t
    
    flatnp = lambda akarr: ak.to_numpy(ak.flatten(akarr))
    for density in [True, False]:
        plt.figure()
        plt.hist(np.abs(flatnp(real_muons['deltat'])), density=density, bins=50, range=(0., 1500.), label='Gen-matched & PV '+r'$\mu$', histtype='step')
        plt.hist(np.abs(flatnp(real_notPVmuons['deltat'])), density=density, bins=50, range=(0., 1500.), label='Gen-matched & Not PV '+r'$\mu$', histtype='step')
        plt.hist(np.abs(flatnp(fake_muons['deltat'])), density=density, bins=50, range=(0., 1500.), label='Not Gen-matched '+r'$\mu$', histtype='step')
        plt.legend()
        plt.title(('Density ' if density else '')+r'$\Delta t^{\mu}_{vtx}$ for Real and Fake Muons')
        plt.xlabel(r'$\Delta t^{\mu}_{vtx}$')
        plt.ylabel('N. events')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"deltat_real_fake{'_density' if density else ''}.png"))

    # pt vs. eta L1 muon - HLT track matching
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rcParams["figure.figsize"][0]*2, rcParams["figure.figsize"][1]))
    hdenom, xbins, ybins, image_denom = ax1.hist2d(flatnp(good_l1mu.pt), flatnp(good_l1mu.eta), range=((0., 100.),(-2.4, 2.4)), bins=50)
    ax1.set_title(f"{ak.sum(ak.num(good_l1mu, axis=1), axis=0)} "+'L1 '+r'$\mu$')
    ax1.set_xlabel(r'$\mu_{p_T}$')
    ax1.set_ylabel(r'$\mu_{\eta}$')
    fig.colorbar(image_denom, ax=ax1)
    hnum, _, _, image_num = ax2.hist2d(flatnp(l1mu_hlttk_matched_muons.pt), flatnp(l1mu_hlttk_matched_muons.eta), range=((0., 100.),(-2.4, 2.4)), bins=50)
    ax2.set_title(f"{ak.sum(ak.num(l1mu_hlttk_matched_muons, axis=1), axis=0)} "+'L1 '+r'$\mu$'+' matched to HLT track')
    ax2.set_xlabel(r'$\mu_{p_T}$')
    ax2.set_ylabel(r'$\mu_{\eta}$')
    fig.colorbar(image_num, ax=ax2)
    fig.suptitle(r'$p_T vs. \eta$ of L1 $\mu$ with and without HLT track matching')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"l1mu_pt_vs_eta_with_without_match.png"))

    # pt vs. eta L1 muon - HLT track matching efficiency
    plt.figure()
    image = plt.imshow((hnum / hdenom).T, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower')
    plt.title(r'$p_T vs. \eta$ of L1 $\mu$ to HLT track matching efficiency')
    plt.xlabel(r'$\mu_{p_T}$')
    plt.ylabel(r'$\mu_{\eta}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"matchedl1mu_pt_vs_eta_efficiency.png"))

    # pt vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['mu'].pt), np.abs(flatnp(real_muons['deltat'])), range=((0., 20.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$\mu_{p_T} vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$\mu_{p_T}$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"pt_vs_deltat_real_density.png"))

    # eta vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['mu'].eta), np.abs(flatnp(real_muons['deltat'])), range=((-5., 5.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$\mu_{\eta} vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$\mu_{\eta}$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"eta_vs_deltat_real_density.png"))

    # t_tk vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['mu'].t), np.abs(flatnp(real_muons['deltat'])), range=((-1500., 1500.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$\mu_t vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$\mu_t$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"ttk_vs_deltat_real_density.png"))

    # sigmat_tk vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['mu']['sigmat']), np.abs(flatnp(real_muons['deltat'])), range=((0., 50.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$\mu_{\sigma_t} vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$\mu_{\sigma_t}$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"sigmattk_vs_deltat_real_density.png"))

    # t_vtx vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['vtx'].t), np.abs(flatnp(real_muons['deltat'])), range=((-1500., 1500.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$vtx_t vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$vtx_t$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"tvtx_vs_deltat_real_density.png"))

    # sigmat_vtx vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['vtx']['sigmat']), np.abs(flatnp(real_muons['deltat'])), range=((0., 50.),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$vtx_{\sigma_t} vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$vtx_{\sigma_t}$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"sigmatvtx_vs_deltat_real_density.png"))

    # x vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['vtx'].x), np.abs(flatnp(real_muons['deltat'])), range=((-0.005, 0.005),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$vtx_x vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$vtx_x$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"x_vs_deltat_real_density.png"))

    # y vs. delta t
    plt.figure()
    _, _, _, image = plt.hist2d(flatnp(real_muons['vtx'].y), np.abs(flatnp(real_muons['deltat'])), range=((-0.005, 0.005),(0., 1500.)), bins=50, label='Real (gen-matched) '+r'$\mu$')
    plt.title('Density '+r'$vtx_y vs. \Delta t^{\mu}_{vtx}$ for Real Muons')
    plt.xlabel(r'$vtx_y$')
    plt.ylabel(r'$\Delta t^{\mu}_{vtx}$')
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"y_vs_deltat_real_density.png"))

if __name__ == "__main__":
    args = parser.parse_args()

    # ROOT files to use
    root_files = args.root_files.split(',')
    root_files = [(os.path.join(root_file, '*.root') if not root_file.endswith('.root') else root_file) for root_file in root_files]

    # Output directory for plots
    if args.plot_dir.find('/') < 0: plot_dir = os.path.join(os.getcwd(), args.plot_dir)
    else: plot_dir = os.path.normpath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # Run code
    main(root_files, plot_dir, args.load_method)
    print('+'*60)
    print_cutflow(EVENT_CUTFLOW, 'Event Cutflow')