import numpy as np
import basf2
import generators as ge
import simulation as si
from ROOT import Belle2
import argparse
from time import perf_counter_ns
import glob

from utils import append_cdc_preliminary_modules
from tracking.path_utils import add_cdc_track_finding
from tracking.path_utils import add_mc_matcher
from scipy.stats import crystalball

argParser = argparse.ArgumentParser(description="Steering file for CAT LINK performance analysis.")
argParser.add_argument('-e', '--event-type', choices=['mumu', 'tautau', 'BBbar','BB'], help="'mumu': mu- mu+, 'tautau': tau- tau+, 'BBbar': B Bbar, 'BB': B+ B-", required=True)
argParser.add_argument('-a', '--tracking-algorithm', choices=['leg', 'cat', 'int'], help="'leg': Legendre tracking, 'cat': CATFinder, 'int': CAT LINK", required=True)
argParser.add_argument('-b', '--background', choices=['none', 'low', 'high'], help='0: no background, 1: low background, 2: high background', required=True)
argParser.add_argument('-o', '--output-file', required=True)

args = argParser.parse_args()

mc_keys=['px', 'py', 'pz', 'mcid', 'mcpdg']
mc_results={}
for key in mc_keys:
    mc_results[key]=[]

reco_keys=['px', 'py', 'pz', 'mcid', 'mcpdg']
reco_results={}
for key in reco_keys:
    reco_results[key]=[]

def getBin(linspace, value):
    for i in range(len(linspace)):
        if value >= linspace[i] and value < linspace[i]+linspace[1]-linspace[0]:
            return i        
    return None

def crystalball_pdf(x, amplitude, alpha, n, mean, sigma, offset):
    return amplitude * crystalball.pdf(x - offset, alpha, n, loc=mean, scale=sigma)

def crystalball_fwhm(alpha, sigma, n):
    return 2 * sigma * (abs(alpha) ** (1 / n)) * (crystalball.cdf(sigma / abs(alpha), alpha, n) + 1 - crystalball.cdf(-sigma / abs(alpha), alpha, n))


class RecoTrackInfo(basf2.Module):
    """Module for benchmarking the track-finding efficiency, fake rate and p_t resolution"""

    def __init__(self, padmax, debug=False):
        super().__init__()

        self.debug = debug
        self.padmax = padmax

        self.ptRange = np.linspace(0, 6, 30, endpoint=False)                        # Bins for p_t histograms
        self.resolutionPtRange = np.linspace(-0.15, 0.15, 120, endpoint=False)      # Bins for CB distributions
        self.recoPts = np.zeros(len(self.ptRange))
        self.fakePts = np.zeros(len(self.ptRange))
        self.mcPts = np.zeros(len(self.ptRange))
        self.efficiencies = np.zeros(len(self.ptRange))
        self.fakeRates = np.zeros(len(self.ptRange))

        self.recoHist = np.zeros(len(self.ptRange))
        self.mcHist = np.zeros(len(self.ptRange))

        self.resHists = [np.zeros(len(self.resolutionPtRange)) for n in self.ptRange]

    def initialize(self):
        self.mcparticles = Belle2.PyStoreArray("MCParticles")
        self.reco = Belle2.PyStoreArray("RecoTracks")

    def event(self):
        def zeropad(x, padlength=self.padmax, padfill=0.0):
            return np.float32(
                np.pad(
                    x,
                    pad_width=(padlength - len(x)),
                    mode="constant",
                    constant_values=padfill,
                )
            )[padlength - len(x) :]
        tmp_reco_results = {}
        for result in reco_results:
            tmp_reco_results[result] = []
        tmp_mc_results = {}
        for result in mc_keys:
            tmp_mc_results[result] = []

        recoIndices = []
        mcIndices = {}

        for i, recoTrack in enumerate(self.reco):
            mc_recotrack = recoTrack.getRelationsWith("MCParticles")
            if mc_recotrack.size() > 0:
                mc_pdg=mc_recotrack.object(0).getPDG()
                mc_id=mc_recotrack.object(0).getArrayIndex()
            else:
                mc_pdg=-9999
                mc_id=-9999

            pred_px=recoTrack.getMomentumSeed().X()
            pred_py=recoTrack.getMomentumSeed().Y()
            pred_pz=recoTrack.getMomentumSeed().Z()
            pred_pt = np.hypot(pred_px, pred_py)
            tmp_reco_results['px'].append(pred_px)
            tmp_reco_results['py'].append(pred_py)
            tmp_reco_results['pz'].append(pred_pz)
            tmp_reco_results['mcid'].append(mc_id)
            tmp_reco_results['mcpdg'].append(mc_pdg)

            if mc_recotrack.size() == 0:
                self.fakePts[getBin(self.ptRange, pred_pt)] += 1

            if mc_recotrack.size() > 0:
                self.recoPts[getBin(self.ptRange, pred_pt)] += 1

            recoIndices += ([mc.getArrayIndex() for mc in mc_recotrack])

            # Resolution
            if mc_recotrack.size() > 0:
                mcPt = np.hypot(mc_recotrack.object(0).getMomentum().X(), mc_recotrack.object(0).getMomentum().Y())
                
                histVal = (pred_pt - mcPt) / mcPt
                histBin = getBin(self.resolutionPtRange, histVal)
                if histBin:
                    self.resHists[getBin(self.ptRange, mcPt)][histBin] += 1

        recoIndices = list(set(recoIndices))

        #save reco track
        for variable in tmp_mc_results:
            reco_results[variable].append(
                zeropad(tmp_reco_results[variable][: self.padmax])
            )
        
        for idx, mcparticle in enumerate(self.mcparticles):
            if mcparticle.isPrimaryParticle() :
                hitrelations = mcparticle.getRelationsWith("CDCHits")
                nhitrelations = hitrelations.size()
                if nhitrelations > 0:
                    mcMomentum = mcparticle.getMomentum()
                    m_primary_particle_px = mcMomentum.X()
                    m_primary_particle_py = mcMomentum.Y()
                    m_primary_particle_pz = mcMomentum.Z()
                    mc_primary_particle_pt = np.hypot(m_primary_particle_px, m_primary_particle_py)
                    tmp_mc_results['px'].append(m_primary_particle_px)
                    tmp_mc_results['py'].append(m_primary_particle_py)
                    tmp_mc_results['pz'].append(m_primary_particle_pz)
                    tmp_mc_results['mcpdg'].append(mcparticle.getPDG())
                    tmp_mc_results['mcpdg'].append(mcparticle.getArrayIndex())

                    self.mcPts[getBin(self.ptRange, mc_primary_particle_pt)] += 1

                    index = mcparticle.getArrayIndex()

                    if not index in list(mcIndices.keys()):
                        mcIndices[index] = mc_primary_particle_pt

        for index in list(mcIndices.keys()):
            bin = getBin(self.ptRange, mcIndices[index])
            self.mcHist[bin] += 1
            if index in recoIndices:
                self.recoHist[bin] += 1

        for variable in tmp_mc_results:
            mc_results[variable].append(
                zeropad(tmp_mc_results[variable][: self.padmax])
            )
        
    def terminate(self):        

        for i in range(len(self.ptRange)):
            try:
                self.efficiencies[i] = self.recoHist[i] / self.mcHist[i]
            except RuntimeWarning:
                self.efficiencies[i] = -1

            try:
                self.fakeRates[i] = self.fakePts[i] / self.mcPts[i]
            except RuntimeWarning:
                self.fakeRates[i] = -1

        saveFile = args.output_file

        with open(saveFile, 'a') as file:
            # Save the bins of the p_t and CB histograms
            file.write(f'\nptRange:{",".join([str(v) for v in self.ptRange.tolist()])}\nresPtRange:{",".join([str(v) for v in self.resolutionPtRange.tolist()])}')

        for hist in self.resHists:
            # Save all of the CB histograms
            with open(saveFile, 'a') as file:
                    file.write(f'\nhistogram:{",".join([str(v) for v in hist.tolist()])}')

        with open(saveFile, 'a') as file:
            # Save the fake rates and efficiencies
            file.write(f'\nfakeRates:{",".join([str(v) for v in self.fakeRates.tolist()])}\nefficiencies:{",".join([str(v) for v in self.efficiencies.tolist()])}')

class CATFinder(basf2.Module):

    def __init__(self):
        super().__init__()
        self._prePerf = []
        self._gnnPerf = []
        self._postPerf = []
        self._evtPerHit = []
        self._hits = []

    def initialize(self):

        # Initialize GNN

        if args.tracking_algorithm == "int":
            self.interface = Belle2.CATFinderInterface()

    def event(self):
        preStartTime = perf_counter_ns()
        
        if args.tracking_algorithm == "int":

            self.interface.preprocess()

            list_cdchit_x_temp = self.interface.getCDCHitXTemp()
            list_cdchit_y_temp = self.interface.getCDCHitYTemp()
            list_cdchit_tdc_temp = self.interface.getCDCHitTdcTemp()
            list_cdchit_adc_temp = self.interface.getCDCHitAdcTemp()
            list_cdchit_sl_temp = self.interface.getCDCHitSlTemp()
            list_cdchit_l_temp = self.interface.getCDCHitLTemp()
            list_cdchit_index = np.arange(len(self.cdchits)).tolist()
        
        elif args.tracking_algorithm == "cat":
            # Perform preprocessing in Python
            pass

        preStopTime = perf_counter_ns()
        self._prePerf.append(np.round((preStopTime - preStartTime) / 1e6, 2))
        gnnStartTime = perf_counter_ns()

        # Run the GNN

        gnnStopTime = perf_counter_ns()
        self._gnnPerf.append(np.round((gnnStopTime - gnnStartTime) / 1e6, 2))
        postStartTime = perf_counter_ns()

        if args.tracking_algorithm == "int":
            self.interface.postprocess(con_point_x.tolist(), con_point_y.tolist(), con_point_z.tolist(), 
                                       con_point_px.tolist(), con_point_py.tolist(), con_point_pz.tolist(), 
                                       con_point_vx.tolist(), con_point_vy.tolist(), con_point_vz.tolist(), 
                                       coord_x.tolist(), coord_y.tolist(), coord_z.tolist(), 
                                       list_cdchit_index, hit_distance)

        elif args.tracking_algorithm == "cat":            
            # Perform postprocessing in Python
            pass

        postStopTime = perf_counter_ns()
        self._postPerf.append(np.round((postStopTime - postStartTime) / 1e6, 2))
        
        self._evtPerHit.append(np.round((self._prePerf[-1] + self._gnnPerf[-1] + self._postPerf[-1]) / len(self.cdchits), 2))
        self._hits.append(len(self.cdchits))

    def terminate(self):
        meanPreTime = np.mean(self._prePerf[1:])
        stdPreTime = np.std(self._prePerf[1:])

        meanGnnTime = np.mean(self._gnnPerf[1:])
        stdGnnTime = np.std(self._gnnPerf[1:])

        meanPostTime = np.mean(self._postPerf[1:])
        stdPostTime = np.std(self._postPerf[1:])

        meanPerHit = np.mean(self._evtPerHit[1:])
        stdPerHit = np.std(self._evtPerHit[1:])

        meanHits = np.mean(self._hits[1:])
        stdHits = np.std(self._hits[1:])

        with open(args.output_file, 'a') as file:
            # Save runtimes and standard deviations of preprocessing, GNN, postprocessing and whole event 
            # (whole event for Legendre only)
            file.write(f'\npgpe:{np.round(meanPreTime, 3)},{np.round(stdPreTime, 3)},
                       {np.round(meanGnnTime, 3)},{np.round(stdGnnTime, 3)},
                       {np.round(meanPostTime, 3)},{np.round(stdPostTime, 3)},
                       0,0')
            # Save the runtimes per CDC hit
            file.write(f'\nperHitRuntime:{meanPerHit},{stdPerHit},{meanHits},{stdHits}')


basf2.set_random_seed('473d7db24c7e3b1cef1d032d9c44756d30cdae82feba45a3d32c4353db93edb')
basf2.B2INFO(f"Running simulation with {args.event_type}-events using {args.tracking_algorithm}-tracking. Backround: {args.background}.")

main = basf2.Path()

with open(args.output_file, 'w') as file:
    file.write('')

if args.background == 'high':
    main.add_module('EventInfoSetter', expList=0, runList=0)
else:
    main.add_module('EventInfoSetter', expList=1003, runList=0)

main.add_module('Progress')

if args.event_type == "mumu":
    ge.add_kkmc_generator(main, 'mu-mu+')
elif args.event_type == "tautau":
    ge.add_kkmc_generator(main, 'tau-tau+')
elif args.event_type == "BBbar":
    ge.add_evtgen_generator(main, 'mixed')
elif args.event_type == "BB":
    ge.add_evtgen_generator(main, 'charged')

if args.background == 'none':
    si.add_simulation(main)
elif args.background == 'low':
    si.add_simulation(main, bkgfiles=glob.glob('/path/to/low/beam-background')) # Low beam-background
elif args.background == 'high':
    si.add_simulation(main, bkgfiles=glob.glob('/path/to/high/beam-background')) # High beam-background

if args.tracking_algorithm == "cat" or args.tracking_algorithm == "int":
    append_cdc_preliminary_modules(main)
    main.add_module(CATFinder())
elif args.tracking_algorithm == "leg":
    main.add_module('StatisticsSummary', storeIntoDatastore=True).set_name('LegendreSummary')
    add_cdc_track_finding(main)
    main.add_module('StatisticsSummary', storeIntoDatastore=True).set_name('LegendreSummary')

main.add_module('DAFRecoFitter')
main.add_module('TrackCreator')
add_mc_matcher(main)

main.add_module(RecoTrackInfo(10))

basf2.process(main)
print(basf2.statistics)

stats = str(basf2.statistics).split("\n")

# Save memory consumption
if args.tracking_algorithm == "cat" or args.tracking_algorithm == "int":
    mem = ""
    for line in stats:
        if line.startswith('CATFinder'):
            mem = line.split('|')[2].strip()
            break
    with open(args.output_file, 'a') as file:
        file.write(f'\nmem:{mem},0')

elif args.tracking_algorithm == "leg":    
    legLine = ''

    for line in stats:
        if line.startswith('LegendreSummary'):
            legLine = line

    memory = int(legLine.split('|')[2].strip())
    runtime = float(legLine.split('|')[4].strip().split('+-')[0].strip())
    runtimeStd = float(legLine.split('|')[4].strip().split('+-')[1].strip())

    with open(args.output_file, 'a') as file:
        # Also save event runtime and standard deviation for Legendre
        file.write(f'\npgpe:0,0,0,0,0,0,{runtime},{runtimeStd}')
        file.write(f'\nmem:{memory},0')
