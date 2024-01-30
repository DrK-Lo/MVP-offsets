"""Validate offset predictions from RDA (sensu Capblancq & Forester) using population mean fitness.

This script is specifically designed for nuisance rda. It runs like MVP_13_RDA_validation except that 
    it overwrites mvp13.retrieve_offset_data so that there is only a multiplication factor of 1 for the
    number of expected files to be found. It also uses partial functions from MVP_13_RDA_validation_shuffle.

Usage
-----
conda activate mvp_env
python MVP_13_RDA_validation.py seed slimdir outerdir

"""
from pythonimports import makedir, ColorText, formatclock

import sys
from datetime import datetime as dt
from os import path as op
from functools import partial

import MVP_06_validate_RONA as mvp06
import MVP_10_train_lfmm2_offset as mvp10
import MVP_13_RDA_validation as mvp13
import MVP_13_RDA_validation_shuffle as mvp13shuffle

def main():
    """Like mvp13.main but a little different to account for nuisance."""
    # get predicted offset files
    offset_dfs = mvp13shuffle.retrieve_offset_data(mvp13.seed, factor=1)
    
    # get a list of subsampled individuals, map samp top subpopID, and get population locations for each subpopID
    subset, locations, envdata = mvp06.get_pop_data(mvp13.slimdir, mvp13.seed)

    # map subpopID to list of samps - key = subpopID val = list of individual sample names
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()

    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))
    
    # get fitness matrices for individuals and pops (pops are mean fitness)
    fitness_mat = mvp13.retrieve_fitness_data(mvp13.slimdir, mvp13.seed, subset)
    
    # calculate validation scores
    performance_dicts = mvp13.calculate_performance(offset_dfs, fitness_mat, popsamps, samppop)
    
    # create figs
    # skip making figs
    
    # DONE!
    print(ColorText('\nDONE!!').bold().green())
    print(ColorText(f'\ttime to complete: {formatclock(dt.now() - t1, exact=True)}\n'))

    pass


if __name__ == '__main__':
    thisfile, mvp13.seed, mvp13.slimdir, mvp13.outerdir = sys.argv
    
    nuis_envs = op.basename(mvp13.outerdir).split("_")[0].split('-')
#     print(nuis_envs)
#     exit()
    
    t1 = dt.now()
    
    # details about demography and selection
    mvp13.params = mvp10.read_params_file(mvp13.slimdir)
    mvp13.level = mvp13.params.loc[mvp13.seed, 'level']
    mvp13.ntraits = 2 + len(nuis_envs)  # sal + temp + nuis_envs
#     mvp13.ntraits, mvp13.level = mvp13.params.loc[mvp13.seed, ['N_traits', 'level']]

    # set globally
    mvp13.norm = mvp13.Normalize(vmin=-1.0, vmax=1.0)
    mvp13.rda_dir = op.join(mvp13.outerdir, 'rda')
    mvp13.rda_outdir = op.join(mvp13.rda_dir, 'offset_outfiles')
#     mvp13.fig_dir = makedir(op.join(mvp13.rda_dir, 'validation/figs'))
#     mvp13.heat_dir = makedir(op.join(mvp13.rda_dir, 'validation/heatmap_textfiles'))
#     mvp13.pkl_dir = makedir(op.join(mvp13.rda_dir, 'validation/pkl_files'))
    mvp13.corr_dir = makedir(op.join(mvp13.rda_dir, 'validation/corrs'))
    mvp13.offset_dir = makedir(op.join(mvp13.rda_dir, 'validation/offset_dfs'))

#     # dict for pretty labels in figures
#     mvp13.label_dict = {
#         'TRUE' : 'RDA outliers',
#         'FALSE' : 'all loci',
#         'CAUSAL' : 'causal loci',
#         'NEUTRAL' : 'neutral loci',
#         'nocorr' : 'no correction',
#         'structcorr' : 'structure-corrected',
#         'sal_opt' : 'sal',
#         'temp_opt' : 'temp'
#     }

#     # background color for figures
#     mvp13.background_cmap = mvp13.create_cmap(['white', 'gold'], grain=1000)

    # add to namespaces
#     mvp06.label_dict = mvp13.label_dict
    mvp06.level = mvp13.level

    # add to mvp13shuffle namespace
    mvp13shuffle.rda_outdir = mvp13.rda_outdir
    mvp13shuffle.offset_dir = mvp13.offset_dir
    mvp13shuffle.ntraits = mvp13.ntraits
    mvp13shuffle.seed = mvp13.seed
#     mvp13shuffle.level = mvp13.level
#     mvp13shuffle.label_dict = mvp13.label_dict
#     mvp13shuffle.heat_dir = mvp13.heat_dir
    
#     # overwrite retrieve_offset_data, create_histo_subplots, create_heatmap_subplots, fill_slope_heatmaps
#     mvp13.retrieve_offset_data = partial(mvp13shuffle.retrieve_offset_data, factor=1)

    main()
