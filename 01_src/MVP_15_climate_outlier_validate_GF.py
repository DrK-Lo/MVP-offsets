"""Validate climate outlier offset predictions for Gradient Forests.

Usage
-----
conda activate mvp_env
python MVP_15_climate_outlier_validate_GF.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
"""
from pythonimports import *

import MVP_10_train_lfmm2_offset as mvp10  # for session_info.show()
import MVP_summary_functions as mvp


def get_offsets():
    """Read in outlier climate scenario offsets."""
    print(ColorText('\nRetrieving offset dataframes ...').bold().custom('gold'))

    fitdir = op.join(outlier_outerdir, 'GF/fitting_outfiles')

    offset_files = fs(fitdir, endswith='offset.txt')

    # read in dataframes
    gf_offsets = wrap_defaultdict(list, 2)
    for f in pbar(offset_files, desc='reading offsets'):
        seed, ind_or_pooled, marker_set, outlier_clim, *_ = op.basename(f).split("_")
        
        df = pd.read_table(f).T
        df.index = [str(float(outlier_clim))]
        
        gf_offsets[seed][marker_set].append(df)
        
    # combine into dataframe so outlier_clim is rows for each seed-marker_set combo
    for seed in keys(gf_offsets):
        for marker_set in keys(gf_offsets[seed]):
            gf_offsets[seed][marker_set] = pd.concat(gf_offsets[seed][marker_set])

    return gf_offsets


def get_fitness(seeds):
    # fitness_dir created in 02.04.01_calculate_climate_outlier_fitness.ipynb
    print(ColorText('\nReading fitness matrices ...').bold().custom('gold'))

    fitness = {}
    for seed in pbar(seeds):

        dfs = []
        for f in fs(fitness_dir, startswith=seed, endswith='.txt'):
            seed, outlier_clim = op.basename(f).replace('.txt', '').split("_")
            df = pd.read_table(f)  # 1-row, npop columns
            df.index = [outlier_clim]
            df.columns = df.columns.astype(int)
            dfs.append(df)

        fitness[seed] = pd.concat(dfs)

    return fitness


def annotate_seeds(df):
    """Annotate seed with simulation level info as in create_level_df() from ipynbs in 02.01.00_save_level_scores_replicates."""
    from pythonimports import pbar
    from collections import defaultdict
    import MVP_10_train_lfmm2_offset as mvp10
    import numpy as np

    params = mvp10.read_params_file('/work/lotterhos/MVP-NonClinalAF/src')

    seed_dict = defaultdict(dict)
    for seed in pbar(params.index):
        # get simulation parameters
        glevel, plevel, _blank_, landscape, popsize, *migration = params.loc[seed, 'level'].split("_")
        migration = '-'.join(migration)  # for m_breaks to m-breaks (otherwise m-constant)

        assert _blank_ == ''

        # level of pleiotropy and selection
        if plevel != '1-trait':
            num, trait_str, *pleio, equality, S_str = plevel.split('-')
            plevel = '2-trait'
            pleio = ' '.join(pleio)
            slevel = f'{equality}-S'
        else:
            pleio = 'no pleiotropy'
            slevel = np.nan

        for sublevel, subval in zip(['glevel', 'plevel', 'pleio', 'slevel', 'landscape', 'popsize', 'migration'],
                                    [glevel,    plevel,   pleio,   slevel,   landscape,   popsize,   migration]):
            
            seed_dict[sublevel][seed] = subval

    for sublevel, subdict in seed_dict.items():
        df[sublevel] = df.seed.map(subdict)

    return df


# def validate(gf_offsets, fitness):
#     """Validate offset predictions by correlating with fitness for all pops, or blocks of populations.

#     Parameters
#     ----------
#     offset_dfs - nested dictionary with final values dataframes
#     fitness - dict, with key = seed and val = dataframe for fitness of column pop in row environment
    
#     Notes
#     -----
#     - blocks are each 9 pops; in northwest, range center, and southeast
#     """
#     print(ColorText('\nValidating results ...').bold().custom('gold'))

#     # calculate correlation between offset and fitness
#     validation_dict = defaultdict(dict)
#     block_dict = wrap_defaultdict(dict, 2)
    
#     # create a dataframe that seaborn can easily use for full validation and block validation
#     validation = pd.DataFrame(columns=['seed', 'marker_set', 'program', 'outlier_clim', 'score', 'block'])
#     for (seed, marker_set, outlier_clim), offset in unwrap_dictionary(gf_offsets, progress_bar=True):

#         score_dict = fitness[seed].corrwith(  # key = outlier_clim, val = kendall tau
#             offset['offset'],
#             axis=1,
#             method='kendall'
#         ).to_dict()

#         for outlier_clim, score in score_dict.items():
# #             new_row = pd.DataFrame((seed, marker_set, 'GF', outlier_clim, score, 'all'), index=validation.columns).T
# #             validation = validation.append(new_row)            
#             validation.loc[nrow(validation), : ] = (seed, marker_set, 'GF', outlier_clim, score, 'all')

#         # validate with blocks of populations to see effect of climate distance
#         for block, pops in mvp.block_pops.items():
#             score_dict = fitness[seed][pops].corrwith(offset['offset'].loc[pops],
#                                                       axis=1,
#                                                       method='kendall').to_dict()  # key = outlier_clim, val = correlation

#             for outlier_clim, score in score_dict.items():
# #                 new_row = pd.DataFrame((seed, marker_set, 'GF', outlier_clim, score, block), index=validation.columns).T
# #                 validation = validation.append(new_row)
#                 validation.loc[nrow(validation), :] = (seed, marker_set, 'GF', outlier_clim, score, block)

#     # add simulation level info to the dataframe
#     validation = annotate_seeds(validation)

#     # save
#     f = op.join(validation_dir, 'climate_outlier_validation_scores.txt')
#     validation.to_csv(f, sep='\t', index=False, header=True)

#     print(f'\nsaved validation scores to : {f}')

#     pass


def calc_validation(seed):
    """Calculate validation scores for a given simulation seed."""
    import pandas as pd
    from pythonimports import unwrap_dictionary, nrow
    import MVP_summary_functions as mvp
    
    # create a dataframe that seaborn can easily use for full validation and block validation
    validation = pd.DataFrame(columns=['seed', 'marker_set', 'program', 'outlier_clim', 'score', 'block'])
    for marker_set, offset in gf_offsets[seed].items():

        score_dict = fitness[seed].corrwith(  # key = outlier_clim, val = kendall tau
            offset,
            axis=1,
            method='kendall'
        ).to_dict()

        for outlier_clim, score in score_dict.items():
            validation.loc[nrow(validation), : ] = (seed, marker_set, 'GF', outlier_clim, score, 'all')

        # validate with blocks of populations to see effect of climate distance
        for block, pops in mvp.block_pops.items():
            score_dict = fitness[seed][pops].corrwith(  # key = outlier_clim, val = correlation
                offset[pops],
                axis=1,
                method='kendall'
            ).to_dict()

            for outlier_clim, score in score_dict.items():
                validation.loc[nrow(validation), :] = (seed, marker_set, 'GF', outlier_clim, score, block)

    # add simulation level info to the dataframe
    validation = annotate_seeds(validation)
    
    return validation


def validate(gf_offsets, fitness):
    """Parallel implementation to calculate validation score (waaaay faster!!! 6 hrs vs < 1 min).
    
    Parameters
    ----------
    offset_dfs - nested dictionary with final values dataframes
    fitness - dict, with key = seed and val = dataframe for fitness of column pop in row environment
  
    Notes
    -----
    - blocks are each 9 pops; in northwest, range center, and southeast
    """
    # load data to engines
    dview['gf_offsets'] = gf_offsets
    dview['fitness'] = fitness
    dview['annotate_seeds'] = annotate_seeds
    
    sleeping(5)  # shhhhhhh baby's sleeping!
    
    # validate in parallel
    jobs = []
    for seed in keys(gf_offsets):
        jobs.append(
            lview.apply_async(calc_validation, seed)
        )
    
    watch_async(jobs)
    
    # retrieve dataframes
    dfs = []
    for j in jobs:
        dfs.append(j.r)
        
    validation = pd.concat(dfs)
    
    # save
    f = op.join(validation_dir, 'climate_outlier_validation_scores.txt')
    validation.to_csv(f, sep='\t', index=False, header=True)

    print(f'\nsaved validation scores to : {f}')
    
    pass


def main():
    # retrieve saved offsets
    gf_offsets = get_offsets()

    # retrieve fitness information for new climates
    fitness = get_fitness(gf_offsets.keys())

    # calculate correlation between offset and fitness
    validate(gf_offsets, fitness)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)
    assert outerdir != outlier_outerdir
    
    # created in 02.04.01 or 02.04.06
    fitness_dir = op.join(op.dirname(outlier_outerdir), 'fitness_mats')

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    # timer
    t1 = dt.now()

    # get some dirs
    garden_dir = op.join(outlier_outerdir, 'GF/garden_files')  # created in MVP_14_climate_outlier_fit_GF.py
    validation_dir = makedir(op.join(outlier_outerdir, 'GF/validation'))

    lview, dview, cluster_id = start_engines(n=36, profile=f'climate_outlier_{op.basename(outlier_outerdir)}')

    main()
