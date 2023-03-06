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

import MVP_10_train_lfmm2_offset as mvp10
import MVP_summary_functions as mvp


def get_offsets():
    print(ColorText('\nRetrieving offset dataframes ...').bold().custom('gold'))
    gf_offsets = wrap_defaultdict(None, 3)

    fitdir = op.join(outlier_outerdir, 'GF/fitting_outfiles')

    offset_files = fs(fitdir, endswith='offset.txt')

    for f in pbar(offset_files, desc='reading offsets'):
        seed, ind_or_pooled, marker_set, outlier_clim, *_ = op.basename(f).split("_")

        gf_offsets[seed][marker_set][outlier_clim] = pd.read_table(f)

    return gf_offsets


def get_fitness(seeds, fitness_dir='/home/b.lind/offsets/climate_outlier_runs/fitness_mats'):
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
    print(ColorText('\nAnnotating seeds ...').bold().custom('gold'))

    params = mvp10.read_params_file('/work/lotterhos/MVP-NonClinalAF/src')

    for row in pbar(df.index):
        seed = df.loc[row, 'seed']

        # get simulation parameters
        glevel, plevel, _blank_, landscape, popsize, *migration = params.loc[seed, 'level'].split("_")
        migration = '-'.join(migration)  # for m_breaks to m-breaks (otherwise m-constant)

        assert _blank_ == ''

        # level of pleiotropy and selection
    #     pleio = 'no pleiotropy' if '-no-' in plevel else 'pleiotropy'
        if plevel != '1-trait':
            num, trait_str, *pleio, equality, S_str = plevel.split('-')
            plevel = '2-trait'
            pleio = ' '.join(pleio)
            slevel = f'{equality}-S'
        else:
            pleio = 'no pleiotropy'
            slevel = np.nan

        df.loc[
            row,
            ['glevel', 'plevel', 'pleio', 'slevel', 'landscape', 'popsize', 'migration']# , 'marker_set', 'seed']
        ] = [glevel,    plevel,   pleio,   slevel,   landscape,   popsize,   migration]# ,   marker_set,   seed]

    return df


def validate(gf_offsets, fitness):
    print(ColorText('\nValidating results ...').bold().custom('gold'))

    # calculate correlation between offset and fitness
    validation_dict = wrap_defaultdict(None, 3)
    for (seed, marker_set, outlier_clim), offset in unwrap_dictionary(gf_offsets):

        for outlier_clim in fitness[seed].index:
            validation_dict[seed][marker_set][outlier_clim] = offset['offset'].corr(fitness[seed].loc[outlier_clim],
                                                                              method='kendall')

    # for each seed create a dataframe with index for each climate scenario (outlier_clim) and columns for each marker set
    marker_dfs = {}
    for seed, marker_dict in validation_dict.items():
        marker_dfs[seed] = pd.DataFrame(marker_dict)

    # create a dataframe that seaborn can easily use
    validation = pd.DataFrame(columns=['seed', 'marker_set', 'program', 'outlier_clim', 'score'])
    for seed, marker_df in pbar(marker_dfs.items()):
        df = marker_df.copy()

        for marker_set in df.columns:
            for outlier_clim in df.index:
                validation.loc[nrow(validation), :] = (seed, marker_set, 'GF', outlier_clim, df.loc[outlier_clim, marker_set])

    # add simulation level info to the dataframe
    validation = annotate_seeds(validation)

    # save
    f = op.join(validation_dir, 'climate_outlier_validation_scores.txt')
    validation.to_csv(f, sep='\t', index=False, header=True)

    print(f'\nsaved validation scores to : {f}')

    pass


def main():

    gf_offsets = get_offsets()

    fitness = get_fitness(gf_offsets.keys())

    validate(gf_offsets, fitness)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    # timer
    t1 = dt.now()

    # get some dirs
    garden_dir = op.join(outlier_outerdir, 'GF/garden_files')  # created in MVP_14_climate_outlier_fit_GF.py
    validation_dir = makedir(op.join(outlier_outerdir, 'GF/validation'))

    # created in 02.04.01_calculate_climate_outlier_fitness.ipynb:
    fitness_dir = '/home/b.lind/offsets/climate_outlier_runs/fitness_mats'

    lview, dview, cluster_id = start_engines(n=36, profile=f'climate_outlier_{op.basename(outlier_outerdir)}')

    main()
