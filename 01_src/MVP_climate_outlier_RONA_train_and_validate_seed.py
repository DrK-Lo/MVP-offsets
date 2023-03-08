"""Calculate and validate RONA to climate outlier scenarios for a specific seed.

Usage
-----
conda activate mvp_env
python MVP_climate_outlier_RONA_trainer.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
seed - str
    - the simulation seed to train and validate on climate outlier scenarios
    
Notes
-----
- called from MVP_20_climate_outlier_train_and_validate_RONA.py
"""
from pythonimports import *

import MVP_05_train_RONA as mvp05
from MVP_05_train_RONA import get_rona_elements, calc_rona_elements
import MVP_06_validate_RONA as mvp06
import MVP_15_climate_outlier_validate_GF as mvp15
import MVP_summary_functions as mvp

get_rona_elements.__module__ = '__main__'
calc_rona_elements.__module__ = '__main__'


def validate(seed, fitness, marker_sets=['all', 'adaptive', 'neutral']):
    print(ColorText('\nValidating RONA predictions ...').bold().custom('gold'))
    validation = pd.DataFrame(columns=['seed', 'marker_set', 'program', 'outlier_clim', 'score'])
    for marker_set in pbar(marker_sets):
        rona = pklload(op.join(new_rona_outdir, f'{seed}_{marker_set}_RONA_results.pkl'))

        for env, rona_dict in rona.items():
            rona_offset = pd.DataFrame(rona_dict).T

            scores = rona_offset.corrwith(fitness, axis=1, method='kendall')

            for outlier_clim, score in scores.items():
                validation.loc[nrow(validation), : ] = (seed, marker_set, 'RONA', outlier_clim, score)
            
    return validation


def get_garden_files():
    print(ColorText('\nGetting climate outlier scenarios ...').bold().custom('gold'))
    garden_files = fs(garden_dir)

    popenvdata = pd.DataFrame(columns=['sal_opt', 'temp_opt'])
    for f in garden_files:
        outlier_clim = op.basename(f).rstrip('.txt').split("_")[-1]

        df = pd.read_table(f, index_col=0)
        sal, temp = df.iloc[0][['sal_opt', 'temp_opt']]

        popenvdata.loc[str(float(outlier_clim)), :] = sal, temp

    return popenvdata


def main():
    # get future environmental data for climate outlier scenarios
    popenvdata = get_garden_files()
    dview['popenvdata'] = popenvdata

    # get results of linear models trained across landscape from MVP_05_train_RONA.py
    linear_models = pklload(op.join(rona_outdir, f'{seed}_linear_model_results.pkl'))
    dview['results'] = linear_models

    # determine which linear models were significant
    sig_models = mvp05.retrieve_significant_models(linear_models)
    dview['sig_models'] = sig_models
    mvp05.sig_models = sig_models

    # calculate elements of RONA in parallel
    freqfile = op.join(mvp05.rona_training_dir, f'{seed}_Rout_Gmat_sample_maf-gt-p01_RONAready_pooled_all.txt')
    rona_elements = mvp05.scatter_rona_elements(freqfile, lview, dview)

    print(ColorText('\nCalculating RONA ...').bold().custom('gold'))
    for marker_set in ['all', 'adaptive', 'neutral']:
        mvp05.calculate_rona(rona_elements, sig_models, marker_set)

    # load fitness of pops to climate outlier scenarios
    fitness = mvp15.get_fitness([seed])[seed]

    # validate fitness predictions and annotate seeds with simulation metadata
    validation = mvp15.annotate_seeds(validate(seed, fitness))

    # save
    vfile = op.join(new_corr_dir, f'{seed}_climate_outlier_validation.txt')
    validation.to_csv(vfile, sep='\t', index=False, header=True)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir, seed = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    run = op.basename(outerdir)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    garden_dir = op.join(op.dirname(outlier_outerdir), 'garden_files')
    rona_outdir = op.join(outerdir, 'RONA/training/training_outfiles')

    new_corr_dir = makedir(op.join(outlier_outerdir, 'RONA/validation/corrs'))
    new_rona_outdir = makedir(op.join(outlier_outerdir, 'RONA/rona_results'))

    mvp05.rona_outdir = new_rona_outdir  # where to save rona files in mvp05.calculate_rona
    mvp05.rona_training_dir = op.join(outerdir, 'RONA/training/training_files')
    mvp05.seed = seed

    lview, dview, cluster_id = start_engines(n=36, profile=f'climate_outlier_{run}_RONA')
#     lview, dview = get_client(cluster_id='1678128757-4zyr', profile='lotterhos')

    dview['calc_rona_elements'] = calc_rona_elements
    dview['get_rona_elements'] = get_rona_elements

    # timer
    t1 = dt.now()

    main()
