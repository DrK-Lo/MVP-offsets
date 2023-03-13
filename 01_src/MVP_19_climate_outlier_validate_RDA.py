"""Validate climate outlier offset predictions for RDA.

Usage
-----
conda activate mvp_env
python MVP_19_climate_outlier_validate_RDA.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
"""
from pythonimports import *

import MVP_15_climate_outlier_validate_GF as mvp15
import MVP_summary_functions as mvp


def check_completed():
    """Make sure all jobs completed."""
    print(ColorText('\nEnsuring completeness ...').bold().custom('gold'))
    catfiles = fs(cat_dir, endswith='_rda_commands.txt')
    shfiles = fs(shdir, endswith='.sh', exclude='watcher')

    assert len(catfiles) == len(shfiles)

    needed_cmds = []
    all_files = []
    for cat in catfiles:
        cmds = read(cat, lines=True)

        for cmd in cmds:
            if 'pooled' not in cmd:  # ignore individual data
                continue

            rscript, script, seed, *args, rds, use_RDA_outliers, ntraits, mvp_dir = cmd.split()

            structcorr = 'structcorr' if 'structcorr' in rds else 'nocorr'

            basename = f'{seed}_pooled_{use_RDA_outliers}_ntraits-{ntraits}_{structcorr}_rda_offset.txt'
            f = op.join(offset_dir, basename)

            assert basename not in all_files
            all_files.append(basename)

            if not op.exists(f):
                needed_cmds.append(cmd)

            if ntraits == '1':
                f = f.replace('ntraits-1', 'ntraits-2')
                assert op.basename(f) not in all_files
                all_files.append(op.basename(f))

                if not op.exists(f):
                    needed_cmds.append(cmd)

    assert len(needed_cmds) == 0, needed_cmds
    
    assert len(all_files) == luni(all_files)
    
    return all_files


def read_file(f):
    import pandas as pd
    
    df = pd.read_table(f, delim_whitespace=True)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(int)
    
    return df


def read_offset_dfs(all_files):
    print(ColorText('\nReading in offset dfs ...').bold().custom('gold'))
    offset_files = fs(offset_dir, endswith='.txt', exclude='_ind_')

    assert len(offset_files) == len(all_files)

    jobs = []
    for f in offset_files:
        jobs.append(
            lview.apply_async(
                read_file, f
            )
        )

    watch_async(jobs)

    offset_dfs = wrap_defaultdict(dict, 4)
    all_args = []
    for i, j in enumerate(pbar(jobs)):
        seed, ind_or_pooled, use_RDA_outliers, ntraits, structcorr, *args = op.basename(offset_files[i]).split("_")

        arg = ' '.join([seed, ind_or_pooled, use_RDA_outliers, ntraits, structcorr])

        assert arg not in all_args
        all_args.append(arg)

        offset_dfs[seed][ind_or_pooled][use_RDA_outliers][ntraits][structcorr] = j.r

    assert len(jobs) == luni(all_args)

    return offset_dfs


def validate(offset_dfs, fitness):
    """Validate offset predictions by correlating with fitness for all pops, or blocks of populations.

    Parameters
    ----------
    offset_dfs - nested dictionary with final values dataframes
    fitness - dict, with key = seed and val = dataframe for fitness of column pop in row environment
    
    Notes
    -----
    - blocks are each 9 pops; in northwest, range center, and southeast
    """
    print(ColorText('\nValidating results ...').bold().custom('gold'))

    # create a dataframe that seaborn can easily use
    validation = pd.DataFrame(
        columns=['seed', 'ind_or_pooled', 'use_RDA_outliers', 'ntraits', 'structcorr', 'outlier_clim', 'score', 'block']
    )
    
    # calculate correlation between offset and fitness, fill in validation dataframe
    for args, offset in unwrap_dictionary(offset_dfs, progress_bar=True):
        seed, ind_or_pooled, use_RDA_outliers, ntraits, structcorr = args  # FYI
        
        # validate using all populations
        score_dict = offset.corrwith(fitness[seed],
                                     axis=1,
                                     method='kendall').to_dict()  # key = outlier_clim, val = correlation

        for outlier_clim, score in score_dict.items():
            validation.loc[nrow(validation), : ] = (*args, outlier_clim, score, 'all')

        # validate with blocks of populations to see effect of climate distance
        for block, pops in mvp.block_pops.items():
            score_dict = offset[pops].corrwith(fitness[seed][pops],
                                               axis=1,
                                               method='kendall').to_dict()  # key = outlier_clim, val = correlation
            
            for outlier_clim, score in score_dict.items():
                validation.loc[nrow(validation), :] = (*args, outlier_clim, score, block)
        
    # add simulation level info to the dataframe
    validation = mvp15.annotate_seeds(validation)

    validation['program'] = 'rda'
    validation['marker_set'] = validation.use_RDA_outliers.map({'TRUE': 'rda_outliers',
                                                                'FALSE': 'all',
                                                                'CAUSAL': 'adaptive',
                                                                'NEUTRAL': 'neutral'
                                                               })

    # save
    f = op.join(validation_dir, 'climate_outlier_validation_scores.txt')
    validation.to_csv(f, sep='\t', index=False, header=True)

    print(f'\nsaved validation scores to : {f}')

    pass


def main():

    all_files = check_completed()

    offset_dfs = read_offset_dfs(all_files)

    fitness = mvp15.get_fitness(offset_dfs.keys())

    validate(offset_dfs, fitness)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')
    
    pass


if __name__ ==  '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv
    
    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    # timer
    t1 = dt.now()

    # get some dirs
    cmd_dir = op.join(outerdir, 'rda/rda_catfiles')
    cat_dir = op.join(outlier_outerdir, 'rda/rda_catfiles')
    shdir = op.join(outlier_outerdir, 'rda/shfiles')
    validation_dir = makedir(op.join(outlier_outerdir, 'rda/validation'))
    offset_dir = op.join(outlier_outerdir, 'rda/offset_outfiles')

    run = op.basename(outerdir)
    lview, dview, cluster_id = start_engines(n=36, profile=f'{run}_RDA')

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    main()    
