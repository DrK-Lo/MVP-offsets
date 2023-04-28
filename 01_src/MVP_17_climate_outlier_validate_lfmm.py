"""Validate climate outlier offset predictions for lfmm2.

Usage
-----
conda activate mvp_env
python MVP_17_climate_outlier_validate_lfmm2.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
"""
from pythonimports import *

import MVP_11_validate_lfmm2_offset as mvp11
import MVP_15_climate_outlier_validate_GF as mvp15
import MVP_summary_functions as mvp


def check_file_counts():
    """Make sure all of the lfmm commands completed."""
    print(ColorText('\nChecking file counts ...').bold().custom('gold'))
    outfiles = fs(new_outdir)

    batchfiles = fs(new_indir, startswith=run, pattern='outlier_batch', endswith='.txt')

    needed = []
    all_cmds = []
    for batch in pbar(batchfiles):
        cmds = read(batch, lines=True)
        all_cmds.extend(cmds)

        for cmd in cmds:
            outfile = cmd.split()[-2]

            if outfile not in outfiles:
                needed.append(cmd)

    assert len(needed) == 0, '\n'.join(needed)
    assert len(all_cmds) == len(outfiles), (len(all_cmds), len(outfiles))

    return outfiles


def read_lfmm_offset_dfs(outfiles):
    """Read in the dataframes containing offset predictions."""
    print(ColorText('\nReading offset dataframes ...').bold().custom('gold'))
    # read in each dataframe (offset for all pops to the outlier_clim)
    offset_series = wrap_defaultdict(dict, 3)
    for outfile in pbar(outfiles):
        seed, *args, ntraits, outlier_clim, marker_set = op.basename(outfile).rstrip('.txt').split("_")
        df = pd.read_table(outfile)
        offset_series[seed][marker_set][ntraits][outlier_clim] = df

    # combine into dataframes of rows = outlier_clim, columns = population source
    offset_dfs = wrap_defaultdict(dict, 2)
    for seed in keys(offset_series):
        for marker_set in keys(offset_series[seed]):
            for ntraits in keys(offset_series[seed][marker_set]):

                offset_cols = []
                for outlier_clim, offset in offset_series[seed][marker_set][ntraits].items():
                    offset = offset.copy()
                    offset.columns = [outlier_clim]
                    offset_cols.append(offset)

                df = pd.concat(offset_cols, axis=1).T
                df.columns = df.columns.astype(int)
                df.index = df.index.astype(float).astype(str)
                offset_dfs[seed][marker_set][ntraits] = df

    # save offsets
    pkl = op.join(new_offset_dir, f'{run}_climate_outlier_offset_dfs.pkl')
    pkldump(offset_dfs, pkl)
    print(f'\n\twrote offset_dfs to : {pkl}')

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
    print(ColorText('\nValidating offset predictions ...').bold().custom('gold'))
    
    validation = pd.DataFrame(columns=['seed', 'marker_set', 'ntraits', 'program', 'outlier_clim', 'score', 'block'])
    for (seed, marker_set, ntraits), offset in unwrap_dictionary(offset_dfs, progress_bar=True):
        # validate using all populations
        score_dict = offset.corrwith(fitness[seed],
                                     axis=1,
                                     method='kendall').to_dict()  # key = outlier_clim, val = correlation

        for outlier_clim, score in score_dict.items():
            validation.loc[nrow(validation), :] = seed, marker_set, ntraits, 'lfmm2', outlier_clim, score, 'all'
        
        # validate with blocks of populations to see effect of climate distance
        for block, pops in mvp.block_pops.items():
            score_dict = offset[pops].corrwith(fitness[seed][pops],
                                               axis=1,
                                               method='kendall').to_dict()  # key = outlier_clim, val = correlation
            
            for outlier_clim, score in score_dict.items():
                validation.loc[nrow(validation), :] = seed, marker_set, ntraits, 'lfmm2', outlier_clim, score, block

    # add simulation level info to the dataframe
    validation = mvp15.annotate_seeds(validation)

    # save
    f = op.join(validation_dir, 'climate_outlier_validation_scores.txt')
    validation.to_csv(f, sep='\t', index=False, header=True)
    
    print(f'\twrote validation to: {f}')

    pass


def main():

    outfiles = check_file_counts()

    offset_dfs = read_lfmm_offset_dfs(outfiles)

    fitness = mvp15.get_fitness(offset_dfs.keys())

    validate(offset_dfs, fitness)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)
    assert outerdir != outlier_outerdir

    run = op.basename(outerdir)
    
    t1 = dt.now()  # notebook timer

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    # get some dirs
    new_indir = op.join(outlier_outerdir, 'lfmm2/lfmm_infiles')
    new_shdir = op.join(outlier_outerdir, 'lfmm2/lfmm_shfiles')
    new_outdir = op.join(outlier_outerdir, 'lfmm2/lfmm_outfiles')
    validation_dir = makedir(op.join(outlier_outerdir, 'lfmm2/validation'))
    new_offset_dir = makedir(op.join(validation_dir, 'offset_dfs'))

    main()
