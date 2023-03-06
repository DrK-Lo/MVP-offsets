"""Gather across seeds from output from MVP_climate_outlier_RONA_train_and_validate_seed.py.

Usage
-----
python MVP_21_climate_gather_RONA.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
"""
from pythonimports import *

import MVP_summary_functions as mvp

def main():
    validation_files = fs(corr_dir, endswith='.txt')

    print(ColorText('\nReading in validation files for each seed ...').bold().custom('gold'))
    dfs = []
    for f in pbar(validation_files):
        df = pd.read_table(f)
        dfs.append(df)

    all_seeds = pd.concat(dfs)

    allseeds_file = op.join(validation_dir, 'RONA_climate_outlier_validation_scores.txt')
    all_seeds.to_csv(allseeds_file, sep='\t', index=False)

    print(f'\tsaved to: {allseeds_file}')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    run = op.basename(outerdir)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)
    
    validation_dir = op.join(outlier_outerdir, 'RONA/validation')
    corr_dir = op.join(validation_dir, 'corrs')

    main()
