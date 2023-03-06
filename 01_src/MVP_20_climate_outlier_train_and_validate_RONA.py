"""Call on MVP_climate_outlier_RONA_train_and_validate_seed.py.

Usage
-----
python MVP_climate_outlier_RONA_train_and_validate_seed.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450
"""
from pythonimports import *

import MVP_00_start_pipeline as mvp00
import MVP_summary_functions as mvp


def create_shfiles(seeds):
    print(ColorText('\nCreating sh files ...').bold().custom('gold'))
    
    shfiles = []
    for seed in seeds:
        job = f'{seed}_climate_outlier_RONA'
        shfile = op.join(shdir, f'{job}.sh')
        
        text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4000M
#SBATCH --output={job}_%j.out
#SBATCH --dependency=afterok:-1
#SBATCH --mail-user=b.lind@northeastern.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36

cd /home/b.lind/code/MVP-offsets/01_src
source $HOME/.bashrc
conda activate mvp_env

python MVP_climate_outlier_RONA_train_and_validate_seed.py {outerdir} {outlier_outerdir} {seed}

'''
        with open(shfile, 'w') as o:
            o.write(text)
            
        shfiles.append(shfile)

    return shfiles


def main():
    seeds = mvp00.get_seeds(op.join(outerdir, 'slimdir'))

    shfiles = create_shfiles(seeds)
    
    pids = sbatch(shfiles)
    
    create_watcherfile(pids,
                       directory=shdir,
                       watcher_name=f'{run}_RONA_outlier_watcher',
                       end_alert=True,
                       time='1:00:00',
                       ntasks=1,
                       mem='4000',
                       added_text = '\n'.join(['',
                                               'source $HOME/.bashrc',
                                               '',
                                               'conda activate mvp_env',
                                               '',
                                               'cd $HOME/code/MVP-offsets/01_src',
                                               ''
                                               f'python MVP_21_climate_gather_RONA.py {outerdir} {outlier_outerdir}'
                                               ''
                                              ])
                      )

if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    run = op.basename(outerdir)
    
    shdir = makedir(op.join(outlier_outerdir, 'RONA/shfiles'))
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()
