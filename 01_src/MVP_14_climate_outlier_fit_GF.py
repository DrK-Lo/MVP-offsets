"""Set up climate outlier gradient forest fitting runs using a directory processed through the MVP offset pipeline.

Use previously trained gradient forest models to project offset to climate outlier scenarios.

Usage
-----
conda activate mvp_env
python MVP_climate_outlier_01_GF_training.py outerdir outlier_outerdir

Parameters
----------
outerdir - path
    - the path to the --outdir argument given to 00_start_pipeline.py; eg /path/to/run_20220919_225-450
outlier_outerdir - path
    - the directory where climate outlier files and results are to be saved (similar to outerdir)
    - eg /path/to/climate_outlier/run_20220919_225-450

"""
from pythonimports import *

import MVP_02_fit_gradient_forests as mvp02
import MVP_summary_functions as mvp


def set_up_fitting_cmds(new_envfiles, num_expected=225):
    """
    Parameters
    ----------
    num_expected int
        the number of predfiles expected (MVP was run in batches of 225 seeds per batch)
    
    """
    print(ColorText('\nSetting up fitting commands ...').bold().custom('gold'))
    predfiles = fs(preddir, pattern = '_pooled_', endswith='predOut.RDS')

    assert len(predfiles) / 3 == num_expected

    # set up fitting commands as in MVP_02_fitting
    basenames = []
    cmd_args = []
    cmd_files = []
    all_args = []
    for i, predfile in enumerate(pbar(predfiles)):
        seed = op.basename(predfile).split("_")[0]

        trainingfile = predfile.replace("_predOut.RDS", "_training.RDS")
        assert op.exists(trainingfile)

        ind_or_pooled, marker_set = op.basename(predfile)\
                                                .split("GF_training_")[1]\
                                                .split("_gradient_forest")[0]\
                                                .split('_')

        for garden_file in new_envfiles:
            new_env = op.basename(garden_file).split("_")[-1].rstrip(".txt")

            basename = f'{seed}_{ind_or_pooled}_{marker_set}_{new_env}'

            assert basename not in basenames

            basenames.append(basename)

            args = (
                rscript_exe,
                fitting_file,
                trainingfile,
                garden_file,
                predfile,
                basename,
                out_dir
            )

            cmd_args.append(args)
            all_args.append(args)

            # limit the number of fitting jobs per script to be same as sent to MVP_02.py (for mem reqs reasons)
            if len(cmd_args) == 300 or len(all_args) == len(predfiles) * len(new_envfiles):
                num = str(len(cmd_files)).zfill(2)
                cmd_file = op.join(cmd_dir, f'cmd_args_{num}.pkl')
                pkldump(cmd_args, cmd_file)

                cmd_args = []

                cmd_files.append(cmd_file)

    return cmd_files


def create_shfiles(cmd_files):
    shfiles = []
    for cmd_file in cmd_files:
        num = op.basename(cmd_file).split("_")[-1].rstrip('.pkl')

        job = f'{run}_outlier_fitting_{num}'

        shfile = op.join(shdir, f'{job}.sh')

        text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300000M
#SBATCH --output={job}_%j.out
#SBATCH --dependency=afterok:-1
#SBATCH --mail-user=b.lind@northeastern.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7

cd /home/b.lind/code/MVP-offsets/01_src
source $HOME/.bashrc
conda activate mvp_env

python MVP_climate_outlier_GF_fitting.py {cmd_file}

'''

        with open(shfile, 'w') as o:
            o.write(text)

        shfiles.append(shfile)

    return shfiles


def main():

    new_envfiles = fs(op.join(op.dirname(outlier_outerdir), 'garden_files'), endswith='.txt')

    cmd_files = set_up_fitting_cmds(new_envfiles)

    shfiles = create_shfiles(cmd_files)

    pids = sbatch(shfiles)

    create_watcherfile(pids,
                       directory=shdir,
                       watcher_name=f'{op.basename(outlier_outerdir)}_GF_outlier_watcher',
                       end_alert=True,
                       time='6:00:00',
                       ntasks=1,
                       mem='4000',
                       rem_flags=['#SBATCH --nodes=1', '#SBATCH --cpus-per-task=36'],
                       added_text = '\n'.join(['',
                                               'source $HOME/.bashrc',
                                               '',
                                               'conda activate mvp_env',
                                               '',
                                               'cd $HOME/code/MVP-offsets/01_src',
                                               ''
                                               f'python MVP_15_climate_outlier_validate_GF.py {outerdir} {outlier_outerdir}'
                                               ''
                                              ])
                      )

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    thisfile, outerdir, outlier_outerdir = sys.argv

    assert op.basename(outerdir) == op.basename(outlier_outerdir)
    assert outerdir != outlier_outerdir

    # timer
    t1 = dt.now()

    run = op.basename(outerdir)

    # make some dirs
    preddir = op.join(outerdir, 'gradient_forests/training/training_outfiles')
#     garden_dir = op.join(op.dirname(outlier_outerdir), 'garden_files')
    fitting_outdir = makedir(op.join(outlier_outerdir, 'GF/fitting_outfiles'))
    cmd_dir = makedir(op.join(outlier_outerdir, 'GF/cmd_pkls'))
    shdir = makedir(op.join(outlier_outerdir, 'GF/fitting_shfiles'))
    out_dir = makedir(op.join(outlier_outerdir, 'GF/fitting_outfiles'))

    rscript_exe = '/home/b.lind/anaconda3/envs/r35/lib/R/bin/Rscript'    
    fitting_file = op.join(op.dirname(thisfile), 'MVP_gf_fitting_script.R')

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
