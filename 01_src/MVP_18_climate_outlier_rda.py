"""Set up scripts to work with RDA scripts created in the MVP offset pipeline.

Usage
-----
conda activate mvp_env
python MVP_climate_outlier_05_rda.py outerdir outlier_outerdir

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


def create_commands():
    """Use previously created RDA commands and update script and args, save to new files."""
    print(ColorText('\nCreating new RDA commands ...').bold().custom('gold'))
    
    rda_cmd_files = fs(cmd_dir, endswith='rda_commands.txt')
    assert len(rda_cmd_files) == 225

    cmd_files = []
    for f in rda_cmd_files:
        cmds = read(f, lines=True)

        new_cmds = []
        for cmd in cmds:
            new_cmd = cmd.split()
            assert len(new_cmd) == 8

            # change the R script used
            assert new_cmd[1] == 'MVP_12_RDA_offset.R'
            new_cmd[1] = 'MVP_climate_outlier_RDA_offset.R'

            # change the outer directory for saving and lookups (remember there are symlinks in outlier_outerdir to outerdir)
            assert op.realpath(new_cmd[4]) == op.realpath(outerdir), f'{new_cmd[4] = } {outerdir = }'
            new_cmd[4] = outlier_outerdir

            new_cmd.append('/home/b.lind/code/MVP-offsets/01_src')  # needed for MVP_climate_outlier_RDA_offset.R

            new_cmds.append(' '.join(new_cmd))

        outlier_f = op.join(cat_dir, op.basename(f))

        assert outlier_f != f
        
        cmd_files.append(outlier_f)

        with open(outlier_f, 'w') as o:
            o.write('\n'.join(new_cmds))

    return cmd_files


def create_links():
    """Symlink original outerdir subdirs to outlier_outerdir for needed input files."""
    subdirs = [
        ('gradient_forests/training', 'training_files'),  # 'gradient_forests/training/training_files',
        ('pca', 'mutfiles'), # 'pca/mutfiles',
        ('pca', 'pca_output')  # 'pca/pca_output',
    ]

    for create, link in subdirs:
        # make a subdir 
        subdir = makedir(op.join(outlier_outerdir, create))

        src = op.join(outerdir, op.join(create, link))
        assert op.exists(src), src

        dst = op.join(subdir, link)

        if not op.exists(dst):
            os.symlink(src, dst)

    pass


def create_shfiles(cmd_files):
    print(ColorText('\nCreating shfiles ...').bold().custom('gold'))
    shfiles = []
    for cmd_file in cmd_files:
        job = f'{run}_' + op.basename(cmd_file).rstrip('.txt')

        text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=1:00:00
#SBATCH --mem=9000
#SBATCH --partition=lotterhos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output={job}_%j.out
#SBATCH --mail-user=b.lind@northeastern.edu
#SBATCH --mail-type=FAIL

source $HOME/.bashrc

module load parallel

conda activate MVP_env_R4.0.3

cd /home/b.lind/code/MVP-offsets/01_src

# run RDA offset estimation
cat {cmd_file} | parallel -j 16 --progress --eta

'''

        shfile = op.join(shdir, f'{job}.sh')

        with open(shfile, 'w') as o:
            o.write(text)

        shfiles.append(shfile)

    return shfiles    


def main():

    # create commands to use previously created files with new outlier climates
    cmd_files = create_commands()

    # create slurm bash files
    shfiles =  create_shfiles(cmd_files)

    print(ColorText('\nsbatching shfiles ...').bold().custom('gold'))
    pids = sbatch(shfiles)

    # get an alert when they're all done
    create_watcherfile(pids,
                       directory=shdir,
                       watcher_name=f'{op.basename(outlier_outerdir)}_rda_outlier_watcher',
                       end_alert=True,
                       time='1:00:00',
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
                                               f'python MVP_19_climate_outlier_validate_RDA.py {outerdir} {outlier_outerdir}'
                                               ''
                                              ])
                      )
    

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':

    thisfile, outerdir, outlier_outerdir = sys.argv
    
    run = op.basename(outerdir)
    
    assert op.basename(outerdir) == op.basename(outlier_outerdir)

    # timer
    t1 = dt.now()

    create_links()

    # make some dirs
    cmd_dir = op.join(outerdir, 'rda/rda_catfiles')
    cat_dir = makedir(op.join(outlier_outerdir, 'rda/rda_catfiles'))
    shdir = makedir(op.join(outlier_outerdir, 'rda/shfiles'))

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
