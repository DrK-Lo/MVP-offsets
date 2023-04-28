"""set up scripts to work with MVP_process_lfmm.R scripts created in the MVP offset pipeline.

Usage
-----
conda activate mvp_env
python MVP_climate_outlier_03_lfmm.py outerdir outlier_outerdir

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


def create_cmds():
    """Create commands for MVP_process_lfmm.R.
    
    # one command for each seed * new_env * ntraits * marker_set
    """
    print(ColorText('\nCreating commands ...').bold().custom('gold'))

    args = []
    outputfiles = []
    new_envfiles = []
    for outdir in pbar(outdirs):
        seed = op.basename(outdir)

        poplabel_file = op.join(indir, f'{seed}_poplabels.txt')

        adaptive_file, neutral_file = sorted(fs(indir, startswith=f'{seed}_', endswith='loci.txt'))
        assert 'adaptive' in adaptive_file
        assert 'neutral' in neutral_file

        envfiles = fs(indir, startswith=seed, pattern='_env_', exclude=['_0', '_100'])
        assert len(envfiles) in [1, 2]

        for new_env in mvp.new_envs:
            for envfile in envfiles:
                envdata = pd.read_table(envfile, header=None)

                ntraits = op.basename(envfile).split("_")[-1].rstrip('.txt')

                if ntraits == 'ntraits-1':
                    envdata[0] = new_env
                elif ntraits == 'ntraits-2':
                    envdata[0] = -1 * new_env
                    envdata[1] = new_env
                else:
                    raise Exception(f'unexpected result: {ntraits = }')

                new_envfile = op.join(new_indir, f'{seed}_lfmm_newenv_{ntraits}_{new_env}.txt')
                assert new_envfile not in new_envfiles
                new_envfiles.append(new_envfile)

                envdata.to_csv(new_envfile, sep='\t', index=False, header=False)

                for cand_file, marker_set in zip(['None', adaptive_file, neutral_file], ['all', 'adaptive', 'neutral']):

                    outputfile = op.join(new_outdir, f'{seed}_lfmm_offsets_{ntraits}_{new_env}_{marker_set}.txt')
                    assert outputfile not in outputfiles
                    outputfiles.append(outputfile)

                    args.append(
                        ['Rscript',
                         training_script,
                         seed,
                         slimdir,
                         envfile,
                         new_envfile,
                         poplabel_file,
                         outputfile,
                         cand_file]
                    )
                    
    # all commands should be unique
    assert len(args) == luni([' '.join(a) for a in args])
    
    return args


def create_shfiles(args):
    """Create batch files - each batch file gets at most 64 commands."""
    print(ColorText('\nCreating shfiles ...').bold().custom('gold'))
    
    # create shfiles too - each shfile is written similarly to those created in mvp10.create_shfiles
    shfiles = []
    batch_cmds = []
    for i, cmd in enumerate(args):
        batch_cmds.append(' '.join(cmd))

        if len(batch_cmds) == 64 or (i + 1) == len(args):  # double num batch_cmds per batch
            batch_num = str(len(shfiles)).zfill(3)

            job = f'{run}_climate_outlier_batch_{batch_num}'

            batch_file = op.join(new_indir, f'{job}.txt')
            with open(batch_file, 'w') as o:
                o.write('\n'.join(batch_cmds))

            shfile = op.join(new_shdir, f'{job}.sh')  # increase time and mem compared to mvp10.create_shfiles
            text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=1-00:00:00
#SBATCH --mem=200000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output={job}_%j.out
#SBATCH --mail-user=b.lind@northeastern.edu
#SBATCH --mail-type=FAIL

source $HOME/.bashrc  # assumed that conda init is within .bashrc

module load parallel

conda activate MVP_env_R4.0.3

# train lfmm2 and calculate offset
cat {batch_file} | parallel -j 32 --progress --eta

conda activate mvp_env

cd /home/b.lind/code/MVP-offsets/01_src

# re run any failed jobs from the cmd_file (seed arg is arbitrary)
python MVP_watch_for_failure_of_train_lfmm2_offset.py 9999999 {shfile} {outerdir} 32

    '''
            with open(shfile, 'w') as o:
                o.write(text)
            shfiles.append(shfile)

            batch_cmds = []

    len(shfiles)
    
    return shfiles

                    
def main():
    # create commands to use previously created files with new outlier climates
    args = create_cmds()
    
    # create slurm bash files
    shfiles = create_shfiles(args)
    
    print(ColorText('\nsbatching shfiles ...').bold().custom('gold'))
    pids = sbatch(shfiles)
    
    # get an alert when they're all done
    create_watcherfile(pids,
                       directory=new_shdir,
                       end_alert=True,
                       watcher_name=f'{run}_lfmm_outlier_watcher',
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
                                               f'python MVP_17_climate_outlier_validate_lfmm.py {outerdir} {outlier_outerdir}'
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
    
    run = op.basename(outerdir)

    # timer
    t1 = dt.now()
    
    run = op.basename(outerdir)
    
    # make some dirs
    mvp10.seed = op.basename(fs(op.join(outerdir, 'slimdir'), endswith='_fitnessmat.txt')[0]).split("_")[0]  # arbitrary seed
    indir, outdir, shdir = mvp10.make_lfmm_dirs(outerdir)
    del mvp10.seed
    outdirs = fs(op.dirname(outdir), dirs=True); assert len(outdirs) == 225  # 225 seeds per batch (ie per outerdir)
    new_indir = makedir(op.join(outlier_outerdir, 'lfmm2/lfmm_infiles'))
    new_shdir = makedir(op.join(outlier_outerdir, 'lfmm2/lfmm_shfiles'))
    new_outdir = makedir(op.join(outlier_outerdir, 'lfmm2/lfmm_outfiles'))
    
    training_script = '/home/b.lind/code/MVP-offsets/01_src/MVP_process_lfmm.R'
    slimdir = op.join(outerdir, 'slimdir')
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    mvp.latest_commit()
    session_info.show(html=False, dependencies=True)
    
    
    main()
    