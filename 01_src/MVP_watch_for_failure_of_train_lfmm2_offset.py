"""If a job fails, find the commands that failed and try again, repeat process until all jobs are completed successfully.

Usage
-----
conda activate mvp_env
python MVP_watch_for_failure_of_train_lfmm2_offset.py shfile outerdir num_engines


Parameters
----------
shfile
    a slurm sbatch file that contains a command that `cat`s a text file to GNU parallel
outerdir
    the directory into which all of the lfmm2 subdirectories were created in MVP_10_train_lfmm2_offset.py
num_engines
    the number of engines to start when parallelizing tasks

"""
from pythonimports import *



def get_cmds():
    """Get the list of commands that were attempted for the job in `shfile`."""
    lines = read(shfile, lines=True)

    for line in lines:
        if line.startswith('cat'):
            cat_file = line.split('|')[0].split()[-1]

            cmds = read(cat_file, lines=True)

    return cmds


def get_needed_cmds(cmds):
    """Which commands need to be rerun?"""
    needed_cmds = []
    for cmd in cmds:
        *args, outfile, _ = cmd.split()
        if not op.exists(outfile):
            needed_cmds.append(cmd)

    return needed_cmds


def get_args():
    """Retrieve input argument at pipeline start."""
    argfiles = fs(outerdir, startswith='pipeline_input_args', endswith='.pkl')
    
    # if the pipeline is started more than once for same outdir, a new .pkl will be created but will not overwrite previous.
        # get the most recent pkl file
    argfile = getmostrecent(argfiles)
    
    args = pklload(argfile)
    
    return args

def run_script(cmd):
    """Execute the command `cmd`."""
    import subprocess
    
    output = subprocess.check_output(cmd.split())
    
    return output


def run_jobs(needed_cmds, condadir, lview, dview):
    """Redo commands that failed."""
    jobs = []
    for cmd in needed_cmds:
        cmd = cmd.replace('Rscript', op.join(condadir, 'MVP_env_R4.0.3/bin/Rscript'))
        jobs.append(lview.apply_async(run_script, cmd))
        
    watch_async(jobs)
    
    pass


def main(iteration=0):
    print(ColorText(f'\nLooking for failed jobs: {iteration = }').bold().custom('gold'))
    
    # get commands from batch file
    cmds = get_cmds()
    
    # figure out which commands failed
    needed_cmds = get_needed_cmds(cmds)
    
    # if there are no commands to be rerun, exit
    if len(needed_cmds) == 0:
        print(ColorText('\nShutting down engines ...').bold().custom('gold'))
        print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
        print(ColorText('\nDONE!!').bold().green(), '\n')
        sys.exit(0)
    
    # retrieve input argument at pipeline start
    args = get_args()
    
    # redo commands that failed
    run_jobs(needed_cmds, args.condadir, lview, dview)
    
    # keep running main until all the jobs are done
    iteration += 1
    main(iteration=iteration)


if __name__ == '__main__':
    thisfile, shfile, outerdir, num_engines = sys.argv
    
    print(
        ColorText(
            f'\nStarting {op.basename(thisfile)} ...'
        ).bold().custom('gold')
    )
    
    # start cluster
    lview, dview, cluster_id = start_engines(n=int(num_engines), profile=f'lfmm2_{seed}')
    
    # timer
    t1 = dt.now()
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()