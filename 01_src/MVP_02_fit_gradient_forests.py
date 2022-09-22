"""Fit trained models from gradient forests to the climate of a transplant location (ie the populations in the simulation).

Usage
-----
conda activate mvp_env
python MVP_01_fit_gradient_forests.py seed slimdir training_outdir rscript_exe

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scriptstraining_outdir - the location of RDS
    outfiles from MVP_01_train_gradient_forests.py. The path endswith: gradient_forests/training/training_outfiles
rscript_exe
    path to R environment's Rscript executable - eg ~/anaconda3/envs/gf_env/bin/Rscript
    R environment must be able to `library(gradientforests)`

Notes
-----
- based on previous testing, this script will require ~250GB of RAM for 7 simultaneous jobs

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon creation of conda environment - see 01_src/README.md
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
# from MVP_01_train_gradient_forests import read_ind_data


def make_fitting_dirs(training_outdir):
    """Create some dirs to put infiles, outfiles, and figs into."""
    print(ColorText('\nCreating directories ...').bold().custom('gold'))
    gf_directory = op.dirname(op.dirname(training_outdir))
    fitting_dir = makedir(op.join(gf_directory, 'fitting/fitting_outfiles'))
    garden_dir = makedir(op.join(gf_directory, 'fitting/garden_files'))
    training_filedir = training_outdir.replace('training_outfiles', 'training_files')

    print(f'\t{gf_directory = }')
    print(f'\t{fitting_dir = }')
    print(f'\t{garden_dir = }')
    print(f'\t{training_filedir = }')

    return garden_dir, fitting_dir, training_filedir


# def load_fitness_matrix():
#     """Load fitness matrix output from simulations."""
#     # an n_deme x n_deme table that indicates the mean fitness of individuals 
#         # from the source deme (in columns) in the transplant deme (in rows) 
        
#     fitness = pd.read_table(op.join(slimdir, f'{seed}_fitnessmat.txt'),
#                             delim_whitespace=True,
#                             header=None)

#     # set column names for popID
#     fitness.columns = range(1, 101, 1)
#     fitness.index = range(1, 101, 1)

#     return fitness


# def get_pop_locations():
#     """Get coordinates for each population in the simulations."""
#     # get the individuals that were subsampled from full simulation
#     subset = read_ind_data()

#     # get x and y coords for each population
#     locations = subset.groupby('subpopID')[['x', 'y']].apply(np.mean)
#     locations.columns = ['lon', 'lat']  # index = subpopID
    
#     return locations


def get_envdata():
    """Read in the environmental data used to train gradient forests.
    
    Notes
    -----
    these files were created in MVP_00_train_gradient_forests.py
    """
    print(ColorText('\nGetting environmental data ...').bold().custom('gold'))
    files = fs(training_filedir, pattern='_envfile_', startswith=seed)
    assert len(files) == 2, files

    envdfs = {}
    envfiles = {}
    for f in files:
        ind_or_pooled = op.basename(f).split("_")[-1].rstrip(".txt")
        envdfs[ind_or_pooled] = pd.read_table(f, index_col=0)
        envfiles[ind_or_pooled] = f

    return envdfs, envfiles


def create_garden_files(envdfs: dict, envfiles: dict) -> defaultdict:
    """For each transplant garden, create a file that has this gardens climate across all individuals/pools.
    
    Notes
    -----
    These are basically "future" climates when transplanting populations.
    """
    print(ColorText('\nCreating "future climates" for each common garden ...'))
    garden_files = defaultdict(list)
    for ind_or_pooled, f in envfiles.items():  # for each way of training gradient forests
        df = envdfs[ind_or_pooled].copy()
        for subpopID in envdfs['pooled'].index:  # for each garden's climate
            # set all individuals/pools to the same value
            df.loc[:] = envdfs['pooled'].loc[subpopID, df.columns].tolist()
            # save
            garden_file = op.join(garden_dir, op.basename(f).replace(".txt", f"_{subpopID}.txt"))
            df.to_csv(garden_file, sep='\t', index=True)
            # record
            assert garden_file not in garden_files[ind_or_pooled]
            garden_files[ind_or_pooled].append(garden_file)
    
    return garden_files


def fit_gradient_forests(gfOut_trainingfile, garden_file, predfile, basename, save_dir):
    """Fit trained Gradient Forest model using `fitting_file` from above.
    
    Notes
    -----
    this effectively runs the commandline command: Rscript MVP_gf_fitting_script.R arg1 arg2 arg3 arg4 arg5
    """
    import subprocess
    
    output = subprocess.check_output(
        [
            rscript_exe,
            fitting_file,
            gfOut_trainingfile,
            garden_file,
            predfile,
            basename,
            save_dir            
        ]
    ).decode('utf-8')
    
    return output


def handle_dead_jobs(jobs, args):
    """Redo any jobs that died."""
    print(ColorText('\nRedoing any jobs that died ...').bold().custom('gold'))
    
    iteration = 0
    while True:
        needed_args = []
        for i, j in enumerate(jobs):
            try:
                _ = j.r
            except:
                # something caused the job to die, probably mem
                needed_args.append(args[i])
        
        if len(needed_args) == 0:
            break
        
        jobs = []
        for arg in needed_args:
            jobs.append(
                lview.apply_async(
                    fit_gradient_forests, *arg
                )
            )
            
        watch_async(jobs, desc=f'iteration: {iteration}')
        
        args = needed_args
        
        iteration += 1
        

    pass


def run_fit_gradient_forests(garden_files):
    """Parallelize fitting of gradient forests model to future climates of transplant locations."""
    print(ColorText('\nFitting gradient forest models to common garden climates ...').bold().custom('gold'))

    # get the RDS output from training
    predfiles = fs(training_outdir, pattern=f'{seed}_', endswith='predOut.RDS')
    assert len(predfiles) == 6  # ind_all ind_adaptive ind_neural, pooled_all pooled_adaptive, pooled_neutral

    # run parallelization
    jobs = []
    basenames = []
    allargs = []
    for predfile in predfiles:
        trainingfile = predfile.replace("_predOut.RDS", "_training.RDS")
        assert op.exists(trainingfile)
        
        # was this predfile from training with individuals or pools? with adaptive loci or all?
        ind_or_pooled, marker_set = op.basename(predfile)\
                                            .split("GF_training_")[1]\
                                            .split("_gradient_forest")[0]\
                                            .split('_')

        for garden_file in garden_files[ind_or_pooled]:
            garden_ID = op.basename(garden_file).split("_")[-1].rstrip(".txt")
            
            basename = f'{seed}_{ind_or_pooled}_{marker_set}_{garden_ID}'
            assert basename not in basenames
            basenames.append(basename)

            args = (trainingfile,
                    garden_file,
                    predfile,
                    basename,
                    fitting_dir)
            allargs.append(args)

            jobs.append(
                lview.apply_async(
                    fit_gradient_forests, *args
                )
            )

    # watch progress of parallel jobs
    watch_async(jobs, desc='fitting gradient forests')
    
    # redo any jobs that died
    handle_dead_jobs(jobs, allargs)

    pass


def main():
    # get environmental data
    envdfs, envfiles = get_envdata()

    # create future climates (transplant climates) for each subpopID
    garden_files = create_garden_files(envdfs, envfiles)

    # parallelize fit of GF models to climates of each common garden (transplant location aka future climate)
    run_fit_gradient_forests(garden_files)

    # DONE!
    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText('\nDONE!!').bold().green())
    print(ColorText(f'\ttime to complete: {formatclock(dt.now() - t1, exact=True)}\n'))

    pass

if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, training_outdir, rscript_exe = sys.argv

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()

    # create dirs
    garden_dir, fitting_dir, training_filedir = make_fitting_dirs(training_outdir)

    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=10, profile=f'GF_{seed}')

    # load objects to cluster
    dview['rscript_exe'] = rscript_exe
    fitting_file = op.join(op.dirname(thisfile), 'MVP_gf_fitting_script.R')
    dview['fitting_file'] = fitting_file

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
