"""Calculate offset using method from lfmm2 for a specific simulation seed.

Calculate using either all loci or only those known to underlie adaptation to the environment.

Usage
-----
conda activate mvp_env
python MVP_10_train_lfmm2_offset.py seed slimdir outerdir email

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scripts
outerdir
    the directory into which all of the lfmm2 subdirectories will be created 
email
    email to be used for slurm notifications

Dependencies
------------
- dependent upon code from github.com/brandonlind/pythonimports
- dependent upon creation of conda environment - see 01_src/README.md

Notes
-----
Gain, C., and O. François. 2021. LEA 3: Factor models in population genetics and ecological genomics with R. Mol Ecol Resour 21:2738–2748.

"""
from pythonimports import *
import MVP_01_train_gradient_forests as mvp01


def make_lfmm_dirs(outerdir):
    """Create directories for infiles, outfiles, and slurm shfiles.
    
    Parameters
    ----------
    outerdir
        the directory into which all of the lfmm2 subdirectories will be created
        
    Returns
    -------
    indir
        directory where the infiles for the process_lfmm.R script
    outdir
        directory where the outfiles from the process_lfmm.R script will be saved
    shdir
        directory where the slurm sbatch scripts will be saved (and slurm.out files will be written)
    """
    if __name__ == '__main__':
        print(ColorText('\nCreating lfmm directories ...').bold().custom('gold'))

    indir = makedir(op.join(outerdir, 'lfmm2/lfmm_infiles'))
    outdir = makedir(op.join(outerdir, 'lfmm2/lfmm_outfiles'))
    shdir = makedir(op.join(outerdir, 'lfmm2/lfmm_shfiles'))

    return indir, outdir, shdir


def create_current_envfile(subset):
    """Create environmental data for the `env` argument of the lfmm genetic.offset function.
    
    Parameters
    ----------
    subset - pd.DataFrame
        information for the individuals that were chosen from the full simulation
        
    Returns
    -------
    lfmm_envfile - str
        path to the file used for the `env` argument of the lfmm genetic.offset function
    """
    print(ColorText('\nCreating envfile for current environments ...').bold().custom('gold'))

    lfmm_envfile = op.join(indir, f'{seed}_lfmm_env.txt')

    current_envdata = subset[['sal_opt', 'temp_opt']]
    current_envdata.to_csv(lfmm_envfile, sep='\t', index=False, header=False)

    return lfmm_envfile


def create_poplabels(subset):
    """Create population labels for the `pop.labels` argument of the lfmm genetic.offset function.
    
    Parameters
    ----------
    subset - pd.DataFrame
        information for the individuals that were chosen from the full simulation
        
    Returns
    -------
    poplabel_file
        path to the file to be used for the `pop.labels` argument of the lfmm genetic.offset function
    """
    print(ColorText('\nCreating pop labels ...').bold().custom('gold'))
    poplabel_file = op.join(indir, f'{seed}_poplabels.txt')

    subset['subpopID'].to_csv(poplabel_file, sep='\t', index=False, header=False)

    return poplabel_file


def create_transplant_envfiles(subset, lfmm_envfile):
    """Create environment files for the `new.env` argument of lfmm genetic.offset function.
    
    Parameters
    ----------
    subset - pd.DataFrame
        information for the individuals that were chosen from the full simulation
    lfmm_envfile - str
        path to the file used for the `env` argument of the lfmm genetic.offset function
        
    Returns
    -------
    garden_files - list
        list of paths; each path to be used for the `new.env` argument of lfmm genetic.offset function
    """
    print(ColorText('\nCreating envfiles for transplant (subpopID) environments ...').bold().custom('gold'))
    # key = env val = dict with key=subpopID and val=env_value
    transplant_envdata = subset.groupby('subpopID')[['sal_opt', 'temp_opt']].apply(np.mean)

    # create transplant environmental files
    garden_files = []
    for subpop_id, (sal_opt, temp_opt) in transplant_envdata.iterrows():
        X_new = subset[['sal_opt', 'temp_opt']].copy()
        X_new['sal_opt'] = sal_opt
        X_new['temp_opt'] = temp_opt

        lfmm_envfile_new = lfmm_envfile.replace(
            '.txt',
            f'_{str(subpop_id).zfill(3)}.txt'  # use three digit ID for file sorting
        )
        X_new.to_csv(lfmm_envfile_new, sep='\t', index=False, header=False)

        garden_files.append(lfmm_envfile_new)

    return garden_files


def create_adaptive_file():
    """Create a file to be used for the `candidate.loci` argument of lfmm genetic.offset function.
    
    Returns
    -------
    adaptive_file
        path to file to be used for the `candidate.loci` argument of lfmm genetic.offset function
    """
    print(ColorText('\nCreating file for adaptive loci ...').bold().custom('gold'))
    mvp01.seed = seed
    mvp01.slimdir = slimdir
    muts = mvp01.read_muts_file()

    # get column indices of {seed}_genotypes.lfmm file that correspond to adaptive loci
    # {seed}_genotypes.lfmm is used as `input` to the lfmm genetic.offset function
    adaptive_loci = muts['VCFrow'][muts['mutID'] != 1]

    adaptive_df = pd.DataFrame(adaptive_loci)
    adaptive_file = op.join(indir, f'{seed}_adaptive_loci.txt')
    adaptive_df.to_csv(adaptive_file, index=False, header=False)

    return adaptive_file


def create_shfiles(lfmm_envfile, poplabel_file, garden_files, adaptive_file):
    """Create slurm job files to estimate genetic offset using lfmm method.
    
    Notes
    -----
    because the number of running jobs on the Discovery cluster is limited, here I run batches
        of commands in parallel for a single job (instead of submitted many individual jobs)
        
    Parameters
    ----------
    lfmm_envfile - str
        path to the file used for the `env` argument of the lfmm genetic.offset function
    poplabel_file - str
        path to the file to be used for the `pop.labels` argument of the lfmm genetic.offset function
    garden_files - list
        list of paths; each path (str) to be used for the `new.env` argument of lfmm genetic.offset function
    adaptive_file - str
        path to file to be used for the `candidate.loci` argument of lfmm genetic.offset function    
    """
    print(ColorText('Combining commands into batches and writing slurm sh files ...').bold().custom('gold'))
    
    # number of CPUs per job so I can be around 160GB mem, at ~4700MB per job (capcity is ~180GB per node)
    thresh = 34
    
    jobcount = 0  # to know when I'm done iterating
    cmds = []  # list of training commands to parallelize for a given job
    shfiles = []
    for garden_file in garden_files:  # for each of the transplant environments
        garden = op.basename(garden_file).split("_")[-1].split('.')[0]  # subpopID
        
        for cand_file, marker_set in zip(['None', adaptive_file], ['all', 'adaptive']):
            output_file = op.join(outdir, f'{seed}_lfmm2_offsets_{marker_set}_{garden}')
            
            cmd = ' '.join(['Rscript',
                            training_script,
                            seed,
                            slimdir,
                            lfmm_envfile,
                            garden_file,
                            poplabel_file,
                            output_file,
                            cand_file])
            cmds.append(cmd)
            jobcount += 1
            
            if len(cmds) == thresh or jobcount == len(garden_files) * 2:
                job = f'{seed}_lfmm_batch_{len(shfiles)}'
                
                # determine amount of memory required (with cushion)
                if len(cmds) == thresh:
                    mem = '165000M' 
                else:
                    val = (len(cmds)*4700) + 5000
                    mem = f'{val}M'
                
                # write commands to a file that I can `cat` to GNU parallel
                cmd_file = op.join(indir, f'{job}.txt')
                with open(cmd_file, 'w') as o:
                    o.write('\n'.join(cmds))
                
                text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=01:00:00
#SBATCH --mem={mem}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={len(cmds)}
#SBATCH --output={job}_%j.out
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc  # assumed that conda init is within .bashrc

module load parallel

conda activate MVP_env_R4.0.3

cat {cmd_file} | parallel -j {len(cmds)} --progress --eta

'''
                # write slurm script to file
                shfile = op.join(shdir, f'{job}.sh')
                with open(shfile, 'w') as o:
                    o.write(text)
                shfiles.append(shfile)
                
                cmds = []  # reset

    return shfiles


# def create_shfiles(lfmm_envfile, poplabel_file, garden_files, adaptive_file):
#     """Create slurm job files to estimate genetic offset using lfmm method.
    
#     Parameters
#     ----------
#     lfmm_envfile - str
#         path to the file used for the `env` argument of the lfmm genetic.offset function
#     poplabel_file - str
#         path to the file to be used for the `pop.labels` argument of the lfmm genetic.offset function
#     garden_files - list
#         list of paths; each path (str) to be used for the `new.env` argument of lfmm genetic.offset function
#     adaptive_file - str
#         path to file to be used for the `candidate.loci` argument of lfmm genetic.offset function
#     """
#     print(ColorText('\nCreating slurm sh files ...').bold().custom('gold'))

#     output_files = []  # make sure I'm creating unique file names
#     shfiles = []
#     for garden_file in garden_files:
#         garden = op.basename(garden_file).split("_")[-1].split('.')[0]  # subpopID

#         for cand_file in ['None', adaptive_file]:
#             # am i using all loci or only those known to be underlying adaptation?
#             adaptive = 'all' if cand_file == 'None' else 'adaptive'

#             # job name for this offset calculation
#             job = f'{seed}_lfmm2_offsets_{adaptive}_{garden}'

#             output_file = op.join(outdir, f'{job}.txt')
#             if output_file in output_files:
#                 raise Exception(f"Output file has already been created for seed {seed}: {output_file}")
#             output_files.append(output_file)

#             # write slurm script to file
#             text = f'''#!/bin/bash
# #SBATCH --job-name={job}
# #SBATCH --time=00:30:00
# #SBATCH --mem=6000M
# #SBATCH --output={job}_%j.out
# #SBATCH --mail-user={email}
# #SBATCH --mail-type=FAIL

# # source $HOME/.bashrc
# # conda deactivate
# # conda activate MVP_env_R4.0.3

# {rscript_exe} \
# {training_script} \
# {seed} \
# {slimdir} \
# {lfmm_envfile} \
# {garden_file} \
# {poplabel_file} \
# {output_file} \
# {cand_file}

# '''
#             shfile = op.join(shdir, f'{job}.sh')
#             with open(shfile, 'w') as o:
#                 o.write(text)
#             shfiles.append(shfile)

#     return shfiles

def kickoff_validation(pids):
    basename = f'{seed}_lfmm_validation'
    shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=3:00:00
#SBATCH --mem=4000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL
#SBATCH --dependency=afterok:{','.join(pids)}

source $HOME/.bashrc

conda activate mvp_env

cd {mvp_dir}

python MVP_11_validate_lfmm2_offset.py {seed} {slimdir} {outerdir}

"""
    shfile = op.join(shdir, f'{basename}.sh')
    with open(shfile, 'w') as o:
        o.write(shtext)
        
    sbatch(shfile)
    
    pass



def main():
    # get the subset of simulated individuals
    subset = mvp01.read_ind_data(slimdir, seed)

    # create file for "current" environment
    lfmm_envfile = create_current_envfile(subset)

    # create pop labels file
    poplabel_file = create_poplabels(subset)

    # create transplant environmental files ("future" climates)
    garden_files = create_transplant_envfiles(subset, lfmm_envfile)

    # save adaptive loci to file
    adaptive_file = create_adaptive_file()

    # create slurm sbatch files
    shfiles = create_shfiles(lfmm_envfile, poplabel_file, garden_files, adaptive_file)

    # submit jobs to slurm
    print(ColorText('\nSending sbatch scripts to slurm ...').bold().custom('gold'))
    pids = sbatch(shfiles)

    # create a watcher file for kicking off validation
    kickoff_validation(pids)
#     create_watcherfile(pids, shdir, watcher_name=f'{seed}_lfmm_offset_watcher', email=email)
    
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, outerdir, email = sys.argv

    print(
        ColorText(
            f'\nStarting {op.basename(thisfile)} ...'
        ).bold().custom('gold')
    )
    
    mvp_dir = op.dirname(op.abspath(thisfile))

    # path to lfmm offset training script
    training_script = op.join(op.dirname(op.abspath(thisfile)), 'MVP_process_lfmm.R')

    # set up timer
    t1 = dt.now()

    # create dirs
    indir, outdir, shdir = make_lfmm_dirs(outerdir)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
