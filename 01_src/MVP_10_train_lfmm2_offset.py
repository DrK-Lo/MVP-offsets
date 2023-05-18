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


def read_params_file(slimdir):
    """Read in the file that contains all parameters for each seed."""
    # find the file
    paramsfile = op.join(slimdir, '0b-final_params-20220428.txt')

    # read in the file, drop meaningless rows
    params = pd.read_table(paramsfile, delim_whitespace=True)
    params = params[params['seed'].notnull()]
    
    # fix misspelling
    params['level'] = params['level'].str.replace('oliogenic', 'oligogenic')

    # set dataframe index as seed ID
    params.index = params['seed'].astype(int).astype(str).tolist()
    
    return params


def determine_adaptive_envs(slimdir, seed):
    """From the seed params file, determine if there is one or two adaptive environments."""
    
    params = read_params_file(slimdir)
    
    n_traits = params.loc[seed, 'N_traits']
    
    return n_traits


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
    outdir = makedir(op.join(outerdir, f'lfmm2/lfmm_outfiles/{seed}'))
    shdir = makedir(op.join(outerdir, 'lfmm2/lfmm_shfiles'))

    return indir, outdir, shdir


def create_current_envfiles(subset, n_traits):
    """Create environmental data for the `env` argument of the lfmm genetic.offset function.
    
    Parameters
    ----------
    subset - pd.DataFrame
        information for the individuals that were chosen from the full simulation
    n_traits - int
        whether there are one or two adaptive traits (envs) - see `determine_adaptive_envs()`
        
    Returns
    -------
    lfmm_envfile - dict
        - There could be one (when n_traits == 2) or two keys (when n_traits == 1)
            - when n_traits==1 we also want to test whether adding a dummy env affects outcome (hence two keys)
        - values are paths to the files used for the `env` argument of the lfmm genetic.offset function
        - if there is only one adaptive trait, then the dict is of length=2 (both envs + only
            adaptive env), otherwise is of length=1 (both envs)
    """
    print(ColorText('\nCreating envfile for current environments ...').bold().custom('gold'))

    lfmm_envfiles = {}
    if n_traits == 1:
        # if there is only one adaptive env (temp_opt), create a file for that env
        lfmm_envfile = op.join(indir, f'{seed}_lfmm_env_ntraits-1.txt')
        current_envdata = subset[['temp_opt']]
        current_envdata.to_csv(lfmm_envfile, sep='\t', index=False, header=False)
        
        lfmm_envfiles[1] = lfmm_envfile

    # whether n_traits==1 or n_traits==2, create a file for both envs (which could include a non-adaptive env)
    lfmm_envfile_both = op.join(indir, f'{seed}_lfmm_env_ntraits-2.txt')
    current_envdata = subset[['sal_opt', 'temp_opt']]
    current_envdata.to_csv(lfmm_envfile_both, sep='\t', index=False, header=False)
    
    lfmm_envfiles[2] = lfmm_envfile_both

    return lfmm_envfiles


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


def create_transplant_envfiles(subset, lfmm_envfiles):
    """Create environment files for the `new.env` argument of lfmm genetic.offset function.

    Parameters
    ----------
    subset - pd.DataFrame
        information for the individuals that were chosen from the full simulation
    lfmm_envfile - dict
        - returned from `create_current_envfiles()`
        - There could be one (when n_traits == 2) or two keys (when n_traits == 1)
            - when n_traits==1 we also want to test whether adding a dummy env affects outcome (hence two keys)
        - values are paths to the files used for the `env` argument of the lfmm genetic.offset function
        - if there is only one adaptive trait, then the dict is of length=2 (both envs + only
            adaptive env), otherwise is of length=1 (both envs)

    Returns
    -------
    garden_files - defaultdict(list)
        list of paths; each path to be used for the `new.env` argument of lfmm genetic.offset function
    """
    print(ColorText('\nCreating envfiles for transplant (subpopID) environments ...').bold().custom('gold'))

    # key = env val = dict with key=subpopID and val=env_value
    transplant_envdata = subset.groupby('subpopID')[['sal_opt', 'temp_opt']].apply(np.mean)

    # create transplant environmental files
    garden_files = defaultdict(list)
    for n_traits, lfmm_envfile in lfmm_envfiles.items():
        envs = ['temp_opt'] if n_traits==1 else ['sal_opt', 'temp_opt']  # this would have been better w/ temp_opt first!

        for subpop_id, (sal_opt, temp_opt) in transplant_envdata.iterrows():
            X_new = subset[envs].copy()  # create dataframe to be overwritten

            X_new['temp_opt'] = temp_opt  # overwrite temp_opt with single value of transplant env

            if 'sal_opt' in envs:  # if both envs are adaptive, or adding in non-adaptive env
                X_new['sal_opt'] = sal_opt  # overwrite sal_opt with single value of transplant env

            lfmm_envfile_new = lfmm_envfile.replace('.txt',
                                                    f'_{str(subpop_id).zfill(3)}.txt')  # use three digit ID for file sorting

            X_new.to_csv(lfmm_envfile_new, sep='\t', index=False, header=False)

            garden_files[n_traits].append(lfmm_envfile_new)

    return garden_files


def create_adaptive_and_neutral_files():
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
    
    # read in the dataframe with individual counts of derived allele
        # this also has the correct order of loci that went into {seed}_genotype.lfmm
        # at the time of this comment muts.index and snps.index had a small number of mismatches
        # however, the data was correct across individuals (checked in MVP_01.py : get_012)
    snpfile = op.join(slimdir, f'{seed}_Rout_Gmat_sample.txt')
    snps = pd.read_table(snpfile, delim_whitespace=True)
    snps['idx'] = range(1, nrow(snps)+1)  # create a 1-based index for cand.loci arg of LEA::genetic.offset

    loci_files = []
    for marker_set in ['adaptive', 'neutral']:
        # get column indices of {seed}_genotypes.lfmm file that correspond to adaptive loci
            # {seed}_genotypes.lfmm is used as `input` to the lfmm genetic.offset function
        if marker_set == 'adaptive':
            loci = muts['mutname'][muts['mutID'] != 1]  # VCFrow is 1-based index, perfect for R
        else:
            loci = muts['mutname'][(muts['causal_temp'] == 'neutral') & (muts['causal_sal'] == 'neutral')]
    
        loci_indices = snps['idx'].loc[loci]
        loci_df = pd.DataFrame(loci_indices)
        loci_file = op.join(indir, f'{seed}_{marker_set}_loci.txt')
        loci_df.to_csv(loci_file, index=False, header=False)
        
        loci_files.append(loci_file)
        
#     # get column indices of {seed}_genotypes.lfmm file that correspond to adaptive loci
#     # {seed}_genotypes.lfmm is used as `input` to the lfmm genetic.offset function
#     adaptive_loci = muts['VCFrow'][muts['mutID'] != 1]  # VCFrow is 1-based index, perfect for R

#     adaptive_df = pd.DataFrame(adaptive_loci)
#     adaptive_file = op.join(indir, f'{seed}_adaptive_loci.txt')
#     adaptive_df.to_csv(adaptive_file, index=False, header=False)
    

    return loci_files


def create_shfiles(lfmm_envfiles, poplabel_file, garden_files, locus_files, thresh=72):
    """Create slurm job files to estimate genetic offset using lfmm method.
    
    Notes
    -----
    because the number of running jobs on the Discovery cluster is limited, here I run batches
        of commands in parallel for a single job (instead of submitted many individual jobs)
        
    Parameters
    ----------
    lfmm_envfiles - dict
        - There could be one (when n_traits == 2) or two keys (when n_traits == 1)
            - when n_traits==1 we also want to test whether adding a dummy env affects outcome (hence two keys)
        - values are paths to the files used for the `env` argument of the lfmm genetic.offset function
        - if there is only one adaptive trait, then the dict is of length=2 (both envs + only
            adaptive env), otherwise is of length=1 (both envs)
    poplabel_file - str
        path to the file to be used for the `pop.labels` argument of the lfmm genetic.offset function
    garden_files - defaultdict(list)
        list of paths; each path to be used for the `new.env` argument of lfmm genetic.offset function
    locus_files - list
        paths to files to be used for the `candidate.loci` argument of lfmm genetic.offset function
    thresh - int
        number of CPUs per job so I can be < 180GB mem, at ~4700MB per job (capcity for lotterhos node is ~180GB per node)
    """
    print(ColorText('\nCombining commands into batches and writing slurm sh files ...').bold().custom('gold'))
    
    adaptive_file, neutral_file = locus_files
    
    # how to know when I'm done creating commands = num_gardens * num_marker_sets
    expected_jobcount = len(flatten(garden_files.values())) * (len(locus_files) + 1)  # hard-coded 1 is "all" loci below
    
    jobcount = 0  # to know when I'm done iterating
    cmds = []  # list of training commands to parallelize for a given job
    shfiles = []
    for n_traits, gfiles in garden_files.items():  # using one or two environments
        for garden_file in gfiles:  # for each of the transplant environments
            garden = op.basename(garden_file).split("_")[-1].split('.')[0]  # subpopID

            for cand_file, marker_set in zip(['None', adaptive_file, neutral_file], ['all', 'adaptive', 'neutral']):
                output_file = op.join(outdir, f'{seed}_lfmm_offsets_ntraits-{n_traits}_{marker_set}_{garden}.txt')

                cmd = ' '.join(['Rscript',
                                training_script,
                                seed,
                                slimdir,
                                lfmm_envfiles[n_traits],
                                garden_file,
                                poplabel_file,
                                output_file,
                                cand_file])
                cmds.append(cmd)
                jobcount += 1

                if len(cmds) == thresh or jobcount == expected_jobcount:
                    batch_num = str(len(shfiles)).zfill(2)
                    job = f'{seed}_lfmm_batch_{batch_num}'

                    # determine amount of memory required (with cushion)
#                     if len(cmds) == thresh:
#                         mem = '175000M' 
#                     else:
#                         val = (len(cmds) * 4900) + 5000
#                         mem = f'{val}M'

                    # write commands to a file that I can `cat` to GNU parallel
                    cmd_file = op.join(indir, f'{job}.txt')
                    with open(cmd_file, 'w') as o:
                        o.write('\n'.join(cmds))

                    # where to write sh text
                    shfile = op.join(shdir, f'{job}.sh')

                    # what text to write to sh
                    text = f'''#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time=06:00:00
#SBATCH --mem=175000M
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --output={job}_%j.out
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc  # assumed that conda init is within .bashrc

module load parallel

conda activate MVP_env_R4.0.3

# train lfmm2 and calculate offset
cat {cmd_file} | parallel -j {len(cmds)} --progress --eta

conda activate mvp_env

cd {mvp_dir}

# re run any failed jobs from the cmd_file
python MVP_watch_for_failure_of_train_lfmm2_offset.py {seed} {shfile} {outerdir} 36

'''
                    # write slurm script to file
                    with open(shfile, 'w') as o:
                        o.write(text)
                    shfiles.append(shfile)

                    cmds = []  # reset

    return shfiles


# def watch_for_failures(shfiles, num_engines=34):
#     """Sometimes jobs can fail, and not all parallel jobs complete. Add dependency to check each job."""
#     print(ColorText('\nSending sbatch scripts to slurm ...').bold().custom('gold'))
    
#     watcher_pids = []
#     for shfile in pbar(shfiles, desc='sbatching'):
#         pid = sbatch(shfile, progress_bar=False)[0]
        
#         watcher_shfile = shfile.replace('.sh', '_watcher.sh')
#         basename = op.basename(watcher_shfile).replace('.sh', '')
        
#         shtext = f"""#!/bin/bash
# #SBATCH --job-name={basename}
# #SBATCH --time=1:00:00
# #SBATCH --mem=165000M
# #SBATCH --partition=short
# #SBATCH --nodes=1
# #SBATCH --cpus-per-task={num_engines}
# #SBATCH --output={basename}_%j.out
# #SBATCH --mail-user={email}
# #SBATCH --mail-type=FAIL
# #SBATCH --dependency=afternotok:{pid}

# source $HOME/.bashrc

# conda activate mvp_env

# cd {mvp_dir}

# python MVP_watch_for_failure_of_train_lfmm2_offset.py {shfile} {outerdir} {num_engines}

# """
        
#         with open(watcher_shfile, 'w') as o:
#             o.write(shtext)
            
#         watcher_pid = sbatch(watcher_shfile, progress_bar=False)
#         watcher_pids.extend(watcher_pid)
    
#     return watcher_pids


def kickoff_validation(pids):
    """Once batch jobs and batch watcher jobs complete, start validation of offset."""
    basename = f'{seed}_lfmm_validation'
    shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=3:00:00
#SBATCH --mem=4000
#SBATCH --partition=short
#SBATCH --nodes=1
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
    
    # how many traits (envs) were adaptive in this sim?
    n_traits = determine_adaptive_envs(slimdir, seed)

    # create file for "current" environment
    lfmm_envfiles = create_current_envfiles(subset, n_traits)

    # create pop labels file
    poplabel_file = create_poplabels(subset)

    # create transplant environmental files ("future" climates)
    garden_files = create_transplant_envfiles(subset, lfmm_envfiles)

    # save adaptive loci to file
    locus_files = create_adaptive_and_neutral_files()

    # create slurm sbatch files
    shfiles = create_shfiles(lfmm_envfiles, poplabel_file, garden_files, locus_files)

    # submit jobs to slurm
#     watcher_pids = watch_for_failures(shfiles)
    pids = sbatch(shfiles)

    # create a watcher file for kicking off validation
    kickoff_validation(pids)
    
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
