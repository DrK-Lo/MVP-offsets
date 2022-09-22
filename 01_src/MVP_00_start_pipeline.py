"""Start pipeline to analyse offset methods using simulations from MVP.

Usage
-----
conda activate mvp_env
MVP_00_start_pipeline.py -s SLIMDIR -o OUTDIR -e EMAIL -c CONDADIR 
                        [--gf] [--rona] [--gdm] [--lfmm] [--all] [-h]

TODO
----
- replace r35 with gf_env - multiple places
- add option for number of engines?
- warn about dependencies
    - a user could run other offsets assuming they already ran dependency offset scripts
    - so warn if --rona is selected but not --gf (rona depends on gf)
"""
from pythonimports import *
import argparse

import MVP_10_train_lfmm2_offset as mvp10


def get_pars():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=pipeline_text,
                                     add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    requiredNAMED = parser.add_argument_group('required arguments')
    
    # required arguments
    requiredNAMED.add_argument("-s", "--slim-dir",
                               required=True,
                               default=argparse.SUPPRESS,
                               dest="slimdir",
                               type=str,
                               help="/path/to/directory/with_all/SLiM_simulations")

    requiredNAMED.add_argument("-o", "--outdir",
                               required=True,
                               default=argparse.SUPPRESS,
                               dest='outdir',
                               type=str,
                               help='''/path/to/parent/directory/of_all_output_analyses
All analyses will have directories nested under
this outdir directory.''')

    requiredNAMED.add_argument("-e", "--email",
                               required=True,
                               dest="email",
                               help='''the email address you would like to have slurm 
notifications sent to''')
    
    requiredNAMED.add_argument("-c", "--condadir",
                               required=True,
                               default=argparse.SUPPRESS,
                               dest="condadir",
                               type=str,
                               help="""/path/to/anaconda3/envs
The directory under which all anaconda envs are stored.""")
    
    # optional arguments
    parser.add_argument("--gf",
                        required=False,
                        action='store_true',
                        dest="run_gf",
                        help='''Boolean: true if used, false otherwise.
Whether to run Gradient Forests analysis.''')
    
    parser.add_argument("--rona",
                        required=False,
                        action='store_true',
                        dest='run_rona',
                        help='''Boolean: true if used, false otherwise.
Whether to run Risk Of Non-Adaptedness analysis.''')
    
    parser.add_argument("--gdm",
                        required=False,
                        action='store_true',
                        dest='run_gdm',
                        help='''Boolean: true if used, false otherwise.
Whether to run Generalized Dissimilarity Models.''')

    parser.add_argument("--lfmm",
                        required=False,
                        action='store_true',
                        dest='run_lfmm',
                        help='''Boolean: True if used, False otherwise.
Whether to run LFMM2 offset models.''')
    
    parser.add_argument("--rda",
                        required=False,
                        action='store_true',
                        dest='run_rda',
                        help='''Boolean: True if used, False otherwise.
Whether to run RDA offset models.''')
    
    parser.add_argument("--all",
                        required=False,
                        action='store_true',
                        dest='run_all',
                        help='''Boolean: True if used, False otherwise.
Whether to run all offset analyes.''')
    
    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n')
    
    # check args
    args = parser.parse_args()
    
    # check for conda envs
    badpaths = []
    for env in ['r35', 'mvp_env']:
        envpath = op.join(args.condadir, env)
        if op.exists(envpath) is False:
            badpaths.append(envpath)
    if len(badpaths) > 0:
        text = """Error, the following conda envs do not exist: \n%s""" % '\n'.join(badpaths)
        raise Exception(text)
        
    pkldump(args, op.join(args.outdir, f'pipeline_input_args_{dt.now().strftime("%m-%d-%Y-%H:%M:%S")}.pkl'))
    
    return args


def get_seeds(slimdir):
    """Get all seeds from the slimdir."""
    files = fs(slimdir, endswith='_Rout_ind_subset.txt')
    
    seeds = []
    for f in files:
        seed, *_ = op.basename(f).split("_")
        if seed not in seeds:
            seeds.append(seed)
            
    return seeds


def execute_gf(seeds, args):
    """Submit training scripts for Gradient Forests.
    
    Notes
    -----
    training scripts will submit fitting and validation scripts autonomously    
    """
    print(ColorText('Creating and sbatching Gradient Forest scripts ...').bold().custom('gold'))
    
    rscript = op.join(args.condadir, 'r35/lib/R/bin/Rscript')
    shdir = makedir(op.join(args.outdir, 'gradient_forests/training/training_shfiles/kickoff_shfiles'))
    
    shfiles = {}
    for seed in pbar(seeds):
        basename = f'{seed}_gf_training'
        shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=00:10:00
#SBATCH --mem=4000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc

conda activate mvp_env

cd {mvp_dir}

python MVP_01_train_gradient_forests.py {seed} {args.slimdir} {args.outdir} 56 {rscript} {args.email}

"""
        shfile = op.join(shdir, f'{basename}.sh')
        with open(shfile, 'w') as o:
            o.write(shtext)
        
        shfiles[seed] = shfile
    
    pids = {}
    for seed, sh in shfiles.items():
        pids[seed] = sbatch(sh, progress_bar=False)[0]
        
    return pids

def execute_rona(seeds, args, gf_pids=None):
    """Submit scripts to train and validate the Risk Of Non-Adaptedness offset."""
    print(ColorText('Creating and sbatching Risk Of Non-Adaptedness scripts ...').bold().custom('gold'))
    
    rona_training_dir = op.join(args.outdir, 'RONA/training/training_files')  # created in MVP_01.py
    rona_outdir = op.join(args.outdir, 'RONA/training/training_outfiles')  # created in MVP_05.py
    shdir = makedir(op.join(args.outdir, 'RONA/shfiles'))
    
    shfiles = []
    for seed in seeds:
        basename = f'{seed}_RONA'
        dependency_text = '' if gf_pids is None else f'#SBATCH --dependency=afterok:{gf_pids[seed]}'
        shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=01:00:00
#SBATCH --mem=4000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=FAIL
{dependency_text}

source $HOME/.bashrc

conda activate mvp_env

cd {mvp_dir}

python MVP_05_train_RONA.py {seed} {args.slimdir} {rona_training_dir} 56

python MVP_06_validate_RONA.py {seed} {args.slimdir} {rona_outdir}

"""
        shfile = op.join(shdir, f'{basename}.sh')
        with open(shfile, 'w') as o:
            o.write(shtext)
        
        shfiles.append(shfile)
    
    pids = sbatch(shfiles)            
    
    pass


def execute_lfmm(seeds, args):
    """Submit training and validation scripts for Latent Factor Mixed Models offset.
    
    Notes
    -----
    MVP_10.py will submit validation scripts autonomously    
    """
    print(ColorText('Creating and sbatching Latent Factor Mixed Model scripts ...').bold().custom('gold'))

    shdir = makedir(op.join(args.outdir, 'lfmm2/lfmm_shfiles/kickoff_shfiles'))
    
    shfiles = []
    for seed in seeds:
        basename = f'{seed}_lfmm'
        
        shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=00:05:00
#SBATCH --mem=500
#SBATCH --partition=short
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc

conda activate mvp_env

cd {mvp_dir}

python MVP_10_train_lfmm2_offset.py {seed} {args.slimdir} {args.outdir} {args.email}

"""
        shfile = op.join(shdir, f'{basename}.sh')
        with open(shfile, 'w') as o:
            o.write(shtext)
        
        shfiles.append(shfile)
    
    pids = sbatch(shfiles)
    
    assert len(seeds) == len(pids)
    lfmm_pids = dict(zip(seeds, pids))
    
    return lfmm_pids

def execute_gdm(seeds, args, gf_pids=None):
    print(ColorText('Creating and sbatching Generalized Dissimilarity Modeling scripts ...').bold().custom('gold'))
    
    shdir = makedir(op.join(args.outdir, 'fst/shfiles'))
    
    gf_training_dir = op.join(args.outdir, 'gradient_forests/training/training_files')  # created in MVP_01.py
    
    shfiles = []
    for seed in seeds:
        basename = f'{seed}_fst'
        dependency_text = '' if gf_pids is None else f'#SBATCH --dependency=afterok:{gf_pids[seed]}'
        
        shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=3:00:00
#SBATCH --mem=4000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=FAIL
{dependency_text}

source $HOME/.bashrc

conda activate mvp_env

cd {mvp_dir}

python MVP_07_calc_WC_pairwise_FST.py {seed} {args.slimdir} {gf_training_dir} 56

"""
        shfile = op.join(shdir, f'{basename}.sh')
        with open(shfile, 'w') as o:
            o.write(shtext)
        shfiles.append(shfile)
    
    sbatch(shfiles)
    
    pass

def execute_rda(seeds, args, gf_pids=None):
    print(ColorText('Creating and sbatching Redundancy Analysis offset scripts ...').bold().custom('gold'))
    
    shdir = makedir(op.join(args.outdir, 'rda/shfiles'))
    gf_training_dir = op.join(args.outdir, 'gradient_forests/training/training_files')
    rda_dir = op.join(args.outdir, 'rda')
    rda_outdir = makedir(op.join(rda_dir, 'rda_files'))
    rda_catdir = makedir(op.join(rda_dir, 'rda_catfiles'))
    
    shfiles = []
    for seed in seeds:
        basename = f'{seed}_rda'
        dependency_text = '' if gf_pids is None else f'#SBATCH --dependency=afterok:{gf_pids[seed]}'
        
        # how many of the traits (envs) in this seed are imposing selection?
        ntraits = mvp10.determine_adaptive_envs(args.slimdir, seed)
        
        snpfile = op.join(gf_training_dir, f'{seed}_Rout_Gmat_sample_maf-gt-p01_GFready_pooled_all.txt')
        
        # files created by Katie
        rda_files = fs(args.slimdir, startswith=f'{seed}_RDA', endswith='.RDS')
        assert len(rda_files) == 2, len(rda_files)  # structure-corrected and -uncorrected
        
        # the files that will be created by MVP_pooled_pca_and_rda.R
        rda_files.extend([
            op.join(rda_outdir, f'{seed}_pooled_RDA.RDS'),
            op.join(rda_outdir, f'{seed}_pooled_RDA_structcorr.RDS')
        ])
        
        # get a list of RDA offset commands
        cmds = []
        for rda_file in rda_files:
            for use_RDA_outliers in ['TRUE', 'FALSE', 'CAUSAL', 'NEUTRAL']:
                cmds.append(
                    f"Rscript MVP_12_RDA_offset.R {seed} {args.slimdir} {args.outdir} {rda_file} {use_RDA_outliers} {ntraits}"
                )
        
        cmdfile = op.join(rda_catdir, f'{seed}_rda_commands.txt')
        with open(cmdfile, 'w') as o:
            o.write('\n'.join(cmds))        
        
        shtext = f"""#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time=1:00:00
#SBATCH --mem=9000
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task={len(cmds)}
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=FAIL
{dependency_text}

source $HOME/.bashrc

module load parallel

conda activate MVP_env_R4.0.3

cd {mvp_dir}

# create PCA data for pool-seq SNPs
Rscript {mvp_dir}/MVP_pooled_pca_and_rda.R {seed} {args.slimdir} {snpfile} {args.outdir} {mvp_dir} # /full/path.R is necessary

# run RDA offset estimation 
cat {cmdfile} | parallel -j {len(cmds)} --progress --eta

# validate offset estimation
conda activate mvp_env
python MVP_13_RDA_validation.py {seed} {args.slimdir} {args.outdir}

"""
        shfile = op.join(shdir, f'{basename}.sh')
        with open(shfile, 'w') as o:
            o.write(shtext)
            
        shfiles.append(shfile)

    # sbatch jobs and retrieve SLURM_JOB_IDs
    pids = sbatch(shfiles)

    # create alert for notification
    create_watcherfile(pids, shdir, watcher_name='rda_watcher', email=args.email)
    
    pass
        

def main():
    # parse arguments
    args = get_pars()
    
    # get a list of simluations (identified by their seed number)
    seeds = get_seeds(args.slimdir)

    # create cluster profiles for each seed

    # which offsets to run?
    if args.run_all is True:
        args.run_gf = True
        args.run_rona = True
        args.run_lfmm = True
        args.run_gdm = True
        args.run_rda = True

    # run gradient forest offset analyses
    if args.run_gf is True:
        gf_pids = execute_gf(seeds, args)

    # run risk of non-adaptedness offset analyses
    if args.run_rona is True:
        if args.run_gf is True:
            execute_rona(seeds, args, gf_pids)
        else:
            execute_rona(seeds, args)

    # run latent factor mixed model offset analyses
    if args.run_lfmm is True:
        lfmm_pids = execute_lfmm(seeds, args)

    # run generalized dissimimilarity model offset analysis
    if args.run_gdm is True:
        if args.run_gf is True:
            execute_gdm(seeds, args, gf_pids)
        else:
            execute_gdm(seeds, args)
    
    # run RDA offset
    if args.run_rda is True:
        if args.run_gf is True:
            execute_rda(seeds, args, gf_pids)
        else:
            execute_rda(seeds, args)


    pass


if __name__ == '__main__':
    pipeline_text = ColorText('''
************************************************************************

                          MVP Offsets Pipeline                          

************************************************************************
''').bold().green().__str__()
    
    # put in global namespace
    thisfile = sys.argv[0]
    mvp_dir = op.dirname(op.abspath(thisfile))
    
    main()