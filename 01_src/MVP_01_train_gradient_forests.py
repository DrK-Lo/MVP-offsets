"""Train Gradient Forests using simulations from the MVP project.

URGENT NOTES
------------
- this script has been updated to ignore sbatching individual GF files (for now)
    - if removed, MVP_02 and MVP_03 must also be updated

Usage
-----
conda activate mvp_env
python MVP_01_train_gradient_forests.py seed slimdir outdir num_engines rscript email

Parameters
----------
seed - the seed number of the simulation - used to find associated files
slimdir - the location of the seed's files output by Katie's post-processing scripts
outdir - where to save files
num_engines - the number of engines to start for parallelizing tasks
rscript - path to R environment's Rscript executable - eg ~/anaconda3/envs/r35/bin/Rscript
imports_dir - path to imports.R from github.com/brandonlind/r_imports - deprecated!
email - email to receive sbatch notifications

Dependencies
------------
- dependent upon code from github.com/brandonlind/pythonimports
- dependent upon creation of conda environment - see 01_src/README.md

Notes
-----
- to be able to submit the individual training using all loci, users will need access to both the
    long and large partitions on the discovery cluster at NEU
"""
from pythonimports import *
import pandas._libs.lib as lib


def make_gf_dirs(outerdir):
    """Create some dirs to put infiles, outfiles, and figs into."""
    print(ColorText('\nCreating directories ...').bold().custom('gold'))
    
    directory = makedir(op.join(outerdir, 'gradient_forests'))
    training_filedir = makedir(op.join(directory, 'training/training_files'))
    shdir = makedir(op.join(directory, 'training/training_shfiles'))
    outfile_dir = makedir(op.join(directory, 'training/training_outfiles'))
    fitting_shdir = makedir(op.join(directory, 'fitting/fitting_shfiles'))

    return training_filedir, shdir, outfile_dir, fitting_shdir


def read_ind_data(slimdir, seed):
    """Get the individuals that were subsampled from full simulation."""
    print(ColorText('\nReading in info for subsampled individuals ...').bold().custom('gold'))
    
    subset = pd.read_table(op.join(slimdir, f'{seed}_Rout_ind_subset.txt'), delim_whitespace=True)
    subset.index = subset['indID'].astype(str).tolist()
    subset['sample_name'] = subset.index.tolist()

    return subset


def read_muts_file():
    """Read in the seed_Rout_muts_full.txt file, name loci."""
    print(ColorText('\nReading muts file ...').bold().custom('gold'))
    
    # get path to file
    muts_file = op.join(slimdir, f'{seed}_Rout_muts_full.txt')

    # read in the table
    muts = pd.read_table(muts_file, delim_whitespace=True, low_memory=False)  # low_memory to avoid mixed dtypes warning

    # update index with locus names
    muts.index = muts['mutname'].tolist()

    # make sure no duplicate locus names remain
    assert luni(muts.index) == nrow(muts)

    return muts


def save_gf_snpfile(snps, subset, z12file):
    """Format and save snp data for gradient forests training script from Lind et al."""
    # transpose so rows=individuals, columns=loci
    gf_snps = snps[subset.index.tolist()].T.copy()  # remove non-individual columns

    # add index col needed for gradient_training.R script
    gf_snps['index'] = gf_snps.index.tolist()

    # save
    gf_snp_files['ind']['all'] = op.join(training_filedir,
                                         op.basename(z12file).replace('.txt', '_GFready_ind_all.txt'))

    gf_snps.to_csv(gf_snp_files['ind']['all'],
                   index=False,
                   sep='\t')

    print('\nsaved individual input to GF training script to:')
    print('\t', gf_snp_files['ind']['all'])

    return gf_snps


def calc_maf(locus):
    """From a series of individual counts of derived alleles, calculate minor allele frequency."""
    z12_counts = Counter(locus.astype(int))
    
    allele_counts = {
        'derived' : (2 * z12_counts[2]) + z12_counts[1],
        'ancestral' : (2 * z12_counts[0]) + z12_counts[1]
    }
    
    try:
        assert 2 * sum(z12_counts.values()) == sum(allele_counts.values())
    except AssertionError as e:
        print(z12_counts)
        print(allele_counts)
        print(sum(z12_counts.values()) , sum(allele_counts.values()))
        raise e
    
    minor_allele = 'derived' if allele_counts['derived'] <= allele_counts['ancestral'] else 'ancestral'
    
    maf = allele_counts[minor_allele]  / sum(allele_counts.values())
    
    return maf


def get_012(subset, muts):
    """Get genotypes in 012 format (counts of derived allele), filter for MAF.
    
    Returns
    -------
    dataframe as would be read in by original `snpfile`
        except:
            - filtered for MAF < 0.01 (see Notes)
            - transposed so rows=individuals, columns=loci (see `save_gf_snpfile`)
            - an added column called 'index' (see `save_gf_snpfile`)
    
    Notes
    -----
    filtering for MAF should be redundant as the input file `snpfile` should already be filtered
        but as of this version, only AF < 0.01 had been filtered.
        Even after filtering includes AF > 0.99, the code here in current form should still be accurate.
    """
    print(
        ColorText(
            '\nGetting 012 data ...'
        ).bold().custom('gold')
    )

    # read in the dataframe with individual counts of derived allele
    snpfile = op.join(slimdir, f'{seed}_Rout_Gmat_sample.txt')
    snps = pd.read_table(snpfile, delim_whitespace=True)
    
    # calculate MAF, remove low frequency SNPs from dataset
    tqdm.pandas(desc='calculating MAF')
    snps['maf'] = snps.progress_apply(lambda locus: calc_maf(locus), axis=1)
    print('\t', f"There are {sum(snps['maf'] < 0.01)} loci with MAF < 0.01 in the snps file.")
    
    # remove low frequency SNPs
    snps = snps[snps['maf'] >= 0.01]
    print('\t', f'After removing low MAF SNPs, there are {nrow(snps)} SNPs in the dataset.')

    # are all of the subset individuals in the z12 file?
    assert all(subset.index.isin(snps.columns))

    # are all of the z12 individuals in the subset set?
    assert all([ind in subset.index.tolist() for ind in snps.columns if ind != 'maf'])

    # do my maf calculations match those from Katie?
    muts_full = muts[['a_freq_subset', 'mutname']].copy()
    mut_maf = muts_full['a_freq_subset'].apply(lambda freq: freq if freq <=0.5 else 1-freq)  # convert to maf
    mut_maf = pd.Series(dict(zip(muts_full['mutname'], mut_maf)))
    try:
        assert all(mut_maf.loc[snps.index].round(4) == snps['maf'].round(4))
    except AssertionError as e:
        print(ColorText('my maf calculations do not match Katies to four decimals').bold().custom('red'))
        raise e

    # save for other purposes (eg RONA)
    print(ColorText('\nSaving 012 file ...').bold().custom('gold'))
    z12file = snpfile.replace('.txt', '_maf-gt-p01.txt')  # new file name = maf > 0.01
    snps.to_csv(z12file, sep='\t', index=True)

    # save for GF training script
    gf_snps = save_gf_snpfile(snps, subset, z12file)

    return z12file, gf_snps


def pop_freq(df):
    """For each locus, get frequency of derived allele for each pop.
    
    Notes
    -----
    popsamps is expected to be in globals() (eg dview) - see create_pop_freqs
    """
    from collections import defaultdict
    import pandas as pd

    pop_freqs = defaultdict(dict)
    for pop, samps in popsamps.items():
        pop_freqs[pop].update(
            dict(  # key = locus, val = pop_AF
                df[samps].apply(sum, axis=1) / (2*len(samps))  # calc pop frequency of derived allele
            )
        )

    return pd.DataFrame(pop_freqs)


def create_pop_freqs(subset, z12file):
    """Create population-level derived allele frequencies."""
    print(ColorText('\nCreating population-level derived allele frequencies ...').bold().custom('gold'))
    
    # assign samps to pop
    samppop = dict(zip(subset.index, subset.subpopID))
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()
    dview['popsamps'] = popsamps
    sleeping(10)

    # calc pop freqs in parallel using z12 file (counts of minor allele)
    jobs = parallel_read(z12file,
                         lview=lview,
                         dview=dview,
                         functions=create_fundict(pop_freq),
                         verbose=False,
                         index_col=0,
#                          delim_whitespace=True,  # cannot be set when setting `sep`; still works
                         sep=lib.no_default,  # work around for current implementation of parallel_read and get_skipto_df
                         maintain_dataframe=False)

    freqs = pd.concat(jobs).T  # rows=pops, cols=loci
    
    # do my freqs match those from Katie?
    mut_freqs = pd.read_table(op.join(slimdir, f'{seed}_Rout_af_pop.txt'), delim_whitespace=True)
    for locus in pbar(freqs.columns, desc='asserting frequency calculations'):  # iterate freqs in case of MAF filter
        assert all(freqs[locus] == mut_freqs[locus])
        
    freqs['index'] = freqs.index.tolist()  # for compatibility with gradient_training.R script
    print(f'\n{freqs.shape = }  (npop x (nloci + index_col))')
    
    # save for gradient forest
    gf_snp_files['pooled']['all'] = gf_snp_files['ind']['all'].replace('_ind_all.txt', '_pooled_all.txt')
    freqs.to_csv(gf_snp_files['pooled']['all'],
                 sep='\t',
                 index=False)

    # save for RONA - rows=loci, cols=pops
    rona_file = gf_snp_files['pooled']['all'].replace("_GFready_", "_RONAready_").replace('/gradient_forests/', '/RONA/')
    makedir(op.dirname(rona_file))
    freqs[freqs.columns[:-1]].T.to_csv(rona_file, sep='\t', index=True)  # before transposing, remove 'index' column

    return samppop, freqs


def create_rangefiles(subset, samppop):
    """Create range-wide data (for MVP, effectively the same as envdata)."""
    print(ColorText('\nCreating range files ...').bold().custom('gold'))

    # INDIVIDUAL-LEVEL DATA
    rangedata = subset[['y', 'x', 'sal_opt', 'temp_opt']].copy()
    rangedata.columns = ['lat', 'lon', 'sal_opt', 'temp_opt']

    # save
    gf_range_files['ind'] = op.join(training_filedir, f'{seed}_rangefile_GFready_ind.txt')
    rangedata.to_csv(gf_range_files['ind'], index=False, sep='\t')
    print(f"{gf_range_files['ind'] = }")

    # POOL-SEQ DATA
    # create intermediate data.frame
    _ = rangedata.copy()
    _['subpopID'] = _.index.map(samppop)

    # get pop-level data
    pool_rangedata = _.groupby('subpopID')[['lat', 'lon', 'sal_opt', 'temp_opt']].apply(np.mean)
    pool_rangedata.index.name = None  # remove index label

    # save
    gf_range_files['pooled'] = gf_range_files['ind'].replace('_ind.txt', '_pooled.txt')
    pool_rangedata.to_csv(gf_range_files['pooled'], index=False, sep='\t')
    print(f"{gf_range_files['pooled'] = }")

    return rangedata, pool_rangedata    


def create_envfiles(rangedata, pool_rangedata):
    """Create files for environmental data for Gradient Forests training."""
    print(ColorText('\nCreating envdata files ...').bold().custom('gold'))
    
    # INDIVIDUAL-LEVEL DATA
    envdata = rangedata[['sal_opt', 'temp_opt']].copy()
    # save
    gf_env_files['ind'] = op.join(training_filedir, f'{seed}_envfile_GFready_ind.txt')
    envdata.to_csv(gf_env_files['ind'], sep='\t', index=True)
    print(f"{gf_env_files['ind'] = }")

    # POOL-SEQ DATA
    pool_envdata = pool_rangedata[['sal_opt', 'temp_opt']].copy()
    gf_env_files['pooled'] = gf_env_files['ind'].replace('_ind.txt', '_pooled.txt')
    # save
    pool_envdata.to_csv(gf_env_files['pooled'],
                        sep='\t',
                        index=True)
    print(f"{gf_env_files['pooled'] = }")

    pass


def subset_adaptive_and_netural_loci(muts, gf_snps, freqs):
    """Identify adaptive and neutral loci via the muts file, save to file, subset ind and pooled data."""
    print(ColorText('\nSubsetting adaptive and neutral loci ...').bold().custom('gold'))
    
    for locus_type in ['adaptive', 'neutral']:
        # identify loci
        if locus_type == 'adaptive':
            loci = muts.index[muts['mutID'] != 1]
        else:
            loci = muts.index[(muts['causal_temp'] == 'neutral') & (muts['causal_sal'] == 'neutral')]
        
        # save file
        locus_file = op.join(training_filedir, f'{seed}_{locus_type}_loci.pkl')
        pkldump(loci, locus_file)
        print(f'{locus_type} loci file = {locus_file}')
        # symlink for RONA
        try:
            dst = locus_file.replace('/gradient_forests/', '/RONA/')
            os.symlink(locus_file, dst)
        except FileExistsError as e:  # if the symlink has already been created
            pass
        except FileNotFoundError as e:  # perhaps user did not say they wanted to run RONA
            makedir(op.dirname(dst))
            os.symlink(locus_file, dst)
            
        # subset Gradient Forest training data and save
        gf_loci = gf_snps[list(loci) + ['index']].copy()
        print(f'{locus_type} {gf_loci.shape = }')
        
        gf_snp_files['ind'][locus_type] = gf_snp_files['ind']['all'].replace("_all.txt", f"_{locus_type}.txt")
        
        gf_loci.to_csv(gf_snp_files['ind'][locus_type],
                       sep='\t',
                       index=False)
        
        print(f"{gf_snp_files['ind'][locus_type] = }")
        
        # subset pooled Gradient Forest data and save
        pooled_gf_loci = freqs[loci].copy()
        pooled_gf_loci['index'] = pooled_gf_loci.index.tolist()
        print(f'{locus_type} {pooled_gf_loci.shape = }')
        
        gf_snp_files['pooled'][locus_type] = gf_snp_files['pooled']['all'].replace("_all.txt", f"_{locus_type}.txt")

        pooled_gf_loci.to_csv(gf_snp_files['pooled'][locus_type],
                              sep='\t',
                              index=False)

        print(f"{gf_snp_files['pooled'][locus_type] =}")    

    pass


def create_training_shfiles():
    """Create slurm sbatch files to submit GF training jobs to queue."""
    print(ColorText('Creating slurm sbatch files ...').bold().custom('gold'))

    mytime = {
        'ind': {'all': '5-00:00:00',
                'adaptive': '1:00:00',
                'neutral' : '2-00:00:00'},
        
        'pooled': {'all': '1-00:00:00',
                   'adaptive': '1:00:00',
                   'neutral' : '12:00:00'}
    }

    mymem = {
        'ind': {'all': '900000M',
                'adaptive': '4000M',
                'neutral' : '400000M'},
        
        'pooled': {'all': '300000M',
                   'adaptive': '4000M',
                   'neutral': '150000M'}
    }

    partition = wrap_defaultdict(lambda: 'short', 2)
    partition['ind']['all'] = 'long'
    partition['ind']['neutral'] = 'long'
    partition['pooled']['neutral'] = 'short'
    partition['pooled']['adaptive'] = 'lotterhos'

    shfiles = []
    for ind_or_pooled in ['ind', 'pooled']:
        for marker_set in ['all', 'adaptive', 'neutral']:  # all loci, those under selection, or neutral
            basename = f'{seed}_GF_training_{ind_or_pooled}_{marker_set}'
            shfile = op.join(shdir, f'{basename}.sh')

            # set up variables
            _time = mytime[ind_or_pooled][marker_set]
            _mem = mymem[ind_or_pooled][marker_set]
            _partition = partition[ind_or_pooled][marker_set]
            _snpfile = op.basename(gf_snp_files[ind_or_pooled][marker_set])
            _envfile = op.basename(gf_env_files[ind_or_pooled])
            _rangefile = op.basename(gf_range_files[ind_or_pooled])

            shtext = f'''#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time={_time}
#SBATCH --mem={_mem}
#SBATCH --partition={_partition}
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc  # assumed that conda init is within .bashrc
conda deactivate
conda activate r35

cd {training_filedir}

{rscript_exe} \\
{training_script} \\
{_snpfile} \\
{_envfile} \\
{_rangefile} \\
{basename} \\
{outfile_dir}

'''
            with open(shfile, 'w') as o:
                o.write(shtext)
            shfiles.append(shfile)

    return shfiles


def submit_jobs(shfiles):
    """Submit training jobs to slurm, queue up fitting and validation scripts to run as soon as training completes."""
    print(ColorText('\nSubmitting training scripts to slurm ...').bold().custom('gold'))
    
    pids = sbatch(shfiles)

    # sbatch jobs to fit GF models to common garden climates, then to validate GF
    shtext = '\n'.join(
        [f'cd {mvp_dir}',
         '',
         'source $HOME/.bashrc',
         '',
         'conda activate mvp_env',
         '',
         f'python MVP_02_fit_gradient_forests.py {seed} {slimdir} {outfile_dir} {rscript_exe}',
         '',
         f'python MVP_03_validate_gradient_forests.py {seed} {slimdir} {outdir}/gradient_forests',
         ''
        ]
    )

    create_watcherfile(pids,
                       fitting_shdir,
                       watcher_name=f'{seed}_gf_fitting',
                       time='1-00:00:00',
                       ntasks=1,
                       rem_flags=['#SBATCH --nodes=1', '#SBATCH --cpus-per-task=7'],
                       mem='300000M',
                       begin_alert=False,
                       end_alert=False,
                       added_text=shtext)
    
    pass


def main():
    # get the individuals that were subsampled from full simulation
    subset = read_ind_data(slimdir, seed)

    # read in the seed_Rout_muts_full.txt file, name loci
    muts = read_muts_file()

    # get counts of minor allele for each sample
    z12file, gf_snps = get_012(subset, muts)

    # calc allele freqs per pop
    samppop, freqs = create_pop_freqs(subset, z12file)

    # create range files for script and save
    rangedata, pool_rangedata = create_rangefiles(subset, samppop)

    # create envdata for the script and save
    create_envfiles(rangedata, pool_rangedata)

    # subset data for adaptive and neutral loci
    subset_adaptive_and_netural_loci(muts, gf_snps, freqs)

    # create shfiles to train Gradient Forests
    shfiles = create_training_shfiles()
    
    # submit jobs to queue
    submit_jobs([sh for sh in shfiles if '_ind_' not in op.basename(sh)])   # REMOVE TO ALLOW INDIVIDUAL-LEVEL SBATCHING

    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, outdir, num_engines, rscript_exe, email = sys.argv
    
    mvp_dir = op.dirname(op.abspath(thisfile))

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))
    training_script = op.join(mvp_dir, 'MVP_gf_training_script.R')

    # set up timer
    t1 = dt.now()

    # for looking up file names - of the form dict[ind_or_pooled][marker_set]
    gf_snp_files = defaultdict(dict)
    gf_range_files = defaultdict(dict)
    gf_env_files = defaultdict(dict)

    # create dirs
    training_filedir, shdir, outfile_dir, fitting_shdir = make_gf_dirs(outdir)

    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=int(num_engines), profile=f'GF_{seed}')

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
