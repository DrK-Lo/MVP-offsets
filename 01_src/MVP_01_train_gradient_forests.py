"""Train Gradient Forests using simulations from the MVP project.

Usage
-----
python 01_train_gradient_forests.py seed, slimdir, outdir, num_engines, rscript, imports_dir, email

Parameters
----------
seed - the seed number of the simulation - used to find associated files
slimdir - the location of the seed's files output by Katie's post-processing scripts
outdir - where to save files
num_engines - the number of engines to start for parallelizing tasks
rscript - path to R environment's Rscript executable - eg ~/anaconda3/envs/r35/bin/Rscript
imports_dir - path to imports.R from github.com/brandonlind/r_imports
email - email to receive sbatch notifications

Dependencies
------------
- dependent upon code from github.com/brandonlind/pythonimports

Notes
-----
- to be able to submit the individual training using all loci, users will need access to both the
    long and large partitions on the discovery cluster at NEU

TODO
----
- create code to create .txt file from vcf file
"""
from pythonimports import *


def make_gf_dirs(outerdir) -> tuple:
    """Create some dirs to put infiles, outfiles, and figs into."""
    print(ColorText('\nCreating directories ...').bold().custom('gold'))
    directory = makedir(op.join(outerdir, 'gradient_forests'))
    training_filedir = makedir(op.join(directory, 'training/training_files'))
    shdir = makedir(op.join(directory, 'training/training_shfiles'))
    outfile_dir = makedir(op.join(directory, 'training/training_outfiles'))

    return training_filedir, shdir, outfile_dir


def read_ind_data(slimdir, seed) -> pd.DataFrame:
    """Get the individuals that were subsampled from full simulation."""
    print(ColorText('\nReading in info for subsampled individuals ...').bold().custom('gold'))
    subset = pd.read_table(op.join(slimdir, f'{seed}_Rout_ind_subset.txt'), delim_whitespace=True)
    subset.index = ('i' + subset['indID'].astype(str)).tolist()  # this will match to the 'causal' file
    subset['sample_name'] = subset.index.tolist()

    return subset


def update_locus(locus: str, locus_list: list) -> str:
    """Since sims can simulate mutations at same 'locus', create new name for duplicate loci names."""
    matches = []
    for name in locus_list:
        prefix, *suffix = name.split('_')  # only my update loci names will have an underscore
        if prefix == locus:
            matches.append(name)

    if len(matches) > 0:
        # update locus name if there are duplicates
        locus = f'{locus}_{len(matches)+1}'

    return locus


def read_muts_file() -> pd.DataFrame:
    """Read in the seed_Rout_muts_full.txt file, convert `VCFrow` to 0-based, name loci."""
    print(ColorText('\nReading muts file ...').bold().custom('gold'))
    # get path to file
    muts_file = op.join(slimdir, f'{seed}_Rout_muts_full.txt')

    # read in the table
    muts = pd.read_table(muts_file, delim_whitespace=True, low_memory=False)  # low_memory to avoid mixed dtypes warning

    # convert to 0-based indexing for python
    assert 0 not in muts['VCFrow'].tolist()
    muts['VCFrow'] = muts['VCFrow'] - 1  # convert to 0-based python

    # update locus names
    found = []
    for row in muts.index:
        locus = 'LG' + \
                muts.loc[row, 'LG'].astype(int).astype(str) + \
                '-' + \
                muts.loc[row, 'pos_pyslim'].astype(int).astype(str)
        if locus in found:
            locus = update_locus(locus, found)
        found.append(locus)

    # update index with locus names
    muts.index = found

    # make sure no duplicate locus names remain
    assert luni(muts.index) == nrow(muts)

    return muts


def convert_012(df: pd.DataFrame, inds: list) -> pd.DataFrame:
    """Convert individual names to i-format, genotypes to counts of minor allele, and subset for `inds`.

    Parameters
    ----------
    - df : pandas.DataFrame from a file of type seed_plusneut_MAF01.recode2.vcf.txt
    - inds : list of i-formatted sample names (eg i0, i1, i2) to filter from full VCF; these are the 
        individuals that were subsampled from the full simulation.
    """
    from collections import Counter
    from tqdm import tqdm as pbar  # progress bar
    from pythonimports import flatten

    # first convert sample names (change eg tsk_0 to i0; tsk_25 to i25)
    firstcols = []
    newcols = []
    for col in df.columns:
        if col.startswith('tsk_'):
            col = col.replace('tsk_', 'i')  # convert to eg i0, i1, i2 ...
        else:
            firstcols.append(col)
        newcols.append(col)
    df.columns = newcols

    # subset for inds in `inds`
    df = df[firstcols + inds]

    # assert genotype convention = that all genotypes for each individual contain "|"
    assert all(  # assert for all individuals
        df[inds].apply(
            lambda gts: all(['|' in gt for gt in gts]),  # all genotypes contain "|"
            axis=1
        )
    )

    # figure out minor allele counts for each individual and across all individuals
    for locus in pbar(df.index, desc='determining minor allele'):
        # count each allele across all samples for `locus`
        allele_counts = Counter(
            flatten(
                [list(gt.replace("|", "")) for gt in df.loc[locus, inds]]  # technically don't need to replace |
            )
        )

        # identify minor allele
        if allele_counts['0'] < allele_counts['1']:
            minor_allele = '0'
        else:
            minor_allele = '1'

        # get minor allele counts for each individual
        df.loc[locus, inds] = [gt.count(minor_allele) for gt in df.loc[locus, inds]]

        # calculate MAF and AF
        df.loc[locus, 'MAF'] = allele_counts[minor_allele] / (2*len(inds))
        df.loc[locus, 'AF'] = allele_counts['1'] / (2*len(inds))  # '1' is ALT/derived allele

    # replace metadata
    df.loc[df.index, 'FORMAT'] = 'minor_allele_count'

    # assert expectations
    assert max(df['MAF']) <= 0.50
    assert all((0 <= df['AF']) & (df['AF'] <= 1))

    return df


def save_gf_snpfile(snps, subset, snpfile) -> pd.DataFrame:
    """Format ans save snp data for gradient forests training script from Lind et al."""
    # transpose so rows=individuals, columns=loci
    gf_snps = snps[subset.index.tolist()].T.copy()  # remove non-individual columns

    # add index col needed for gradient_training.R script
    gf_snps['index'] = gf_snps.index.tolist()

    # save
    gf_snp_files['ind']['all'] = op.join(training_filedir,
                                         op.basename(snpfile).replace('.txt', '_GFready_ind_all.txt'))

    gf_snps.to_csv(gf_snp_files['ind']['all'],
                   index=False,
                   sep='\t')

    print(gf_snp_files['ind']['all'])

    return gf_snps


def get_012(muts, subset) -> pd.DataFrame:
    """Get genotypes from vcf, convert to counts of global minor allele (012) in parallel."""
    print(
        ColorText(
            '\nConverting genotypes to counts of global minor allele using parallel engines ...'
        ).bold().custom('gold')
    )
    # map VCF index to locus name so we can assign locus names to the vcf.txt file
    vcf_index_to_locus = defaultdict(lambda: 'no_name')
    vcf_index_to_locus.update(
        dict(zip(muts['VCFrow'], muts.index))
    )

    # update function to apply during parallelization
    functions = create_fundict(convert_012,
                               kwargs={'inds': subset.index.tolist()})

    # read in the dataframe in chunks, apply function to each chunk in parallel
    snpfile = op.join(slimdir, f'{seed}_plusneut_MAF01.recode2.vcf.txt')
    snps = parallel_read(snpfile,
                         lview=lview,
                         dview=dview,
                         functions=functions,
                         verbose=False)

    # set locus names as row names
    loci = snps.index.map(vcf_index_to_locus)
    snps.index = loci.tolist()

    # any snp with name='no_name' should be excluded because MAF < 0.01
    assert snps[snps.index == 'no_name']['MAF'].min() > 0
    assert snps[snps.index == 'no_name']['MAF'].max() < 0.01

    # remove loci with low MAF
    snps = snps[snps.index != 'no_name']

    print('\t', f'There are {nrow(snps)} loci with MAF > 0.01 in the snps file.')

    # are all of the subset individuals in the z12 file? A: yes!
    assert all(subset.index.isin(snps.columns))

    # are all of the z12 individuals in the subset set? A: yes!
    assert all([ind in subset.index.tolist() for ind in snps.columns if ind.startswith('i')])

    # save for other purposes (eg RONA)
    print(ColorText('\nSaving 012 file ...').bold().custom('gold'))
    z12file = snpfile.replace('.txt', '_012.txt')
    snps.to_csv(z12file, sep='\t', index=True)

    # save for GF training script
    gf_snps = save_gf_snpfile(snps, subset, snpfile)

    return z12file, gf_snps


def pop_freq(df: pd.DataFrame) -> pd.DataFrame:
    """For each locus, get MAF for each pop."""
    from collections import defaultdict
    import pandas as pd

    pop_freqs = defaultdict(dict)
    for pop, samps in popsamps.items():
        pop_freqs[pop].update(
            dict(  # key = locus, val = pop_MAF
                df[samps].apply(sum, axis=1) / (2*len(samps))  # calc frequency of minor allele
            )
        )

    return pd.DataFrame(pop_freqs)


def create_pop_freqs(subset, z12file):
    """Create population-level MAF frequencies."""
    print(ColorText('\nCreating population-level MAF frequencies ...').bold().custom('gold'))
    # assign samps to pop
    samppop = dict(zip(subset.index, subset.subpopID))
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()
    dview['popsamps'] = popsamps
    time.sleep(10)

    # calc pop freqs in parallel using z12 file (counts of minor allele)
    jobs = parallel_read(z12file,
                         lview=lview,
                         dview=dview,
                         functions=create_fundict(pop_freq),
                         verbose=False,
                         index_col=0,
                         maintain_dataframe=False)

    freqs = pd.concat(jobs).T
    freqs['index'] = freqs.index.tolist()  # for compatibility with gradient_training.R script

    print(f'\n{freqs.shape = }  (npop x (nloci + index_col))')

    # save for gradient forest
    gf_snp_files['pooled']['all'] = gf_snp_files['ind']['all'].replace('_ind_all.txt', '_pooled_all.txt')
    freqs.to_csv(gf_snp_files['pooled']['all'],
                   sep='\t',
                   index=False)
    # save for RONA
    rona_file = gf_snp_files['pooled']['all'].replace("_GFready_", "_RONAready_").replace('/gradient_forests/', '/RONA/')
    makedir(op.dirname(rona_file))
    freqs[freqs.columns[:-1]].T.to_csv(rona_file, sep='\t', index=True)  # before transposing, remove 'index' column


    return samppop, freqs


def create_rangefiles(subset, samppop):
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


def create_envfiles(rangedata, pool_rangedata) -> None:
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


def subset_adaptive_loci(muts, gf_snps, freqs):
    """Identify adaptive loci via the muts file, save to file, subset ind and pooled data."""
    print(ColorText('\nSubsetting adaptive loci ...').bold().custom('gold'))
    # identify the loci under selection
    adaptive_loci = muts.index[muts['mutID'] != 1]

    # save adaptive loci
    locus_file = op.join(training_filedir, f'{seed}_adaptive_loci.pkl')
    pkldump(adaptive_loci, locus_file)
    print(f'adaptive loci file = {locus_file}')

    # subset Gradient Forest training data and save
    adaptive_gf_snps = gf_snps[list(adaptive_loci) + ['index']].copy()
    print(f'{adaptive_gf_snps.shape = }')

    gf_snp_files['ind']['adaptive'] = gf_snp_files['ind']['all'].replace("_all.txt", "_adaptive.txt")

    adaptive_gf_snps.to_csv(gf_snp_files['ind']['adaptive'],
                            sep='\t',
                            index=False)

    print(f"{gf_snp_files['ind']['adaptive'] = }")

    # subset pooled Gradient Forest data and save
    pooled_adaptive_gf_snps = freqs[adaptive_loci].copy()
    pooled_adaptive_gf_snps['index'] = pooled_adaptive_gf_snps.index.tolist()
    print(f'{pooled_adaptive_gf_snps.shape = }')

    gf_snp_files['pooled']['adaptive'] = gf_snp_files['pooled']['all'].replace("_all.txt", "_adaptive.txt")

    pooled_adaptive_gf_snps.to_csv(gf_snp_files['pooled']['adaptive'],
                                   sep='\t',
                                   index=False)

    print(f"{gf_snp_files['pooled']['adaptive'] =}")

    pass


def create_training_shfiles():
    mytime = {'ind': {'all': '7-00:00:00', 'adaptive': '1:00:00'},
              'pooled': {'all': '23:00:00', 'adaptive': '1:00:00'}}

    mymem = {'ind': {'all': '25000M', 'adaptive': '4000M'},
             'pooled': {'all': '150000M', 'adaptive': '4000M'}}

    shfiles = []
    for ind_or_pooled in ['ind', 'pooled']:
        for all_or_adaptive in ['all', 'adaptive']:  # all loci or only those under selection
            basename = f'{seed}_GF_training_{ind_or_pooled}_{all_or_adaptive}'
            shfile = op.join(shdir, f'{basename}.sh')

            # set up variables
            _time = mytime[ind_or_pooled][all_or_adaptive]
            _mem = mymem[ind_or_pooled][all_or_adaptive]
            _snpfile = op.basename(gf_snp_files[ind_or_pooled][all_or_adaptive])
            _envfile = op.basename(gf_env_files[ind_or_pooled])
            _rangefile = op.basename(gf_range_files[ind_or_pooled])

            shtext = f'''#!/bin/bash
#SBATCH --job-name={basename}
#SBATCH --time={_time}
#SBATCH --mem={_mem}
#SBATCH --output={basename}_%j.out
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL

source $HOME/.bashrc
conda deactivate
conda activate r35

cd {training_filedir}

{rscript} \\
{training_script} \\
{_snpfile} \\
{_envfile} \\
{_rangefile} \\
{basename} \\
{outfile_dir} \\
{imports_dir}

'''
            with open(shfile, 'w') as o:
                o.write(shtext)
            shfiles.append(shfile)

    return shfiles


def main():
    # get the individuals that were subsampled from full simulation
    subset = read_ind_data(slimdir)

    # read in the seed_Rout_muts_full.txt file, convert `VCFrow` to 0-based, name loci
    muts = read_muts_file()

    # get counts of minor allele for each sample
    z12file, gf_snps = get_012(muts, subset)

    # calc allele freqs per pop
    samppop, freqs = create_pop_freqs(subset, z12file)

    # create range files for script and save
    rangedata, pool_rangedata = create_rangefiles(subset, samppop)

    # create envdata for the script and save
    create_envfiles(rangedata, pool_rangedata)

    # subset data for adaptive loci
    subset_adaptive_loci(muts, gf_snps, freqs)

    # create shfiles to train Gradient Forests
    shfiles = create_training_shfiles()

    # sbatch training files, create job to email once training jobs complete
    print(ColorText('\nSubmitting training scripts to slurm ...').bold().custom('gold'))
    pids = sbatch(shfiles[::-1])  # reverse sbatching so I can avoid sbatching individual-all training sets
    create_watcherfile(pids, shdir, 'gf_training_watcher', email)

    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, outdir, num_engines, rscript, imports_dir, email = sys.argv

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))
    training_script = op.join(op.dirname(thisfile), 'MVP_gf_training_script.R')

    # set up timer
    t1 = dt.now()

    # for looking up file names - of the form dict[ind_or_pooled][all_or_adaptive]
    gf_snp_files = defaultdict(dict)
    gf_range_files = defaultdict(dict)
    gf_env_files = defaultdict(dict)

    # create dirs
    training_filedir, shdir, outfile_dir = make_gf_dirs(outdir)

    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=int(num_engines))

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
