"""Train Gradient Forests using simulations from the MVP project.

Usage
-----
python 01_train_gradient_forests.py seed filedir outdir

Parameters
----------
seed - the seed number of the simulation - used to find associated files
slimdir - the location of the seed's files output by Katie's post-processing scripts
outdir - where to save files
engines - the number of engines to start for parallelizing tasks

TODO
----
- create code to create .txt file from vcf file
"""
from pythonimports import *


def make_gf_dirs(outdir) -> tuple:
    """Create some dirs to put infiles, outfiles, and figs into."""
    print('\n', ColorText('Creating directories ...').bold())
    DIR = makedir(op.join(outdir, 'training'))
    training_dir = makedir(op.join(DIR, 'gradient_forests/training_files'))
    shdir = makedir(op.join(DIR, 'gradient_forests/training_shfiles'))
    outdir = makedir(op.join(DIR, 'gradient_forests/training_outfiles'))
    
    return training_dir, shdir, outdir


def read_ind_data() -> pd.DataFrame:
    """Get the subset of individuals that were subsampled from full simulation."""
    subset = pd.read_table(op.join(slimdir, f'{seed}_Rout_ind_subset.txt'), delim_whitespace=True)
    subset.index = ('i' + subset['indID'].astype(str)).tolist()  # this will match to the 'causal' file
    subset['sample_name'] = subset.index.tolist()
    
    return subset


def update_locus(locus:str, locus_list:list) -> str:
    """Since sims can simulate mutations at same 'locus', create new name for duplicate loci names."""
    matches = []
    for name in locus_list:
        prefix,*suffix = name.split('_')  # only my update loci names will have an underscore
        if prefix==locus:
            matches.append(name)

    if len(matches) > 0:
        # update locus name if there are duplicates
        locus = f'{locus}_{len(matches)+1}'

    return locus


def read_muts_file() -> pd.DataFrame:
    """Read in the seed_Rout_muts_full.txt file, convert `VCFrow` to 0-based, name loci."""
    print('\n', ColorText('Reading muts file ...').bold())
    # get path to file
    muts_file = op.join(slimdir, f'{seed}_Rout_muts_full.txt')

    # read in the table
    muts = pd.read_table(muts_file, delim_whitespace=True)

    # convert to 0-based indexing for python
    assert 0 not in muts['VCFrow'].tolist()
    muts['VCFrow'] = muts['VCFrow'] - 1  # convert to 0-based python

    # update locus names
    found = []
    for row in muts.index:
        locus = 'LG' + \
                muts.loc[row, 'LG'].astype(str) + \
                '-' + \
                muts.loc[row, 'pos_pyslim'].astype(str)
        if locus in found:
            locus = update_locus(locus, found)
        found.append(locus)

    # update index with locus names
    muts.index = found

    # make sure no duplicate locus names remain
    assert luni(muts.index) == nrow(muts)

    return muts


def convert_012(df:pd.DataFrame, inds:list) -> pd.DataFrame:
    """Convert individual names to i-format, genotypes to counts of minor allele, and subset for `inds`.
    
    Parameters
    ----------
    - df : pandas.DataFrame from a file of type seed_plusneut_MAF01.recode2.vcf.txt
    - inds : list of i-formatted sample names (eg i0, i1, i2) to filter from full VCF
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

def save_gf_snpfile(snps, subset, snpfile) -> None:
    """Save for gradient forests training script from Lind et al."""
    # transpose so rows=individuals, columns=loci
    gf_snps = snps[subset.index.tolist()].T.copy()  # remove non-individual columns

    # add index col needed for gradient_training.R script
    gf_snps['index'] = gf_snps.index.tolist()

    # save
    gf_snp_files['ind']['all'] = op.join(training_dir,
                                         op.basename(snpfile).replace('.txt', '_GFready_ind_all.txt'))

    gf_snps.to_csv(gf_snp_files['ind']['all'],
                   index=False,
                   sep='\t')

    print(gf_snp_files['ind']['all'])

    pass


def get_012(muts, subset) -> pd.DataFrame:
    """Get genotypes from vcf, convert to 012 in parallel."""
    print('\n', ColorText('Converting genotypes to counts of global minor allele using parallel engines ...').bold())
    # map VCF index to locus name so we can assign locus names to the vcf.txt file
    VCF_index_to_locus = defaultdict(lambda: 'no_name')
    VCF_index_to_locus.update(
        dict(zip(muts['VCFrow'], muts.index))
    )
    
    # update function to apply during parallelization
    functions = create_fundict(convert_012,
                               kwargs={'inds' : subset.index.tolist()})

    # read in the dataframe in chunks, apply function to each chunk in parallel
    snpfile = op.join(slimdir, f'{seed}_plusneut_MAF01.recode2.vcf.txt')
    snps = parallel_read(snpfile,
                         lview=lview,
                         dview=dview,
                         functions=functions,
                         verbose=False)

    # set locus names as row names
    loci = snps.index.map(VCF_index_to_locus)
    snps.index = loci.tolist()
    
    # any snp with name='no_name' should be excluded because MAF < 0.01
    assert snps[snps.index=='no_name']['MAF'].min() > 0
    assert snps[snps.index=='no_name']['MAF'].max() < 0.01
    
    # remove loci with low MAF
    snps = snps[snps.index != 'no_name']
    
    print('\t', f'There are {nrow(snps)} with MAF > 0.01 in the snps file.')
    
    # are all of the subset individuals in the z12 file? A: yes!
    assert all(subset.index.isin(snps.columns))
    
    # are all of the z12 individuals in the subset set? A: yes!
    assert all([ind in subset.index.tolist() for ind in snps.columns if ind.startswith('i')])
    
    # save for other purposes (eg RONA)
    print('\n', ColorText('Saving 012 file ...').bold())
    z12file = snpfile.replace('.txt', '_012.txt')
    snps.to_csv(z12file, sep='\t', index=True)
    
    # save for GF training script
    save_gf_snpfile(snps, subset, snpfile)
    
    return z12file


def pop_freq(df:pd.DataFrame) -> pd.DataFrame:
    """For each locus, get MAF for each pop."""
    from collections import defaultdict
    import pandas as pd
    
    pop_freqs = defaultdict(dict)
    for pop,samps in popsamps.items():
        pop_freqs[pop].update(
            dict(  # key = locus, val = pop_MAF
                df[samps].apply(sum, axis=1) / (2*len(samps))  # count frequency of minor allele
            )
        )

    return pd.DataFrame(pop_freqs)


def create_pop_freqs(subset, z12file):
    """Create population-level MAF frequencies."""
    # assign samps to pop
    samppop = dict(zip(subset.index, subset.subpopID))
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()
    dview['popsamps'] = popsamps; time.sleep(10)

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

    print(f'\n{freqs.shape = }')

    # save AFTER TRANSPOSING so that subpopID are columns
    gf_snp_files['pooled']['all'] = gf_snp_files['ind']['all'].replace('_ind_all.txt', '_pooled_all.txt')

    freqs.T.to_csv(gf_snp_files['pooled']['all'],
                 sep='\t',
                 index=True)

    return samppop, popsamps, freqs


def create_rangefiles(subset):
    
    Print('\n', ColorText('Creating range files ...').bold())
    
    ### INDIVIDUAL-LEVEL DATA ###
    rangedata = subset[['y', 'x', 'sal_opt', 'temp_opt']].copy()
    rangedata.columns = ['lat', 'lon', 'sal_opt', 'temp_opt']
    print(f'{nrow(rangedata) = }')
    
    # save
    gf_range_files['ind'] = op.join(filedir, f'{seed}_rangefile_GFready_ind.txt')
    rangedata.to_csv(gf_range_files['ind'], index=False, sep='\t')
    print(f"{gf_range_files['ind'] = }")
    
    ### POOL-SEQ DATA  ###
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
    ### INDIVIDUAL-LEVEL DATA ###
    envdata = rangedata[['sal_opt', 'temp_opt']].copy()
    # save
    gf_env_files['ind'] = op.join(filedir, f'{seed}_envfile_GFready_ind.txt')
    envdata.to_csv(gf_env_files['ind'], sep='\t', index=True)
    print(f"{gf_env_files['ind'] = }")
    
    ### POOL-SEQ DATA ###
    pool_envdata = pool_rangedata[['sal_opt', 'temp_opt']].copy()
    gf_env_files['pooled'] = gf_env_files['ind'].replace('_ind.txt', '_pooled.txt')
    # save
    pool_envdata.to_csv(gf_env_files['pooled'],
                        sep='\t',
                        index=True)
    print(f"{gf_env_files['pooled'] = }")


def main():
    # get the subset of individuals that were subsampled from full simulation
    subset = read_ind_data()
    
    # Read in the seed_Rout_muts_full.txt file, convert `VCFrow` to 0-based, name loci
    muts = read_muts_file()
    
    # get counts of minor allele for each sample
    z12file = get_012(muts, subset)
    
    # calc allele freqs per pop
    samppop, popsamps, freqs = create_pop_freqs(subset, z12file)
    
    # create range files for script and save
    rangedata, pool_rangedata = create_rangefiles(subset)
    
    pass


if __name__ == '__main__':
    thisfile, seed, slimdir, outdir, num_engines = sys.argv
    
    # for looking up file names - of the form dict[ind_or_pooled][all_or_adaptive]
    gf_snp_files = defaultdict(dict)
    gf_range_files = defaultdict(dict)
    gf_env_files = defaultdict(dict)
    
    # create dirs
    training_dir, sh_dir, outfile_dir = make_gf_dirs(outdir)
    
    # start cluster
    print('\n', ColorText('Starting engines ...').bold())
    lview, dview, cluster_id = start_engines(n=int(num_engines))
    
    # print versions of packages and environment
    print('\n', ColorText('Environment info :').bold())
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()
    