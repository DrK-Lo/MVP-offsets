"""Calculate population pairwise FST according to Weir & Cockerham 1984.

Usage
-----
conda activate mvp_env
python MVP_07_calc_WC_pairwise_FST.py seed slimdir gf_training_dir num_engines

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scripts
num_engines
    number of engines to use to parallelize calculations necessary to estimate FST

Notes
-----
- negative FST values are set to 0
- diagonal of FST matrix is set to 0 (pops for columns pops for rows)

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon code from github.com/brandonlind/pythonimports

TODO
----
- create FST map figure

"""
from pythonimports import *
from MVP_01_train_gradient_forests import read_ind_data

import allel


def get_gt_data():
    """Get genotype data in format compatible for sci-kit allel."""
    print(ColorText(
        '\nConverting minor allele counts to scikit.allel format ...'
    ).bold().custom('gold'))

    # map derived allele count to genotypes for scikit allel
    z12_trans = {
        0 : [0,0],  # zero derived alleles almost always means homozygous REF
        1 : [0,1],
        2 : [1,1]
    }

    # read in file with genotypes as counts of global minor allele
#     z12file = op.join(slimdir, f'{seed}_plusneut_MAF01.recode2.vcf_012.txt')
    z12file = op.join(slimdir, f'{seed}_Rout_Gmat_sample_maf-gt-p01.txt')
    df = pd.read_table(z12file, index_col=0)
    df = df[[col for col in df.columns if not col=='maf']]

    # translate 012 to scikit allel format
    df = df.apply(lambda series: series.map(z12_trans))

    # load the adaptive loci
    adaptive_loci = pklload(op.join(gf_training_dir, f'{seed}_adaptive_loci.pkl'))
    
    # load the neutral loci
    neutral_loci = pklload(op.join(gf_training_dir, f'{seed}_neutral_loci.pkl'))

    # create subsets (technically `loci_set`=all is not a subset)
    gts = {'all': df.copy(),
           'adaptive': df.loc[adaptive_loci].copy(),
           'neutral': df.loc[neutral_loci].copy()}

    return gts


def get_gt_array(pop_i=None, pop_j=None, loci_set=None) -> allel.GenotypeArray:
    """Subset full data to get population data for `pop_i` and `pop_j`.
    
    Notes
    -----
    - returned GenotypeArray does not maintain sample or locus names as indices
    
    """
    import allel
    from pythonimports import flatten
    
    # get genotypes
    gts = mygts[loci_set].copy()
    
    # get the list of samples from each population
    samples = flatten(
        [
            popidx[pop_i] , popidx[pop_j]
        ]
    )
    
    # create GenotypeArray from these two populations
    gtarr = allel.GenotypeArray(
        list([list(y) for y in gts[samples].values.tolist()])
    )
    
    return gtarr


def calc_fst(pop_i=None, pop_j=None, loci_set=None):
    """Calculate population pairwise FST between `pop_i` and `pop_j`."""
    import allel
    import numpy as np
    
    gt_arr = get_gt_array(pop_i=pop_i, pop_j=pop_j, loci_set=loci_set)
    
    a,b,c = allel.weir_cockerham_fst(
        gt_arr,
        subpops=[range(10), range(10, 20)]
    )
    
    fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
    
    # correct any negative values
    if fst < 0:
        fst = 0
    
    return fst


def get_pairwise_fst(popidx, mygts):
    """Compute pairwise FST in parallel."""
    print(ColorText('\nCalculating pairwise FST in parallel ...').bold().custom('gold'))
    pairs = wrap_defaultdict(None, 3)
    args = []
    jobs = []
    for i, pop_i in enumerate(popidx.keys()):
        for j, pop_j in enumerate(popidx.keys()):
            if i < j:  # only unique comparisons
                for loci_set in mygts.keys():
                    args.append((pop_i, pop_j, loci_set))
                    jobs.append(
                        lview.apply_async(
                            calc_fst, **{'pop_i' : pop_i,
                                         'pop_j' : pop_j,
                                         'loci_set' : loci_set}
                        )
                    )
            elif i == j:
                for loci_set in mygts.keys():
                    # add in these diagonal vals to zero now
                    pairs[loci_set][pop_i][pop_j] = 0  # do not need to set pairs[loci_set][pop_j][pop_i] because i==j

    # wait for jobs to finish                    
    watch_async(jobs)
    
    # retrieve results
    for i,job in enumerate(jobs):
        pop_i, pop_j, loci_set = args[i]
        pairs[loci_set][pop_i][pop_j] = job.r
        pairs[loci_set][pop_j][pop_i] = job.r

    # convert to DataFrame and sort columns and rows by subpopID (order is not necessary for pd.DataFrame.corrwith)
    for loci_set in keys(pairs):
        pairs[loci_set] = pd.DataFrame(pairs[loci_set]).sort_index(axis=0).sort_index(axis=1)
    
    return pairs


def main():
    # get data for the individuals that were subsampled from full simulation
    subset = read_ind_data(slimdir, seed)

    # get the column indices for each population within each allel.GenotypeArray
    subset['index'] = subset.index.tolist()
    popidx = subset.groupby('subpopID')['index'].apply(list).to_dict()  # key = subpopID val = list of indices

    # get allel-formatted genotypes
    mygts = get_gt_data()

    # load data and functions to engines, give them time to load
    dview['get_gt_array'] = get_gt_array
    dview['popidx'] = popidx
    dview['mygts'] = mygts
    sleeping(10, desc='\tsleeping while data loads to engines')

    # calculate population pairwise FST using parallel engines
    pairs = get_pairwise_fst(popidx, mygts)

    # save
    print(ColorText('\nSaving pairwise FST data ...').bold().custom('gold'))
    for loci_set, df in pairs.items():
        f = op.join(fst_dir, f'{seed}_{loci_set}_pairwise_FST.txt')
        df.to_csv(f, index=True, sep='\t')

        print(ColorText(f'\t{loci_set} loci').bold())
        print('\t', f, '\n')

    # done
    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')
        
    pass


if __name__ == "__main__":
    # get input arguments
    thisfile, seed, slimdir, gf_training_dir, num_engines = sys.argv
    
    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()
    
    # set up dirs
    fst_dir = makedir(op.join(op.dirname(op.dirname(op.dirname(gf_training_dir))),
                              'fst'))

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=int(num_engines), profile=f'fst_{seed}')

    main()
