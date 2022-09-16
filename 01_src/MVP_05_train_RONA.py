"""Calculate the Risk Of Non-Adaptedness for each subpopulation transplanted to all others.

Usage
-----
conda activate mvp_env
python MVP_05_train_RONA.py seed rona_training_dir num_engines

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
rona_training_dir
    path to directory created from `outdir` arg to MVP_01_train_gradient_forests.py: outdir/RONA/training/training_files
num_engines
    number of engines to use to parallelize calculations necessary to estimate RONA

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from scipy.stats import linregress
from MVP_01_train_gradient_forests import read_ind_data


def get_envdata(subset):
    """Get environmental data for each pop for each environment."""
    print(ColorText('\nGetting environmental data for each subpopID ...').bold().custom('gold'))
    # map samps to pop, and get a list of all samps per pop
    samppop = dict(zip(subset.index, subset.subpopID))  # key = individual_ID val = subpopID

    # assign subpopID to each individual's environmental data (same for each individual within a pop)
    rangedata = subset[['y', 'x', 'sal_opt', 'temp_opt']].copy()
    rangedata['popID'] = rangedata.index.map(samppop)

    # map population ID to environmental data
    envdict = dict()
    for env in ['sal_opt', 'temp_opt']:
        envdict[env] = rangedata.groupby('popID')[env].apply(np.mean).to_dict()

    # convert dict to dataframe - rownames = popID (note no zero index)
    popenvdata = pd.DataFrame(envdict)

    return popenvdata


def train(freqs):
    """Calculate linear regression slope, intercept, & p-val for each locus for each environmental variable."""
    from scipy.stats import linregress
    from tqdm import tqdm as pbar
    from collections import defaultdict
    from pythonimports import wrap_defaultdict
    
    freqs.columns = freqs.columns.astype(int)  # convert from string to int to avoid pop name lookup errors
    
    # estimate linear models
    results = wrap_defaultdict(tuple, 2)
    for env in popenvdata.columns:
        for locus in freqs.index:
            slope,intercept,rval,pval,stderr = linregress(popenvdata[env],
                                                          freqs[popenvdata.index].loc[locus])
            results[env][locus] = slope,intercept,pval
    
    return results


def calculate_linear_models(seed):
    print(ColorText('\nCalculating linear models for each locus for each env ...').bold().custom('gold'))
    # get population-level allele frequencies for the derived allele
    freqfile = op.join(rona_training_dir, f'{seed}_Rout_Gmat_sample_maf-gt-p01_RONAready_pooled_all.txt')

    # calculate linear models in parallel
    jobs = parallel_read(freqfile,
                         lview=lview,
                         dview=dview,
                         functions=create_fundict(train),
                         verbose=False,
                         maintain_dataframe=False,
                         index_col=0)

    # gather results
    results = defaultdict(dict)
    for envdict in jobs:
        for env,locusdict in envdict.items():
            results[env].update(locusdict)

    # save
    seed = op.basename(freqfile).split("_")[0]
    respkl = op.join(rona_outdir, f'{seed}_linear_model_results.pkl')
    pkldump(results, respkl)

    print(f'\tsaved results to: {respkl}')

    return results, freqfile


def retrieve_significant_models(results):
    """Determine which of the loci had linear models with pvals <= 0.05."""
    print(
        ColorText(
            '\nDetermining which loci had significant linear models ...'
        ).bold().custom('gold')
    )
    sig_models = defaultdict(list)
    for env,locusdict in results.items():
        for locus,(slope,intercept,pval) in pbar(locusdict.items(), desc=env):
            if pval <= 0.05:
                sig_models[env].append(locus)

    time.sleep(1)

    for env,loci in sig_models.items():
        print(env,
            'had',
            len(loci),
            'loci with significant linear models out of',
            len(results[env].keys()),
            'loci')

    return dict(sig_models)


def calc_rona_elements(garden, env, pop, aaf_present):
    """Calculate and return the abs element for each of n loci in `aaf_present.index`."""
    ef_fut = popenvdata[env].loc[garden]
    
    rona_elements = {}
    for locus_i in aaf_present.index:
        s_present, i_present, _pval = results[env][locus_i]
        # nans can be appended if locus has missing data for that pop, account for this when summing/avging
        rona_elements[locus_i] = abs(
            (s_present * ef_fut) + i_present - aaf_present[pop].loc[locus_i]
        )
    
    return rona_elements


def get_rona_elements(freqs):
    """Calculate the typical summation element of RONA for loci with significant linear models."""
    from pythonimports import wrap_defaultdict
    from tqdm import tqdm as pbar
    
    freqs.columns = freqs.columns.astype(int)  # convert from string to int to avoid pop name lookup errors
    
    elements = wrap_defaultdict(None, 3)
    for env,loci in sig_models.items():
        interloci = set(loci).intersection(freqs.index)  # don't want whole table - need because freqs is a chunk
        for garden in pbar(popenvdata.index, desc=env):
            for pop in freqs.columns:
                elements[env][garden][pop] = calc_rona_elements(garden,
                                                                env,
                                                                pop,
                                                                freqs.loc[interloci])
    return elements


def scatter_rona_elements(freqfile):
    print(
        ColorText(
            '\nCalculating locus elements of RONA for loci with significant linear models ...'
        ).bold().custom('gold')
    )
    # calculate rona elements in parallel
    jobs = parallel_read(freqfile,
                         lview=lview,
                         dview=dview,
                         verbose=False,
                         index_col=0,
                         functions=create_fundict(get_rona_elements),
                         maintain_dataframe=False)

    # gather elements
    rona_elements = wrap_defaultdict(dict, 3)
    for j in pbar(jobs, desc='extracting element data'):
        for env,gardendict in j.items():
            for garden,popdict in gardendict.items():
                for pop,elementdict in popdict.items():
                    # `elementdict` is returned from `calc_rona_elements` - key=locus, val=summation_element
                    rona_elements[env][garden][pop].update(elementdict)

    return rona_elements


def calculate_rona(rona_elements, sig_models, marker_set='all'):
    """Calculate the real rona by summing and averaging elements."""
    print(f'using {marker_set} loci ...')
    
    rona = wrap_defaultdict(None, 3)  # one RONA per pop per climate
    rona_loci_counts = wrap_defaultdict(None, 3)
    all_counts = wrap_defaultdict(list, 2)
    
    if marker_set != 'all':
        loci_pkl = op.join(rona_training_dir, f'{seed}_{marker_set}_loci.pkl')  # from MVP_01.py
        subset_loci = pklload(loci_pkl)

    for env, gardendict in rona_elements.items():
        for garden, popdict in pbar(gardendict.items(), desc=env):
            for pop, elementdict in popdict.items():
                # make sure I got what I was expecting to get - number of loci with sig linear models
                assert len(elementdict) == len(sig_models[env])

                # for each group of loci, calc RONA by according to equation in Rellstab et al. (2016) by 
                    # averaging rona_elements while accounting for missing data
                
                # get all of the typical summation elements for loci within `sig_models[env]` that have sig models
                    # these should all be in `elementdict.keys()`
                interloci = set(sig_models[env]).intersection(elementdict.keys())
                assert len(interloci) == len(sig_models[env]) == len(elementdict.keys())
                
                if marker_set != 'all':
                    interloci = set(interloci).intersection(subset_loci)

                elements = [elementdict[locus] for locus in interloci]
                if sum(el==el for el in elements) > 0:
                    # if at least one instance of non-np.nan data:
                    _mean = np.nanmean(elements)  # np.nanmean accounts for missing data
                else:
                    # avoid RuntimeWarning: Mean of empty slice
                    # this happens when a pop has missing data at all of the few number of loci given (eg 2)
                        # and each `el` in `elements` was therefore `np.nan`
                        # this would only happen with empirical data ++ when the loci set is very small
                    _mean = np.nan
                    
                rona[env][garden][pop] = _mean
                rona_loci_counts[env][garden][pop] = len(interloci)
                all_counts[garden][env].append(interloci)

    # save RONA results
    rona_file = op.join(rona_outdir, f'{seed}_{marker_set}_RONA_results.pkl')
    pkldump(rona, rona_file)

    print('\tsaved RONA results to: ', rona_file, '\n')
    
    pass


def main():
    # get data for the individuals that were subsampled from full simulation
    subset = read_ind_data(slimdir, seed)

    # get environmental values for each subpopID, load to parallel engines
    dview['popenvdata'] = get_envdata(subset)  # load popenvdata to engines
    sleeping(10)

    # calculate linear model fits between pop allele freq and climate env
    results, freqfile = calculate_linear_models(seed)
    dview['results'] = results  # load results to engines
    sleeping(10)

    # determine which loci had significant linear models with environments
    sig_models = retrieve_significant_models(results)
    dview['sig_models'] = sig_models  # load sig_models to engines
    time.sleep(10)  # make sure sig_models loads completely to all engines

    # calculate the locus-level element of the RONA equation
    dview['calc_rona_elements'] = calc_rona_elements  # load function to engines
    rona_elements = scatter_rona_elements(freqfile)

    # calculate RONA according to equation from Rellstab et al. 2016
    print(ColorText('\nCalculating RONA for each common garden ...').bold().custom('gold'))
    for marker_set in ['all', 'adaptive', 'neutral']:
        calculate_rona(rona_elements, sig_models, marker_set)

    # done
    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input arguments
    thisfile, seed, slimdir, rona_training_dir, num_engines = sys.argv
    
    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    # set up timer
    t1 = dt.now()

    # make dirs
    rona_outdir = makedir(
        op.join(op.dirname(rona_training_dir), 'training_outfiles')
    )

    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=int(num_engines))

    main()
