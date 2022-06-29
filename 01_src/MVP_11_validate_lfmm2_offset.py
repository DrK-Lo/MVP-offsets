"""Validate offset predictions from lfmm using population mean fitness.

Usage
-----
conda activate mvp_env
python MVP_11_validate_lfmm2_offset.py seed slimdir outerdir

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scripts
outerdir
    the directory into which all of the lfmm2 subdirectories were created in MVP_10_train_lfmm2_offset.py

Dependencies
------------
- dependent upon completion of MVP_10_train_lfmm2_offset.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from myfigs import save_pdf

import MVP_03_validate_gradient_forests as mvp03
import MVP_06_validate_RONA as mvp06
import MVP_10_train_lfmm2_offset as mvp10

from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import linregress


def check_file_counts():
    """Since I used GNU parallel (where jobs can die silently) make sure all of the output files exist."""
    print(ColorText('\nChecking file counts ...').bold().custom('gold'))
    # get the directories where the commands and output files are located
    indir, outdir, shdir = mvp10.make_lfmm_dirs(outerdir)
    
    # get a list of batch files with commands (one per line) and the outfiles produced
    batch_files = fs(indir, pattern=f'{seed}_lfmm_batch')
    outfiles = fs(outdir, pattern=f'{seed}_lfmm2_offsets')
    
    # get a list of commands from the batch files
    cmds = flatten([read(f, lines=True) for f in batch_files])
    
    # there should be just as many outfiles as there are commands
    try:
        assert len(outfiles) == len(cmds)
    except AssertionError as e:
        print(ColorText(f'\nError: there are missing outfiles for seed {seed}').fail())
        raise e

    return outfiles


def read_lfmm_offsets(outfiles):
    print(ColorText('\nReading in lfmm offset predictions ...').bold().custom('gold'))
    
    offset_dfs = defaultdict(dict)
    for outfile in outfiles:
        df = pd.read_table(outfile)
        *args, marker_set, garden = op.basename(outfile).split("_")
        offset_dfs[marker_set][int(garden)] = df  # one column, index = source population subpopID

    offsets = {}
    for marker_set in keys(offset_dfs):
        offset_cols = []
        for garden in range(1, 101, 1):
            offset_col = offset_dfs[marker_set][garden].copy()
            offset_col.columns = [garden]
            offset_cols.append(offset_col)

        # transpose so that source deme is column and transplant location (garden) is row
        offsets[marker_set] = pd.concat(offset_cols, axis=1).T
        
    return offset_dfs, offsets


def create_heatmap(corrs, marker_set, locations, title=None, performance='garden', save=False):
    # fill in heatmap
    heatmap = mvp03.blank_dataframe()
    for garden,corr in corrs.items():
        x, y = locations.loc[garden]
        heatmap.loc[y, x] = corr
        
    # plot heatmap
    sns.heatmap(heatmap,
                cmap='viridis',
                cbar_kws={'label': "Spearman's $\\rho$"})
    plt.title(title)
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')

    if save is True:
        figfile = op.join(fig_dir, f'{seed}_{marker_set}_lfmm_{performance}_performance_heatmap.pdf')
        save_pdf(figfile)
        
        heatfile = figfile.replace(fig_dir, heat_dir).replace('.pdf', '.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True, header=True)
    
    plt.show()
    plt.close()
    
    pass


def garden_performance_slope_heatmap(offset, fitness, adaptive_or_all, locations, fig_dir=None):
    """Create a heatmap for each common garden that displays slope of regression: fitness ~ offset."""
    # get slopes and fill in the heatmap
    heatmap = mvp03.blank_dataframe()
    for garden in fitness.index:
        x,y = locations.loc[garden]
        heatmap.loc[y, x] = linregress(offset.loc[garden], fitness.loc[garden]).slope

    # plot the heatmap
    _ = sns.heatmap(heatmap,
                    cmap='viridis',
                    cbar_kws={'label': "slope of fitness ~ lfmm offset"})
    plt.title(f'slope in garden for {adaptive_or_all} loci')
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')

#     print('\tgarden_performance_slope_heatmap()')
    if fig_dir is not None:
        figfile = op.join(fig_dir, f'{seed}_{adaptive_or_all}_lfmm_garden_slope_heatmap.pdf')
        save_pdf(figfile)
        
        heatfile = figfile.replace(fig_dir, heat_dir).replace('.pdf', '.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True, header=True)

    plt.show()
    plt.close()

    pass


def source_performance_slope_heatmap(offset, fitness, adaptive_or_all, locations, fig_dir=None):
    # 1. calculate slopes
    slopes = {}
    for source_pop in fitness.index:
        slopes[source_pop] = linregress(offset[source_pop], fitness[source_pop]).slope

    # 2. fill heatmap
    heatmap = mvp03.blank_dataframe()
    for source_pop, slope in slopes.items():
        x,y = locations.loc[source_pop]
        heatmap.loc[y, x] = slope

    # 3. show heatmap
    sns.heatmap(heatmap,
                cmap='viridis',
                cbar_kws={'label': "slope of fitness ~ offset"})

    plt.title(f'slope per source pop for{adaptive_or_all} loci')
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')

    # 4. save heatmap figure and datatable
    if fig_dir is not None:
        figfile = op.join(fig_dir, f'{seed}_source_slope_heatmap-{ind_or_pooled}_{adaptive_or_all}.pdf')
        save_pdf(figfile)
        
        heatfile = figfile.replace(fig_dir, heat_dir).replace('.pdf', '.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True, header=True)
        
    plt.show()
    plt.close()
    
    pass


def fig_wrapper(subset, locations, envdata, offsets, fitness_mat, fig_dir=None):
    print(ColorText('\nCreating figures ...').bold().custom('gold'))
    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))

    garden_performance = {}
    source_performance = {}
    for marker_set, offset in offsets.items():
        # 2 - GARDEN PERFORMANCE - how well offset was predicted at the common garden location across samples
        # 2.1 spearman's correlation coefficient (val) for each garden (key)
        garden_performance[marker_set] = offset.corrwith(fitness_mat,
                                                         axis='columns',
                                                         method='spearman')

        # 2.2 plot histogram
        garden_performance[marker_set].hist()
        title = f'{seed}\ngarden performance\nloci={marker_set}'
        plt.title(title)
        plt.ylabel('count')
        plt.xlabel("Spearman's $\\rho$")
        if fig_dir is not None:
            save_pdf(
                op.join(fig_dir, f'{seed}_{marker_set}_lfmm_garden_performance_histogram.pdf')
            )
        plt.show()
        plt.close()

        # 2.3 create heatmap
        create_heatmap(garden_performance[marker_set],
                       marker_set,
                       locations,
                       title=f'{seed}\n{marker_set} loci\ngarden performance',
                       performance='garden',
                       save=True)

        # 2.4 calculate and plot slope of relationship between fitness ~ offset at each garden
        # color for the environment (temp_opt, sal_opt)
        for env, env_series in envdata.items():
            colormap = 'Reds' if env=='temp_opt' else 'Blues_r'
            cmap = plt.cm.get_cmap(colormap)

            colors = offset.columns.map(env_series).to_series().apply(mvp06.color, cmap=cmap, norm=norm)
            mvp06.garden_performance_scatter(offset, fitness_mat, f'{marker_set}_{env}',
                                             locations, env_series, colors, norm=norm, cmap=cmap, seed=seed,
                                             fig_dir=fig_dir, program=f'lfmm')
            plt.close()  # needed so that garden_performance_slope_heatmap doesn't create fig on top of this

        # 2.5 calculate the slope of the linear model between fitness ~ offset at each garden
        garden_performance_slope_heatmap(offset, fitness_mat, marker_set, locations, fig_dir=fig_dir)


        # 3 - SOURCE PERFORMACE - how well performace was predicted for the source population across gardens
        # 3.1 spearman's correlation coefficient (val) for each garden (key)
        source_performance[marker_set] = offset.corrwith(fitness_mat,
                                                         axis='index',
                                                         method='spearman')

        # 3.2 plot histogram
        source_performance[marker_set].hist()
        title = f'{seed}\nsource pool performance\nloci={marker_set}'
        plt.title(title)
        if fig_dir is not None:
            save_pdf(
                op.join(fig_dir, f'{seed}_{marker_set}_lfmm_source_performance_histogram.pdf')
            )
        plt.show()
        plt.close()

        # 3.3 create heatmap
        create_heatmap(source_performance[marker_set],
                       marker_set,
                       locations,
                       performance='source',
                       title=f'{seed}\n{marker_set} loci\nsource performance',
                       save=False)

        # 3.4 calculate and plot slope of relationship between fitness ~ offset for each ind/pool across gardens
        source_performance_slope_heatmap(offset, fitness_mat, marker_set, locations, fig_dir=None)

    pass


def main():
    # make sure all of the expected outfiles were produced
    outfiles = check_file_counts()

    # get predicted offset values
    offset_dfs, offsets = read_lfmm_offsets(outfiles)
    
    # get the population mean fitness for source populations (columns) in transplant gardens (rows)
    print(ColorText('\nGetting population data ...').bold().custom('gold'))
    fitness_mat = mvp06.load_pooled_fitness_matrix(slimdir, seed)
    
    # get the subset of simulated individuals and population data
    subset, locations, envdata = mvp06.get_pop_data(slimdir, seed)

    fig_wrapper(subset, locations, envdata, offsets, fitness_mat, fig_dir=fig_dir)
    
    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, outerdir = sys.argv
    
    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))
    
    # set up timer
    t1 = dt.now()
    
    # set globally
    norm = Normalize(vmin=-1.0, vmax=1.0)
    fig_dir = makedir(op.join(outerdir, 'lfmm2/validation/figs'))
    heat_dir = makedir(op.join(outerdir, 'lfmm2/validation/heatmap_textfiles'))
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()
