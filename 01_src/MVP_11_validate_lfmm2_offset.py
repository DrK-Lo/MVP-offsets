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
from myfigs import histo_box, gradient_image, create_cmap

import MVP_03_validate_gradient_forests as mvp03
import MVP_06_validate_RONA as mvp06
import MVP_10_train_lfmm2_offset as mvp10
import MVP_13_RDA_validation as mvp13

from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import linregress


def check_file_counts():
    """Since I used GNU parallel (where jobs can die silently) make sure all of the output files exist."""
    print(ColorText('\nChecking file counts ...').bold().custom('gold'))
    # get the directories where the commands and output files are located
    mvp10.seed = seed
    indir, outdir, shdir = mvp10.make_lfmm_dirs(outerdir)
    
    # get a list of batch files with commands (one per line) and the outfiles produced
    batch_files = fs(indir, pattern=f'{seed}_lfmm_batch')
    outfiles = fs(outdir, startswith=f'{seed}_lfmm_offsets_ntraits', endswith='.txt')
    
    # get a list of commands from the batch files
    cmds = flatten([read(f, lines=True) for f in batch_files])
    
    # there should be just as many outfiles as there are commands
    try:
        assert len(outfiles) == len(cmds)
    except AssertionError as e:
        print(ColorText(f'\nError: there are missing outfiles for seed {seed}').fail())
        print(ColorText(f'\t{len(cmds) = }').fail())
        print(ColorText(f'\t{outdir = }').fail())
        print(ColorText(f'\t{len(outfiles) = }').fail())
        raise e

    return outfiles


def read_lfmm_offset_dfs(outfiles):
    """Read in .txt files containing population offset to a particular common garden."""
    print(ColorText('\nReading in lfmm offset predictions ...').bold().custom('gold'))
    
    offset_series = wrap_defaultdict(dict, 2)
    for outfile in outfiles:
        df = pd.read_table(outfile)  # single column data.frame, index = source population subpopID
        *args, ntraits, marker_set, garden = op.basename(outfile).split("_")
        offset_series[marker_set][ntraits][int(garden.replace('.txt', ''))] = df

    offset_dfs = defaultdict(dict)
    for marker_set in keys(offset_series):
        for ntraits in keys(offset_series[marker_set]):
            offset_cols = []
            for garden in range(1, 101, 1):
                try:
                    offset_col = offset_series[marker_set][ntraits][garden].copy()
                except KeyError as e:
                    print(marker_set, ntraits, garden)
                    raise e
                offset_col.columns = [garden]
                offset_cols.append(offset_col)

            # transpose so that source deme is column and transplant location (garden) is row
            df = pd.concat(offset_cols, axis=1).T
            df.columns = df.columns.astype(str)
            df.index = df.index.astype(int)
            offset_dfs[marker_set][ntraits] = df
            
    # save offsets
    pkl = op.join(offset_dir, f'{seed}_offset_dfs.pkl')
    pkldump(offset_dfs, pkl)
    print(f'\n\twrote offset_dfs to : {pkl}')
        
    return offset_dfs


def calculate_performance(fitness, offset_dfs, marker_sets=['all', 'adaptive', 'neutral']):
    """Calculate correlation between fitness and offset within and across gardens."""
    print(ColorText("\nCalculating performance ...").bold().custom('gold'))
    
    garden_performance = defaultdict(dict)
    source_performance = defaultdict(dict)
    garden_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)
    source_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)

    for marker_set in marker_sets:
        for ntraits, offset in offset_dfs[marker_set].items():

            # correlation of fitness and offset within gardens across transplants
            garden_performance[marker_set][ntraits] = offset.corrwith(
                fitness,
                axis='columns',  # across columns for each row
                method='kendall')      

            # slope of fitness ~ offset within gardens
            for garden in pbar(fitness.index, desc=f'{marker_set} {ntraits} garden performance'):
                # record slope
                garden_slopes[marker_set][ntraits].loc[garden] = linregress(
                    offset.loc[garden, fitness.columns],
                    fitness.loc[garden]
                ).slope

            # correlation of fitness and offset for transplants across gardens
            source_performance[marker_set][ntraits] = offset.corrwith(
                fitness,
                axis='index',  # across rows for each column
                method='kendall')

            # slope of fitness ~ offset across gardens for individual pops
            for source_pop in pbar(fitness.columns, desc=f'{marker_set} {ntraits} population performance'):
                # record slope
                source_slopes[marker_set][ntraits].loc[source_pop] = linregress(
                    offset[source_pop].loc[fitness.index],
                    fitness[source_pop]
                ).slope

    performance_dicts = {'garden_performance' : garden_performance,
                         'source_performance' : source_performance,
                         'garden_slopes' : garden_slopes,
                         'source_slopes' : source_slopes}
    
    pkl = op.join(corr_dir, f'{seed}_performance_dicts.pkl')
    pkldump(performance_dicts, pkl)
    print(f'\twrote performance_dicts to : {pkl}')

    return performance_dicts


def create_histo_subplots(performance_dict, performance_name, pdf, cmap='viridis', N=5,
                          marker_sets=['all', 'adaptive', 'neutral'],
                          histplot_kws={}, boxplot_kws={}):
    """Create a panel of histograms for performance of either garden or source (within `performance_dict`).
    
    Parameters
    ----------
    performance_dict - nested dictionary
        - contstructed as - performance_dict[ind_or_pooled][marker_set][which_traits][structcorr]
        - the final value is a pandas.Series of Kendall's tau validation scores (cor between offset
            and prediction)
    performance_name - str
        - the type of performance measured, used for figure label and file; either 'garden performance'
            or 'source performance'
    cmap
        - the type of color map to signify the sign and magnitude of Kendall's tau
    N - int
        - the number of breaks in the color map
    marker_sets - list
        - a list of marker sets - see `label_dict` set to mvp13 namespace
    histplot_kws - dict
        - kwargs passed to histo_box
    boxplot_kws - dict
        - kwargs passed to histo_box
        
    
    Notes
    -----
        - creates one figure for pooled data and another figure for individual data
        - each figure contains rows (one for each marker set in `marker_sets`) and two columns
            each for 1-env RDA and 2-env RDA (where the two columns are RDAs with(out) population
            structure correction)
    """
    print(ColorText(f'\nCreating histo boxplots for {performance_name}...').bold().custom('gold'))
    
    # get color map
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap, lut=N)
        
    # update kwargs
    if 'edgecolor' not in keys(histplot_kws):
        histplot_kws.update(
            dict(edgecolor='white')
        )
    if 'linewidth' not in keys(boxplot_kws):
        boxplot_kws.update(
            dict(linewidth=0.80)
        )

    # differences between ntraits=1 and ntraits=2
    total_traits = ['ntraits-1', 'ntraits-2'] if ntraits == 1 else ['ntraits-2']

    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), len(total_traits),
                             sharey='all',
                             figsize=(10, 10) if ntraits == 1 else (5, 10))

    # fill in subplots with histograms
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]

        for col, which_traits in enumerate(total_traits):
            ax = row_axes[col] if ntraits == 1 else row_axes

            # create the histo_boxplot
            ax_box, ax_hist = histo_box(
                performance_dict[marker_set][which_traits].copy(),
                ax=ax, histbins=15, histplot_kws=histplot_kws, boxplot_kws=boxplot_kws
            )

            # add in some labels
            if marker_set == marker_sets[-1]:  # if the last row
                ax_hist.set_xlabel("Kendall's $\\tau$")

            if col == 0:  # if the first column
                ax.set_ylabel('count')

            # color in bars of histograph to match their score on the colormap
            mvp13.color_histogram(ax, cmap=cmap, norm=norm)

            # create a background color for each histogram
            gradient_image(ax, direction=0.5, transform=ax.transAxes, 
                           cmap=background_cmap, cmap_range=(0.0, 0.2))

        
    # add labels, title, colorbar, etc
#     mvp13.add_title_and_colorbar(fig, axes,
#                                  title=f'{performance_name}\n{seed = }\n{level}\n',
#                                  cmap=cmap,
#                                  program='lfmm2')
#     fig.tight_layout()
    mvp13.decorate_figure(marker_sets, fig, axes, xadjust=0.18,
                          title=f'{performance_name}\n{seed = }\n{level}\n', program='lfmm2')

    # save
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
        
    pass


def create_heatmap_subplots(performance_dict, performance_name, pdf, locations,
                            cmap='viridis', use_vmin_vmax=True, marker_sets=['all', 'adaptive', 'neutral']):
    """For garden or source performance, create a heatmap of the simulated landscape showing Kendall's tau."""
    print(ColorText(f'\nCreating heatmap subplots for {performance_name} ...').bold().custom('gold'))
    
    # differences between ntraits=1 and ntraits=2
    total_traits = ['ntraits-1', 'ntraits-2'] if ntraits == 1 else ['ntraits-2']
    
    if use_vmin_vmax is True:
        vmin, vmax = mvp13.get_vmin_vmax(performance_dict)
        print(f'{vmin = } {vmax = }')
    else:
        vmin = -1
        vmax = 1
    
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), len(total_traits),
                             sharey='all',
                             figsize=(10, 13) if ntraits == 1 else (5, 13))

    # fill in subplots with histograms
    heatmaps = defaultdict(dict)
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]

        for col, which_traits in enumerate(total_traits):

            if ntraits == ntraits == 1:
                ax = row_axes[col]
            else:
                ax = row_axes

            try:
                corrs = performance_dict[marker_set][which_traits].copy()
            except KeyError as e:
                print(marker_set, which_traits)
                raise e

            # fill in heatmap
            df = mvp03.blank_dataframe()
            for garden, corr in corrs.items():
                x, y = locations.loc[int(garden)]
                df.loc[y, x] = corr
            heatmaps[marker_set][which_traits] = df.copy()

            # plot heatmap
            _ = sns.heatmap(df, cmap='viridis', cbar=False, ax=ax, vmin=vmin, vmax=vmax)

            # add in some labels
            if marker_set == marker_sets[-1]:  # if the last row
                ax.set_xlabel("longitude")

            if col == 0:  # if the first column
                ax.set_ylabel('latitude')

        
    # add labels, title, colorbar, etc
    title = f'{performance_name}\n{seed = }\n{level}\n'
    mvp13.decorate_figure(marker_sets, fig, axes, title=title, xadjust=0.18,
                          vmin=vmin, vmax=vmax, program='lfmm2')

    # save figure
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()

    # save objects
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\n\twrote heatmaps to : {pkl}')

    pass


def scatter_wrapper(offset_dfs, fitness, envdata, locations, pdf, marker_sets=['all', 'adaptive', 'neutral'], total_traits=None):
    """Wrapper for `mvp06.performance_scatter`."""
    print(ColorText('\nCreating scatter plots ...').bold().custom('gold'))
    for marker_set in marker_sets:
        for home_env in pbar(['sal_opt', 'temp_opt'], desc=marker_set):  # the environment for used to color populations
            for which_traits in total_traits:  # the number of environments used to calculate lfmm
                offset = offset_dfs[marker_set][which_traits]  # already transposed properly
                
                # color for the environment (temp_opt) of source_pop
                colormap = 'Reds' if home_env=='temp_opt' else 'Blues_r'
                cmap = plt.cm.get_cmap(colormap)
                
                colors = fitness.index.map(envdata[home_env]).to_series(index=fitness.index).apply(mvp06.color,
                                                                                                   cmap=cmap,
                                                                                                   norm=norm).to_dict()
                
                for garden_or_source in ['garden', 'source']:
                    # plot performance within gardens across source populations
                    mvp06.performance_scatter(offset.copy(),
                                              fitness.copy(),
                                              f'{label_dict[marker_set]} {which_traits}',
                                              locations,
                                              colors,
                                              pdf,
                                              norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir, home_env=home_env,
                                              program='lfmm2',
                                              garden_or_source=garden_or_source)

    pass


def fig_wrapper(performance_dicts, offset_dfs, fitness, envdata, locations):
    """Create figs."""
    # how many envs were selective?
    total_traits = ['ntraits-1', 'ntraits-2'] if ntraits == 1 else ['ntraits-2']
    
    # figures about performance
    saveloc = op.join(fig_dir, f'{seed}_lfmm_figures.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        
        for performance_name in ['garden_performance', 'source_performance']:
            performance_dict = performance_dicts[performance_name].copy()
            
            create_histo_subplots(performance_dict, performance_name, pdf, boxplot_kws=dict(
                flierprops={
                    'marker': '.',
                    'markerfacecolor': 'gray',
                    'alpha': 0.5,
                    'markeredgewidth' : 0.0  # remove edge
                },
                color='lightsteelblue'
            ))
            
            # garden and source performance heatmaps
            create_heatmap_subplots(performance_dict, performance_name, pdf, locations)

            # garden performance slope of fitness ~ offset
            slope_group = performance_name.split("_")[0]
            mvp06.create_slope_heatmap_subplots(
                performance_name, performance_dicts[f'{slope_group}_slopes'].copy(), locations, pdf,
                total_traits=total_traits, program='lfmm2'
            )

    print(ColorText(f'\nsaved fig to: {saveloc}').bold())
    
    # save scatterplots separately so computers don't get slow trying to display everything
    saveloc = op.join(fig_dir, f'{seed}_lfmm_figures_scatter.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        scatter_wrapper(offset_dfs, fitness, envdata, locations, pdf, total_traits=total_traits)
        
    print(ColorText(f'\nsaved fig to: {saveloc}').bold())
    
    
    pass


def main():
    # make sure all of the expected outfiles were produced
    outfiles = check_file_counts()

    # get predicted offset values
    offset_dfs = read_lfmm_offset_dfs(outfiles)
    
    # get the population mean fitness for source populations (columns) in transplant gardens (rows)
    print(ColorText('\nGetting population data ...').bold().custom('gold'))
    fitness = mvp06.load_pooled_fitness_matrix(slimdir, seed)
    
    # get the subset of simulated individuals and population data
    subset, locations, envdata = mvp06.get_pop_data(slimdir, seed)
    
    # calculate correlation between fitness and offset within and across gardens
    performance_dicts = calculate_performance(fitness, offset_dfs)

    # create figs
    fig_wrapper(performance_dicts, offset_dfs, fitness, envdata, locations)
    
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
    
    # details about demography and selection
    ntraits, level = mvp10.read_params_file(slimdir).loc[seed, ['N_traits', 'level']]
    
    # set globally
    norm = Normalize(vmin=-1.0, vmax=1.0)    
    fig_dir = makedir(op.join(outerdir, 'lfmm2/validation/figs'))
    heat_dir = makedir(op.join(outerdir, 'lfmm2/validation/heatmap_textfiles'))
    corr_dir = makedir(op.join(outerdir, 'lfmm2/validation/corrs'))
    offset_dir = makedir(op.join(outerdir, 'lfmm2/validation/offset_dfs'))
    
    # background color for figures
    background_cmap = create_cmap(['white', 'gold'], grain=1000)
    
    # dict for pretty labels in figures
    label_dict = {
        'all' : 'all loci',
        'adaptive' : 'causal loci',
        'neutral' : 'neutral loci',
        'ntraits-1' : '1-env lfmm',
        'ntraits-2' : '2-env lfmm',
        'sal_opt' : 'sal',
        'temp_opt' : 'temp'
    }
    
    # pass objects to imported namespaces
    mvp03.label_dict = label_dict
    mvp06.label_dict = label_dict
    mvp13.label_dict = label_dict
    mvp06.seed = seed
    mvp06.norm = norm
    mvp06.level = level
    mvp06.heat_dir = heat_dir
    mvp13.ntraits = ntraits
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()
