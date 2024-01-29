"""Validate trained models of Gradient Forests.

URGENT NOTES
------------
- this script is modified to expect only pooled samples
    - i've updated assertion of len(files) == len(rdsfiles)
    - I also exclude _ind_ in the fs function within get_offset_predictions (two fs functions!)
    - if removed, MVP_01 and MVP_02 need to be updated as well
    - I also added input arg `expected` so I can do cross val etc without breaking script
- I updated the script to expect only pooled samples by default, but by ...
    adding on to trailing `expected` arg I can expect individual samples only

Usage
-----
conda activate mvp_env
python MVP_03_validate_gradient_forests.py seed slimdir gf_parentdir

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scripts
gf_parentdir
    the gradient forests upper directory within the directory created from `outdir` from MVP_01_train_gradient_forests.py

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon completion of MVP_02_fit_gradient_forests.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from myfigs import histo_box, gradient_image, create_cmap

import MVP_06_validate_RONA as mvp06
import MVP_10_train_lfmm2_offset as mvp10
import MVP_13_RDA_validation as mvp13

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import linregress


def load_ind_fitness_matrix(slimdir, seed, subset):
    """Load individual fitness matrix.
    
    Notes
    -----
    matrix is an n_deme (rows) x n_individuals(columns) matrix 
        with entries for fitness of individual in deme
    """
    # read in fitness data
    fitness = pd.read_table(op.join(slimdir, f'{seed}_fitnessmat_ind.txt'),
                            delim_whitespace=True,
                            header=None)
    # remove the first row which maps individuals (columns) to subpopID (cell entries)
    fitness = fitness.drop([0])
    # name individuals
    fitness.columns = fitness.columns.astype(str).tolist()
    # reduce to only the subsampled individuals
    fitness = fitness[subset.index]
    
    return fitness


def get_offset_predictions(seed, expected=300, exclude='_ind_'):
    """Get offset predictions output by MVP_02_fit_gradient_forests.py."""
    print(ColorText('\nRetrieving predicted offsets ...').bold().custom('gold'))
    # get the predicted offset from files output from fitting created in ../02_fit_gradient_forests.ipynb
    files = fs(fitting_dir, endswith='offset.txt', startswith=f'{seed}', exclude=exclude)  # CHANGE!

    # make sure just as many RDS files were created from fitting script (ie that all fitting finished)
    rdsfiles = fs(fitting_dir, endswith='.RDS', startswith=f'{seed}', exclude=exclude)
#     assert len(files) == len(rdsfiles) == 600, (len(files), len(rdsfiles))  # 100 gardens * 3 marker sets * ind_or_pooled
    assert len(files) == len(rdsfiles) == expected, (len(files), len(rdsfiles))  # 100 gardens * 3 marker sets

    outfiles = wrap_defaultdict(dict, 3)
    for outfile in files:
        seed, ind_or_pooled, marker_set, garden_ID, *suffix = op.basename(outfile).split("_")
        outfiles[ind_or_pooled][marker_set][int(garden_ID)] = outfile

    # gather the predicted offset values for each individual in each garden
    offset_series = wrap_defaultdict(list, 2)  # for gathering in next loop
    for (ind_or_pooled, marker_set, garden_ID), outfile in unwrap_dictionary(outfiles):
        # read in offset projections
        offset = pd.read_table(outfile, index_col=0)
        offset_series[ind_or_pooled][marker_set].append(
            pd.Series(offset['offset'], name=garden_ID)
        )

    # collapse the predicted offset values for each individual in each garden into one data frame
        # - use for correlation calcs in next cell
    offset_dfs = wrap_defaultdict(None, 2)
    for (ind_or_pooled, marker_set), series_list in unwrap_dictionary(offset_series):
        # collapse all of the offset values from each garden into a single dataframe
        df = pd.concat(series_list,
                       axis=1,
                       keys=[series.name for series in series_list])
        df.index = df.index.astype(str)
        # sort by garden_ID, transpose to conform to convention of `fitness_mat`
        offset_dfs[ind_or_pooled][marker_set] = df[sorted(df.columns)].T

    # save offsets
    pkl = op.join(offset_dir, f'{seed}_offset_dfs.pkl')
    pkldump(offset_dfs, pkl)
    print(f'\n\twrote offset_dfs to : {pkl}')
        
    return offset_dfs


def blank_dataframe():
    """Create a blank dataframe (landscape map) filled with NaN, columns and index are subpopIDs.
    
    Notes
    -----
    instantiating with dtype=float is necessary for sns.heatmap (otherwise sns.heatmap(df.astype(float)))
    """
    df = pd.DataFrame(columns=range(1, 11, 1),
                      index=reversed(range(1,11,1)),  # so that x=1,y=10 is in top left
                      dtype=float)
    return df


def calculate_performance(offset_dfs, fitness_mat, popsamps, samppop):
    print(ColorText('\nCalculating performance ...').bold().custom('gold'))
    
    garden_performance = wrap_defaultdict(dict, 2)
    source_performance = wrap_defaultdict(dict, 2)
    garden_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)
    source_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)
    for (ind_or_pooled, marker_set), offset in unwrap_dictionary(offset_dfs):
        fitness = fitness_mat[ind_or_pooled].copy()
        assert fitness.shape == offset.shape

        # correlation of fitness and offset within gardens across transplants
        garden_performance[ind_or_pooled][marker_set] = offset.corrwith(
            fitness,
            axis='columns',  # across columns for each row
            method='kendall'
        )
        
        # slope of fitness ~ offset within gardens
        for garden in fitness.index:
            garden_slopes[ind_or_pooled][marker_set].loc[garden] = linregress(
                offset.loc[garden, fitness.columns],
                fitness.loc[garden]
            ).slope

        # correlation of fitness and offset for transplants across gardens
        source_performance[ind_or_pooled][marker_set] = offset.corrwith(
            fitness,                                         
            axis='index',  # across rows for each column
            method='kendall')
        if ind_or_pooled == 'ind':  # take average across individuals per pop
            source_performance[ind_or_pooled][marker_set] = source_performance[ind_or_pooled][marker_set].groupby(samppop).mean()
        
        # slope of fitness ~ offset across gardens for individual pops
        if ind_or_pooled == 'ind':
            # get average slope across individuals for each source population
            for source_pop in fitness.index:
                samps = popsamps[source_pop]
                slopes = []
                for samp in samps:
                    slopes.append(
                        linregress(offset[samp].loc[fitness.index], fitness[samp]).slope
                    )
                source_slopes[ind_or_pooled][marker_set].loc[source_pop] = np.mean(slopes)
        else:
            # get slope for each source pop
            for source_pop in fitness.columns:
                # record slope
                source_slopes[ind_or_pooled][marker_set].loc[int(source_pop)] = linregress(
                    offset[source_pop].loc[fitness.index],
                    fitness[source_pop]
                ).slope
        
    # zip up the performance dictionaries
    performance_dicts = {'garden_performance' : garden_performance,
                         'source_performance' : source_performance,
                         'garden_slopes' : garden_slopes,
                         'source_slopes' : source_slopes}
   
    # save
    pkl = op.join(corr_dir, f'{seed}_performance_dicts.pkl')
    pkldump(performance_dicts, pkl)
    print(f'\tsaved performance calculations to:\n\t{pkl}')

    return performance_dicts


def decorate_figure(marker_sets, fig, axes, cmap, vmin=-1, vmax=1, xadjust=0.20, cbar_label="Kendall's $\\tau$"):
    """Add color bar and row labels for each marker set in `marker_sets`.
    
    Parameters
    ----------
    marker_sets - list
        list of marker sets
    fig, axes
        matplotlib constructors
    cmap
        color map
    vmin, vmax
        the extent of values for the min and max of color map
    cbar_label - str
        label for color bar
    """
    # set marker_set label - need to do after filling in all ax's so that ylim is constant
    from matplotlib.lines import Line2D

    # set colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([1.01, 0.094, 0.02, 0.65])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), cax=cbar_ax)
    cbar.set_label(cbar_label, fontdict=dict(fontsize=15))
    
    # add labels
    for row, marker_set in enumerate(marker_sets):
        if isinstance(axes[0], np.ndarray):
            ax = axes[row][0]
        else:
            ax = axes[row]
        ymin, ymax = ax.get_ylim()
        ypos = ymin + ((ymax - ymin) / 2)  # center on axis

        xmin, xmax = ax.get_xlim()
        xpos = xmin - (xadjust * (xmax - xmin))  # subtract to move left of figure
    
        ax.text(xpos, ypos, label_dict[marker_set], rotation='vertical', va='center',
                fontdict=dict(weight='bold', fontsize=12)
               )
        
    pass


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
        - a list of marker sets - see `label_dict`
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
    print(ColorText(f'\nCreating histo boxplots for {performance_name} ...').bold().custom('gold'))
    
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

    # create figs
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), 2,
                             sharey=False,
                             figsize=(10, 10))
        
    # fill in subplots with histograms
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]
        col = 0  # column counter
#         for ind_or_pooled in ['ind', 'pooled']:
        for ind_or_pooled in ['pooled']:  # CHANGE BACK to include 'ind'
            ax = row_axes[col]

            # create the histo_boxplot
            ax_box, ax_hist = histo_box(
                performance_dict[ind_or_pooled][marker_set].copy(),
                ax=ax, histbins=15, histplot_kws=histplot_kws, boxplot_kws=boxplot_kws
            )

            # add in some labels
            if row == 0:
                data_label = label_dict[ind_or_pooled]
                ax_box.set_title(f'{data_label}', fontdict=dict(weight='bold', fontsize=12))
            
            if marker_set == marker_sets[-1]:  # if the last row
                ax_hist.set_xlabel("Kendall's $\\tau$")
            
            # if it's the first column, set ylabel
            if col == 0:
                ax.set_ylabel('count')

            # color in bars of histograph to match their score on the colormap
            mvp13.color_histogram(ax, cmap=cmap, norm=norm)

            # create a background color for each histogram
            gradient_image(ax, direction=0.5, transform=ax.transAxes, 
                           cmap=background_cmap, cmap_range=(0.0, 0.2))

            col += 1
        
    # add labels, title, etc
    fig.suptitle(f'{performance_name}\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    decorate_figure(marker_sets, fig, axes, cmap=cmap)
        
    # save
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
        
    pass


def fill_slope_heatmaps(marker_sets, heatmaps, vmin, vmax):
    """Fill in heatmap subplots to display slope of relationship between fitness and offset."""
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), 2,
                             sharey='row',
                             sharex='col',
                             figsize=(10, 10))
    
    # fill in subplots with histograms
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]
#         for col, ind_or_pooled in enumerate(['ind', 'pooled']):
        for col, ind_or_pooled in enumerate(['pooled']):  # CHANGE BACK to include 'ind'
            ax = row_axes[col]
            
            # add in some labels
            if row == 0:
                data_label = label_dict[ind_or_pooled]
                ax.set_title(f'{data_label}', fontdict=dict(weight='bold', fontsize=12))
            
            heatmap = heatmaps[ind_or_pooled][marker_set]
            if not isinstance(heatmap, pd.DataFrame):  # TODO REMOVE TO ENFORCE SNS ERROR BELOW
                continue
            
            # plot the heatmap
            _ = sns.heatmap(heatmap,
                            cmap='viridis',
                            cbar=False,
                            vmin=vmin,
                            vmax=vmax,
                            ax=ax)
            
            if marker_set == marker_sets[-1]:
                ax.set_xlabel('Longitude (x)')
                
            if col == 0:
                ax.set_ylabel('Latitude (y)')
            
    return fig, axes
    
# def create_garden_slope_plots(performance_name, slope_dict, locations, pdf, marker_sets=['all', 'adaptive', 'neutral']):
#     # determine vmin and vmax
#     minn = math.inf
#     maxx = -math.inf
#     heatmaps = wrap_defaultdict(dict, 2)
#     for (ind_or_pooled, marker_set), garden_slopes in unwrap_dictionary(slope_dict):
#         fitness = fitness_mat[ind_or_pooled].copy()
            
#         # get slopes and fill in the heatmap
#         heatmap = blank_dataframe()
#         for garden, slope in garden_slopes.items():
#             x,y = locations.loc[garden]
#             heatmap.loc[y, x] = slope
#         heatmaps[ind_or_pooled][marker_set] = heatmap.copy()
            
#         hmin = heatmap.min().min()
#         hmax = heatmap.max().max()
#         if hmin < minn:
#             minn = hmin
#         if hmax > maxx:
#             maxx = hmax
    
#     # fill in subplots with histograms
#     fig, axes = fill_slope_heatmaps(marker_sets, heatmaps, vmin=minn, vmax=maxx)

#     fig.suptitle(f'garden performance slope\n{seed = }\n{level}\n', fontsize=15, y=0.98)
#     plt.tight_layout()  # make it pretty
#     decorate_figure(marker_sets, fig, axes, cmap='viridis', vmin=minn, vmax=maxx,
#                     cbar_label="slope of fitness ~ GF offset")
    
#     pdf.savefig(bbox_inches="tight")
#     plt.show()
#     plt.close()
    
#     pass


def create_scatter_plots(offset_dfs, fitness_mat, locations, envdata, samppop, popsamps, pdf,
                         marker_sets=['all', 'adaptive', 'neutral']):
    """Create a figure of the simulated landscape (10x10) with fitness~offset in each location (x, y)."""
    print(ColorText('\nCreating scatter plots').bold().custom('gold'))
    
    # support for nuis_envs
    colormap = {'temp_opt' : 'Reds',
                'sal_opt' : 'Blues_r',
                'TSsd' : 'Greens', 
                'ISO' : 'Oranges',
                'PSsd' : 'Purples'                
               }
    
#     for ind_or_pooled in ['ind', 'pooled']:
    for ind_or_pooled in ['pooled']:  # CHANGE BACK to include 'ind'
        for row, marker_set in enumerate(pbar(marker_sets, desc=ind_or_pooled)):
            offset = offset_dfs[ind_or_pooled][marker_set].copy()
            fitness = fitness_mat[ind_or_pooled].copy()

            for env, env_series in envdata.items():
#                 colormap = 'Reds' if env=='temp_opt' else 'Blues_r'
                cmap = plt.cm.get_cmap(colormap[env])
                
                # determine colors for scatter plot
                if ind_or_pooled == 'ind':
                    indcolors = fitness.columns.map(samppop).map(
                        env_series).to_series(index=fitness.columns.astype(int)).apply(mvp06.color,
                                                                                       cmap=cmap,
                                                                                       norm=norm).to_dict()
                gardencolors = fitness.index.map(
                    env_series).to_series(index=fitness.index).apply(mvp06.color,
                                                                     cmap=cmap,
                                                                     norm=norm).to_dict()

                if ind_or_pooled == 'ind':
                    color_dict = {'garden' : indcolors,
                                  'source' : gardencolors}
                else:
                    color_dict = {'garden' : gardencolors,
                                  'source' : gardencolors}
                
                for garden_or_source in ['garden', 'source']:
                    colors = color_dict[garden_or_source].copy()
                    

                    # plot performance within gardens across source populations
                    mvp06.performance_scatter(offset.copy(),
                                              fitness.copy(),
                                              f'{ind_or_pooled} {label_dict[marker_set]}',
                                              locations,
                                              colors,
                                              pdf,
                                              popsamps=popsamps, norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir,
                                              home_env=env,
                                              program='GF',
                                              garden_or_source=garden_or_source,
                                              ind_or_pooled=ind_or_pooled)

    pass


def create_slope_heatmap_subplots(performance_name, slope_dict, locations, pdf, marker_sets=['all', 'adaptive', 'neutral']):
    """Create a heatmap of the simulated landscape relating the slope of fitness~offset."""
    # determine vmin and vmax, create heatmap dataframes
    minn = math.inf
    maxx = -math.inf
    heatmaps = wrap_defaultdict(dict, 2)
    for (ind_or_pooled, marker_set), garden_slopes in unwrap_dictionary(slope_dict):

        # get slopes and fill in the heatmap
        heatmap = blank_dataframe()
        for garden, slope in garden_slopes.items():  # if performance_name == 'source_performance', garden=source_pop
            x, y = locations.loc[garden]
            heatmap.loc[y, x] = slope

        heatmaps[ind_or_pooled][marker_set] = heatmap.copy()

        hmin = heatmap.min().min()
        hmax = heatmap.max().max()
        if hmin < minn:
            minn = hmin
        if hmax > maxx:
            maxx = hmax

    # fill in subplots with histograms
    fig, axes = fill_slope_heatmaps(marker_sets, heatmaps, vmin=minn, vmax=maxx)

    fig.suptitle(f'{performance_name} slope\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    decorate_figure(marker_sets, fig, axes, cmap='viridis', vmin=minn, vmax=maxx,
                    cbar_label="slope of fitness ~ GF offset")

    # save figure
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
    
    # save heatmaps
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_slope_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\n\tSaved {performance_name} slope heatmaps to: {pkl}')

    pass


def create_heatmap_subplots(performance_dict, performance_name, pdf, samppop, locations,
                            cmap='viridis', use_vmin_vmax=True, marker_sets=['all', 'adaptive', 'neutral']):
    """For garden or source performance, create a heatmap of the simulated landscape showing Kendall's tau."""
    print(ColorText(f'\nCreating heatmap subplots for {performance_name} ...').bold().custom('gold'))
    
    if use_vmin_vmax is True:
        vmin, vmax = mvp13.get_vmin_vmax(performance_dict)
    else:
        vmin = -1
        vmax = 1
        
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), 2,
                             sharey='row',
                             sharex='col',
                             figsize=(10, 10))
        
    # fill in subplots with histograms
    heatmaps = defaultdict(dict)
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]
        col = 0  # column counter
#         for ind_or_pooled in ['ind', 'pooled']:
        for ind_or_pooled in ['pooled']:  # CHANGE BACK to include 'ind'
            ax = row_axes[col]

            corrs = performance_dict[ind_or_pooled][marker_set].copy()

#             if ind_or_pooled == 'ind' and performance_name != 'garden_performance':
#                 # average across individuals for each population
#                 ind_corrs = pd.DataFrame(corrs, columns=['performance'])
#                 ind_corrs['subpopID'] = ind_corrs.index.map(samppop)
#                 corrs = ind_corrs.groupby('subpopID')['performance'].apply(np.mean)

            # fill in heatmap
            df = blank_dataframe()
            for garden, corr in corrs.items():
                x, y = locations.loc[int(garden)]
                df.loc[y, x] = corr
            heatmaps[marker_set][ind_or_pooled] = df.copy()

            # plot heatmap
            _ = sns.heatmap(df,
                            cmap='viridis', cbar=False,
                            ax=ax, vmin=vmin, vmax=vmax
                           )

            # add in some labels
            if row == 0:
                data_label = label_dict[ind_or_pooled]
                ax.set_title(f'{data_label}', fontdict=dict(weight='bold', fontsize=12))

            if marker_set == marker_sets[-1]:  # if the last row
                ax.set_xlabel("Longitude (x)")

            if col == 0:  # if the first column
                ax.set_ylabel('Latitude (y)')

            col += 1
        
    # add labels, title, etc
    fig.suptitle(f'{performance_name}\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    decorate_figure(marker_sets, fig, axes, cmap=cmap, vmin=vmin, vmax=vmax)

    # save figure
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
    
    # save heatmaps
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\n\tSaved {performance_name} heatmaps to: {pkl}')
        
    pass

def fig_wrapper(performance_dicts, offset_dfs, fitness_mat, locations, samppop, popsamps, envdata):
    """Create figs."""
    saveloc = op.join(fig_dir, f'{seed}_GF_figures.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        for performance_name in ['garden_performance', 'source_performance']:
            performance_dict = performance_dicts[performance_name].copy()
            # garden and source performance figs
            create_histo_subplots(performance_dict,
                                  performance_name, 
                                  pdf,
                                  boxplot_kws=dict(
                                      flierprops={
                                          'marker': '.',
                                          'markerfacecolor': 'gray',
                                          'alpha': 0.5,
                                          'markeredgewidth' : 0.0  # remove edge
                                      },
                                      color='lightsteelblue'
                                  ))

            # garden and source performance heatmaps
            create_heatmap_subplots(performance_dict, performance_name, pdf, samppop, locations)

            # garden performance slope of fitness ~ offset
            slope_group = performance_name.split("_")[0]
            create_slope_heatmap_subplots(performance_name, performance_dicts[f'{slope_group}_slopes'], locations, pdf)
                
    print(ColorText(f'\nsaved fig to: {saveloc}').bold())

    # save scatterplots separately so computers don't get slow trying to display everything
    saveloc = op.join(fig_dir, f'{seed}_GF_figures_scatter.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        # color for the environment (temp_opt, sal_opt) of source_pop - TODO: infer selected envs from data
        create_scatter_plots(offset_dfs, fitness_mat, locations, envdata, samppop, popsamps, pdf)
        
    print(ColorText(f'\nsaved scatter fig to: {saveloc}').bold())

    pass


def main(expected):
    # get a list of subsampled individuals, their info, population locations for each subpopID, and climate optima
    subset, locations, envdata = mvp06.get_pop_data(slimdir, seed)

    # map subpopID to list of samps - key = subpopID val = list of individual sample names
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()

    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))
    
    # get fitness matrices for individuals and pops (pops are mean fitness)
    fitness_mat = {'ind': load_ind_fitness_matrix(slimdir, seed, subset),
                   'pooled': mvp06.load_pooled_fitness_matrix(slimdir, seed)}
    
    # get predicted offset
    offset_dfs = get_offset_predictions(seed, expected, exclude)
    
    # calculate validation scores
    performance_dicts = calculate_performance(offset_dfs, fitness_mat, popsamps, samppop)
    
    # create figs
    fig_wrapper(performance_dicts, offset_dfs, fitness_mat, locations, samppop, popsamps, envdata)
    
    # DONE!
    print(ColorText('\nDONE!!').bold().green())
    print(f'\ttime to complete: {formatclock(dt.now() - t1, exact=True)}\n')
    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, gf_parentdir, *expected = sys.argv
    
    if len(expected) > 0:
        expected = expected[0]
        if len(expected) > 1 and expected[1] == 'pooled':
            exclude = '_pooled_'
        else:
            exclude = '_ind_'
    else:
        expected = 300

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()
    
    # background color for figures
    background_cmap = create_cmap(['white', 'gold'], grain=1000)
    
    # details about demography and selection
    level = mvp10.read_params_file(slimdir).loc[seed, 'level']
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    # get dirs
    fitting_dir = op.join(gf_parentdir, 'fitting/fitting_outfiles')
    training_outdir = op.join(gf_parentdir, 'training/training_outfiles')
    fig_dir = makedir(op.join(gf_parentdir, 'validation/figs'))
    corr_dir = makedir(op.join(op.dirname(fig_dir), 'corrs'))
    offset_dir = makedir(op.join(op.dirname(fig_dir), 'offset_dfs'))
    heat_dir = makedir(op.join(op.dirname(fig_dir), 'heatmaps'))
    
    # dict for pretty labels in figures
    label_dict = {
        'all' : 'all loci',
        'adaptive' : 'causal loci',
        'neutral' : 'neutral loci',
        'ind': 'individual',
        'pooled': 'pooled',
        'sal_opt' : 'sal',
        'temp_opt' : 'temp',
        'ISO' : 'ISO',
        'PSsd' : 'PSsd',
        'TSsd' : 'TSsd'
    }
    
    # set global variables needed for `mvp06.color`, `mvp06.performance_scatter`, and `mvp13.color_histogram`
    norm = Normalize(vmin=-1.0, vmax=1.0)
    
    # set for namespace
    mvp06.label_dict = label_dict
    mvp06.level = level
    
    main(expected)
