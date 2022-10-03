"""Validate RONA with mean individual fitness per pop from the simulation data.

Validation is done by correlating population mean fitness with RONA (for each env).

Usage
-----
conda activate mvp_env
python MVP_06_validate_RONA.py seed slimdir rona_outdir

Parameters
----------
seed
    the seed number of the simulation - used to find associated files
slimdir
    the location of the seed's files output by Katie's post-processing scripts
rona_outdir
    path to directory created in MVP_05_train_RONA.py ending in: RONA/training/training_outfiles

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon completion of MVP_05_train_RONA.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from myfigs import histo_box, gradient_image, create_cmap

import MVP_01_train_gradient_forests as mvp01
import MVP_03_validate_gradient_forests as mvp03
import MVP_10_train_lfmm2_offset as mvp10
import MVP_13_RDA_validation as mvp13

import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress


def color(val, cmap=None, norm=None):
    """Return rgb on colormap `cmap`."""
    return cmap(norm(val))[:3]


def calculate_performance(fitness, marker_sets=['all', 'adaptive', 'neutral']):
    """Calculate correlation between fitness and offset within and across gardens."""
    print(ColorText("\nCalculating performance ...").bold().custom('gold'))
    
    garden_performance = wrap_defaultdict(dict, 2)
    source_performance = wrap_defaultdict(dict, 2)
    garden_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)
    source_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 2)

    offset_dfs = defaultdict(dict)
    for marker_set in marker_sets:
        # load RONA estimates
        rona = pklload(op.join(rona_outdir, f'{seed}_{marker_set}_RONA_results.pkl'))
        
        for env, rona_dict in rona.items():
            # format rona predictions in the same format as fitness dataframe
                # pd.DataFrame(rona[env]).T is necessary so that source deme is col and garden is row
#             rona_offset = pd.DataFrame(rona_dict).T
            rona_offset = pd.DataFrame(rona_dict).T
            rona_offset.columns = rona_offset.columns.astype(str)
            rona_offset.index = rona_offset.index.astype(int)
            offset_dfs[marker_set][env] = rona_offset.copy()
            
            # correlation of fitness and offset within gardens across transplants
            garden_performance[marker_set][env] = offset_dfs[marker_set][env].corrwith(
                fitness,
                axis='columns',  # across columns for each row
                method='kendall')      

            # slope of fitness ~ offset within gardens
            for garden in pbar(fitness.index, desc=f'{marker_set} {env} garden performance'):
                # record slope
                garden_slopes[marker_set][env].loc[garden] = linregress(
                    offset_dfs[marker_set][env].loc[garden, fitness.columns],
                    fitness.loc[garden]
                ).slope        
        
            # correlation of fitness and offset for transplants across gardens
            source_performance[marker_set][env] = offset_dfs[marker_set][env].corrwith(
                fitness,
                axis='index',  # across rows for each column
                method='kendall'
            )      

            # slope of fitness ~ offset across gardens
            for source_pop in pbar(fitness.columns, desc=f'{marker_set} {env} population performance'):
                # record slope
                source_slopes[marker_set][env].loc[source_pop] = linregress(
                    offset_dfs[marker_set][env][source_pop].loc[fitness.index],
                    fitness[source_pop]
                ).slope
                
    performance_dicts = {'garden_performance' : garden_performance,
                         'source_performance' : source_performance,
                         'garden_slopes' : garden_slopes,
                         'source_slopes' : source_slopes
                        }
    # save performance dicts
    pkl = op.join(corr_dir, f'{seed}_performance_dicts.pkl')
    pkldump(performance_dicts, pkl)
    print(f'\n\twrote performance_dicts to : {pkl}')
    
    # save offsets
    pkl = op.join(offset_dir, f'{seed}_offset_dfs.pkl')
    pkldump(offset_dfs, pkl)
    print(f'\twrote offset_dfs to : {pkl}')

    return performance_dicts, offset_dfs


def get_pop_data(slimdir, seed):
    """Get coordinates for each population in the simulations."""
    # get the individuals that were subsampled from full simulation
    subset = mvp01.read_ind_data(slimdir, seed)

    # get x and y coords for each population
    locations = subset.groupby('subpopID')[['x', 'y']].apply(np.mean)
    locations.columns = ['lon', 'lat']  # index = subpopID

    # get envdata for each subpopID
    envdata = subset.groupby('subpopID')[['sal_opt', 'temp_opt']].apply(np.mean)
    
    return subset, locations, envdata


def load_pooled_fitness_matrix(slimdir, seed):
    """Load fitness matrix output from simulations."""
    # an n_deme x n_deme table that indicates the mean fitness of individuals 
        # from the source deme (in columns) in the transplant deme (in rows) 
        
    fitness = pd.read_table(op.join(slimdir, f'{seed}_fitnessmat.txt'),
                            delim_whitespace=True,
                            header=None)
    
    assert fitness.shape == (100, 100)

    # set column names for popID
    fitness.columns = [str(pop) for pop in range(1, 101, 1)]
    fitness.index = [pop for pop in range(1, 101, 1)]

    return fitness


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
        for env in ['sal_opt', 'temp_opt']:
            ax = row_axes[col]

            # create the histo_boxplot
            ax_box, ax_hist = histo_box(
                performance_dict[marker_set][env].copy(),
                ax=ax, histbins=15, histplot_kws=histplot_kws, boxplot_kws=boxplot_kws
            )

            # add in some labels
            if row == 0:
                data_label = label_dict[env]
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
        
    # set main title
    fig.suptitle(f'{performance_name}\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    mvp03.decorate_figure(marker_sets, fig, axes, cmap=cmap, xadjust=0.18)
        
    # save
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
        
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
        for env in ['sal_opt', 'temp_opt']:
            ax = row_axes[col]

            corrs = performance_dict[marker_set][env].copy()

            # fill in heatmap
            df = mvp03.blank_dataframe()
            for garden, corr in corrs.items():
                x, y = locations.loc[int(garden)]
                df.loc[y, x] = corr
            heatmaps[marker_set][env] = df.copy()

            # plot heatmap
            _ = sns.heatmap(df,
                            cmap='viridis', cbar=False,
                            ax=ax, vmin=vmin, vmax=vmax
                           )

            # add in some labels
            if row == 0:
                data_label = label_dict[env]
                ax.set_title(f'{data_label}', fontdict=dict(weight='bold', fontsize=12))

            if marker_set == marker_sets[-1]:  # if the last row
                ax.set_xlabel("Longitude (x)")

            if col == 0:  # if the first column
                ax.set_ylabel('Latitude (y)')

            col += 1
        
    # add labels, title, etc
    fig.suptitle(f'{performance_name}\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    mvp03.decorate_figure(marker_sets, fig, axes, cmap=cmap, vmin=vmin, vmax=vmax, xadjust=0.18)

    # save figure
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
    
    # save objects
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\n\twrote heatmaps to : {pkl}')
        
    pass


def fill_slope_heatmaps(marker_sets, heatmaps, vmin, vmax, total_traits=['sal_opt', 'temp_opt']):
    """Fill in heatmap subplots to display slope of relationship between fitness and offset."""
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), len(total_traits),
                             sharey='row',
                             sharex='col',
                             figsize=(10, 10) if len(total_traits) == 2 else (5, 10))
    
    # fill in subplots with histograms
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]
        for col, env in enumerate(total_traits):
            if len(total_traits) == 2:
                ax = row_axes[col]
            else:
                ax = row_axes
            
            # add in some labels
            if row == 0:
                data_label = label_dict[env]
                ax.set_title(f'{data_label}', fontdict=dict(weight='bold', fontsize=12))
            
            heatmap = heatmaps[marker_set][env]
            assert isinstance(heatmap, pd.DataFrame)
            
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


def create_slope_heatmap_subplots(performance_name, slope_dict, locations, pdf, marker_sets=['all', 'adaptive', 'neutral'], 
                                  total_traits=['sal_opt', 'temp_opt'], program='RONA'):
    """Create a heatmap of the simulated landscape relating the slope of fitness~offset."""
    # determine vmin and vmax across all heatmaps
    minn = math.inf
    maxx = -math.inf
    heatmaps = wrap_defaultdict(dict, 2)
    for (marker_set, env), garden_slopes in unwrap_dictionary(slope_dict):
            
        # get slopes and fill in the heatmap
        heatmap = mvp03.blank_dataframe()
        for garden, slope in garden_slopes.items():
            x, y = locations.loc[int(garden)]
            heatmap.loc[y, x] = slope
        heatmaps[marker_set][env] = heatmap.copy()
        
        # determine min and max
        hmin = heatmap.min().min()
        hmax = heatmap.max().max()
        if hmin < minn:
            minn = hmin
        if hmax > maxx:
            maxx = hmax
            
    # fill in subplots with histograms
    fig, axes = fill_slope_heatmaps(marker_sets, heatmaps, vmin=minn, vmax=maxx, total_traits=total_traits)

    fig.suptitle(f'garden performance slope\n{seed = }\n{level}\n', fontsize=15, y=0.98)
    plt.tight_layout()  # make it pretty
    mvp03.decorate_figure(marker_sets, fig, axes, cmap='viridis', vmin=minn, vmax=maxx, xadjust=0.18,
                          cbar_label=f"slope of fitness ~ {program} offset")
    
    # save figure
    pdf.savefig(bbox_inches="tight")
    plt.show()
    plt.close()
    
    # save objects
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_slope_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\twrote heatmaps to : {pkl}')
    
    pass


def fig_setup(locations):
    """Get figure position (order) of each population on a 10x10 subplot."""
    count = 0
    figpos = {}
    for y in reversed(range(1, 11, 1)):
        for x in range(1, 11, 1):
            pop = locations[(locations['lon']==x) & (locations['lat']==y)].index[0]
            figpos[count] = pop
            count += 1
            
    # set up big fig
    fig, axes = plt.subplots(10, 10,
                             sharex='all',
                             sharey='all',
                             figsize=(15, 10))
    return figpos, fig, axes


def performance_scatter(
    offset, fitness, figlabel, locations, colors, pdf, popsamps=None, cmap=None, norm=None, seed=None, fig_dir=None,
    program='RONA', home_env=None, rona_env='', garden_or_source='garden', ind_or_pooled='pooled'
):
    """Create a map of pops using coords, show relationsip between RONA and fitness."""
    figpos, fig, axes = fig_setup(locations)
    
    # create each of the population subfigures in the order matplotlib puts them into the figure
    for subplot, ax in enumerate(axes.flat):
        garden = figpos[subplot]  # which garden now?; if garden_or_source=='source' object garden is read as 'source'
        if garden_or_source == 'garden':
            ax.scatter(offset.loc[garden, fitness.columns],
                       fitness.loc[garden],
                       c=fitness.columns.astype(int).map(colors))
        else:
            if ind_or_pooled == 'ind':
                for samp in popsamps[garden]:
                    ax.scatter(offset.loc[fitness.index, samp],
                               fitness[samp],
                               color=colors[garden]
                              )
            else:
                assert ind_or_pooled == 'pooled'
                ax.scatter(offset.loc[fitness.index, str(garden)],
                           fitness[str(garden)],
                           c=[colors[garden] for pop in fitness.index])  # only color home garden the same color

        # decide if I need to label longitude (x) or latitude (y) axes
        x, y = locations.loc[garden]  
        if subplot in range(0, 110, 10):
            ax.set_ylabel(int(y))
        if subplot in range(90, 101, 1):
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(45)
                label.set_rotation_mode('anchor')
            ax.set_xlabel(int(x))

    # set colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:,:])
    cbar.ax.set_title(label_dict[home_env])

    # is this RONA?
    if rona_env != '':
        rona_env = f' associated to {rona_env}'
    
    # set labels
    fig.supylabel('fitness', x=0.08, ha='center', va='center', fontsize=14, weight='bold')
    fig.supxlabel('predicted offset', x=0.431, y=0.045, ha='center', va='center', fontsize=14, weight='bold')
    fig.suptitle(
        f'{seed}\n' +\
        f'{program} {garden_or_source} performance for {label}{rona_env}\n' +\
        f'transplanted pops colored by home environment\n{level}'
    )

    # save
    if fig_dir is not None:
        pdf.savefig(bbox_inches="tight")
    
    plt.show()
    plt.close()
    
    del figpos, fig, axes
    
    pass


def color(val, cmap=None, norm=None):
    """Return rgb on colormap `cmap`."""
    return cmap(norm(val))[:3]


def scatter_wrapper(offset_dfs, fitness, envdata, locations, pdf, marker_sets=['all', 'adaptive', 'neutral'],
                    garden_or_source='garden'):
    """Wrapper for `performance_scatter`."""
    print(ColorText('\nCreating scatter plots ...').bold().custom('gold'))
    for marker_set in marker_sets:
        for home_env in pbar(['sal_opt', 'temp_opt'], desc=marker_set):  # the environment for used to color populations
            for rona_env in ['sal_opt', 'temp_opt']:  # the environment used to calculate RONA
                offset = offset_dfs[marker_set][rona_env]  # already transposed properly
                
                # determine colors for scatter plot
                colormap = 'Reds' if home_env=='temp_opt' else 'Blues_r'
                cmap = plt.cm.get_cmap(colormap)
                colors = fitness.index.map(envdata[home_env]).to_series(index=fitness.index).apply(color,
                                                                                                   cmap=cmap,
                                                                                                   norm=norm).to_dict()
                
                for garden_or_source in ['garden', 'source']:
                    # plot performance within gardens across source populations
                    performance_scatter(offset.copy(),
                                        fitness.copy(),
                                        f'{label_dict[marker_set]}',
                                        locations,
                                        colors,
                                        pdf,
                                        norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir,
                                        rona_env=rona_env, home_env=home_env,
                                        garden_or_source=garden_or_source)        

    pass


def fig_wrapper(performance_dicts, samppop, offset_dfs, fitness, envdata, locations):
    """Create figs."""
    # save histograms and heatmaps
    saveloc = op.join(fig_dir, f'{seed}_RONA_figures.pdf')
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
            create_heatmap_subplots(performance_dict, performance_name, pdf, samppop, locations)
            
            # garden performance slope of fitness ~ offset
            slope_group = performance_name.split("_")[0]
            create_slope_heatmap_subplots(performance_name, performance_dicts[f'{slope_group}_slopes'].copy(), locations, pdf)

#             if performance_name == 'garden_performance':
#                 create_slope_heatmap_subplots(performance_name, performance_dicts['garden_slopes'].copy(), locations, pdf)

#             if performance_name == 'source_performance':
#                 create_slope_heatmap_subplots(performance_name, performance_dicts['source_slopes'].copy(), locations, pdf)

    print(ColorText(f'\nsaved fig to: {saveloc}').bold())

    # save scatterplots separately so computers don't get slow trying to display everything
    saveloc = op.join(fig_dir, f'{seed}_RONA_figures_scatter.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        scatter_wrapper(offset_dfs, fitness, envdata, locations, pdf)

    print(ColorText(f'\nsaved scatter fig to: {saveloc}').bold())

    pass


def main():
    # get pop data
    print(ColorText('\nGetting population information ...').bold().custom('gold'))
    
    # get fitness matrix for pops
    fitness = load_pooled_fitness_matrix(slimdir, seed)

    # get a list of subsampled individuals, map samp top subpopID, and get population locations for each subpopID
    subset, locations, envdata = get_pop_data(slimdir, seed)

    # map subpopID to list of samps - key = subpopID val = list of individual sample names
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()

    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))
    
    # calculate correlations within and across gardens
    performance_dicts, offset_dfs = calculate_performance(fitness)
    
    # create figs
    fig_wrapper(performance_dicts, samppop, offset_dfs, fitness, envdata, locations)
    
    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')


if __name__ == '__main__':
    # get input arguments
    thisfile, seed, slimdir, rona_outdir = sys.argv

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()
    
    # details about demography and selection
    level = mvp10.read_params_file(slimdir).loc[seed,'level']

    # create dirs
    rona_dir = op.dirname(op.dirname(rona_outdir))
    fig_dir = makedir(op.join(rona_dir, 'validation/figs'))
    heat_dir = makedir(op.join(op.dirname(fig_dir), 'heatmap_objects'))
    corr_dir = makedir(op.join(op.dirname(fig_dir), 'corrs'))
    offset_dir = makedir(op.join(op.dirname(fig_dir), 'offset_dfs'))

    # set globally
    norm = Normalize(vmin=-1.0, vmax=1.0)
    
    # dict for pretty labels in figures
    label_dict = {
        'all' : 'all loci',
        'adaptive' : 'causal loci',
        'neutral' : 'neutral loci',
        'sal_opt' : 'sal',
        'temp_opt' : 'temp'
    }
    mvp03.label_dict = label_dict
    
    # background color for figures
    background_cmap = create_cmap(['white', 'gold'], grain=1000)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()

