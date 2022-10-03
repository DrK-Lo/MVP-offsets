"""Validate offset predictions from lfmm using population mean fitness.

Usage
-----
conda activate mvp_env
python MVP_11_validate_lfmm2_offset.py seed slimdir outerdir

"""
from pythonimports import *
from myfigs import *

import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

import MVP_01_train_gradient_forests as mvp01
import MVP_03_validate_gradient_forests as mvp03
import MVP_06_validate_RONA as mvp06
import MVP_10_train_lfmm2_offset as mvp10
import MVP_11_validate_lfmm2_offset as mvp11


def retrieve_offset_data(seed):
    """Read in RDA offset data, assert expected number of files.
    
    Parameters
    ----------
    seed - str
        - simulation seed ID for finding files
        
    Returns
    -------
    offset_dfs - nested dictionary
        - final value is pd.DataFrame in same format as fitness dataframes    
    """
    print(ColorText('\nRetrieving offset data ...').bold().custom('gold'))
    
    # get predicted offset files
    offset_files = fs(rda_outdir, startswith=seed, endswith='offset.txt')
    
    # make sure there are the expected number of files
    factor = 2 if ntraits == 1 else 1
    expected_num = 2*4*2*factor  # pool/ind * TRUE/FALSE/CAUSAL/NEUTRAL * structcorr/uncorr * factor
    assert len(offset_files) == expected_num, f'{len(offset_files) = }, {expected_num = }'
    
    # read in predicted offset data
    offset_dfs = wrap_defaultdict(dict, 3)  # defaultdict(dict) if ntraits=2
    for f in offset_files:
        seed, ind_or_pooled, marker_set, which_traits, structcorr, *_ = op.basename(f).split("_")
        df = pd.read_table(f, index_col=0, delim_whitespace=True)
        offset_dfs[ind_or_pooled][marker_set][which_traits][structcorr] = df
        print(seed, ind_or_pooled, marker_set, which_traits, structcorr)

    # save offsets
    pkl = op.join(offset_dir, f'{seed}_offset_dfs.pkl')
    pkldump(offset_dfs, pkl)
    print(f'\n\twrote offset_dfs to : {pkl}')
    
    return offset_dfs


def retrieve_fitness_data(slimdir, seed, subset):
    """Retrieve individual and deme fitness across gardens.
    
    Returns
    -------
    fitness_mat - dict
        - keys for ind_or_pooled, values = pd.DataFrame
    """
    print(ColorText('\nRetrieving fitness data ...').bold().custom('gold'))
    
    # get fitness matrices for individuals and pops (pops are mean fitness)
    pfit = mvp06.load_pooled_fitness_matrix(slimdir, seed)
    pfit.columns = pfit.columns.astype(str)

    fitness_mat = {'ind': mvp03.load_ind_fitness_matrix(slimdir, seed, subset),
                   'pooled': pfit.copy()}
    
    return fitness_mat


def calculate_performance(offset_dfs, fitness_mat, popsamps, samppop):
    """Calculate correlation between fitness and offset within and across gardens."""
    print(ColorText('\nCalculating performance ...').bold().custom('gold'))
    
    garden_performance = wrap_defaultdict(dict, 3)
    source_performance = wrap_defaultdict(dict, 3)
    garden_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 4)
    source_slopes = wrap_defaultdict(pd.Series(dtype=float).copy, 4)

    for (ind_or_pooled, marker_set, which_traits, structcorr), offset in unwrap_dictionary(offset_dfs):
        fitness = fitness_mat[ind_or_pooled].copy()
        assert fitness.shape == offset.shape

        # correlation of fitness and offset within gardens across transplants
        garden_performance[ind_or_pooled][marker_set][which_traits][structcorr] = offset.corrwith(
            fitness,
            axis='columns',  # across columns for each row
            method='kendall'
        )

        # slope of fitness ~ offset within gardens
        for garden in fitness.index:
            # record slope
            garden_slopes[ind_or_pooled][marker_set][which_traits][structcorr].loc[garden] = linregress(
                offset.loc[garden, fitness.columns],
                fitness.loc[garden]
            ).slope

        # correlation of fitness and offset for transplants across gardens
        source_performance[ind_or_pooled][marker_set][which_traits][structcorr] = offset.corrwith(
            fitness,                                         
            axis='index', # across rows for each column
            method='kendall'
        )

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
                source_slopes[ind_or_pooled][marker_set][which_traits][structcorr].loc[source_pop] = np.mean(slopes)
        else:
            # get slope for each source pop
            for source_pop in fitness.columns:
                # record slope
                source_slopes[ind_or_pooled][marker_set][which_traits][structcorr].loc[int(source_pop)] = linregress(
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


def color_histogram(ax, cmap, norm):
    """Color each bar of a histogram in `ax` with a color from a color map, `cmap`."""
    for patch in ax.patches:
        tau = patch.get_x()
        color = mvp06.color(tau, cmap=cmap, norm=norm)
        patch.set_facecolor(color)

    pass


def decorate_figure(marker_sets, fig, axes, title=None, cmap='viridis', vmin=-1, vmax=1, xadjust=0.39,
                    cbar_label="Kendall's $\\tau$", program='RDA'):
    """For histo_subplots and heatmap_subplots, add main figure title, labels for rows (`marker_sets`) columns."""
    # set marker_set label - need to do after filling in all ax's so that ylim is constant
    from matplotlib.lines import Line2D
        
    # set main title
    fig.suptitle(title,
                 fontsize=15,
                 y=0.98)

    # set colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([1.01, 0.094, 0.02, 0.65])
    cbar = fig.colorbar(sm,
                        ax=axes[:,:] if program=='RDA' else axes[:],
                        cax=cbar_ax)
    cbar.set_label(cbar_label, fontdict=dict(fontsize=15))

    # make it pretty
    fig.tight_layout()
    
    for row, marker_set in enumerate(marker_sets):
        if program != 'RDA' and ntraits==2:
            ax = axes[row]
        else:
            ax = axes[row][0]
        ymin, ymax = ax.get_ylim()
        ypos = ymin + ((ymax - ymin) / 2)  # center on axis

        xmin, xmax = ax.get_xlim()
        xpos = xmin - (xadjust * (xmax - xmin))  # subtract to move left of figure
    
        ax.text(xpos, ypos, label_dict[marker_set], rotation='vertical', va='center',
                fontdict=dict(weight='bold', fontsize=12)
               )

    # set ntrait label - do after tight_layout so positions are correct
    if program == 'RDA':
        total_traits = zip([0, 2], ['1-env RDA', '2-env RDA']) if ntraits == 1 else zip([0], ['2-env RDA'])
#         ypos = 0.82
        for col, which_trait in total_traits:
            xmin = axes[0][col].get_position().x0
            xmax = axes[0][col + 1].get_position().x1
            xpos = xmin + ((xmax - xmin) / 2)

            ymin = axes[0][col].get_position().y0
            ymax = axes[0][col].get_position().y1
            ypos = ymax + (0.18 * (ymax - ymin))

            fig.text(xpos, ypos, which_trait, ha='center', fontdict=dict(weight='bold', fontsize=12))
            
    elif 'lfmm' in program:
        if ntraits == 1:
            total_traits = zip([axes[0][0], axes[0][1]],
                               [f'1-env {program}', f'2-env {program}'])
        else:
            total_traits = zip([axes[0]],
                               [f'2-env {program}'])
            
        for ax, which_trait in total_traits:
            xmin = ax.get_position().x0
            xmax = ax.get_position().x1
            xpos = xmin + ((xmax - xmin) / 2)
            
            ymin = ax.get_position().y0
            ymax = ax.get_position().y1
            ypos = ymax + (0.035 * (ymax - ymin))
            
            fig.text(xpos, ypos, which_trait, ha='center', fontdict=dict(weight='bold', fontsize=12))

    else:
        raise Exception(f'Program `{program}` is not accounted for in code.')
        
    pass


def create_histo_subplots(performance_dict, performance_name, pdf, cmap='viridis', N=5,
                          marker_sets=['FALSE', 'CAUSAL', 'NEUTRAL', 'TRUE'],
                          histplot_kws={}, boxplot_kws={}):
    """Create a panel of myfigs.histo_box plots for performance of either garden or source (within `performance_dict`).
    
    Parameters
    ----------
    performance_dict - nested dictionary
        - contstructed as - performance_dict[ind_or_pooled][marker_set][which_traits][structcorr]
        - the final value is a pandas.Series of Kendall's tau validation scores (cor between offset
            and prediction)
    performance_name - str
        - the type of performance measured, used for figure label and file; either 'garden performance'
            or 'source performance'
    pdf
        - file opened with from PdfPages
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

    # create figs
    for ind_or_pooled in ['ind', 'pooled']:
        # create subplots to fill in
        fig, axes = plt.subplots(len(marker_sets), len(total_traits) * 2,
                                 sharey='all', layout='tight',
                                 figsize=(10, 10) if ntraits == 1 else (5, 10))
        
        # fill in subplots with histograms
        for row, marker_set in enumerate(marker_sets):
            row_axes = axes[row]

            col = 0  # column counter
            for which_traits in total_traits:
                for structcorr in ['nocorr', 'structcorr']:
                    ax = row_axes[col]

                    # create the histo_boxplot
                    ax_box, ax_hist = histo_box(
                        performance_dict[ind_or_pooled][marker_set][which_traits][structcorr].copy(),
                        ax=ax, histbins=15, histplot_kws=histplot_kws, boxplot_kws=boxplot_kws
                    )

                    # add in some labels
                    if row == 0:
                        structure_label = label_dict[structcorr]
                        ax_box.set_title(f'{structure_label}')

                    if marker_set == marker_sets[-1]:  # if the last row
                        ax_hist.set_xlabel("Kendall's $\\tau$")
                        
                    if col == 0:  # if the first column
                        ax.set_ylabel('count')

                    # color in bars of histograph to match their score on the colormap
                    color_histogram(ax, cmap=cmap, norm=norm)

                    # create a background color for each histogram
                    gradient_image(ax, direction=0.5, transform=ax.transAxes, 
                                   cmap=background_cmap, cmap_range=(0.0, 0.2))

                    col += 1
        
        # add labels, title, etc
        decorate_figure(marker_sets, fig, axes, vmin=-1, vmax=1, program='RDA',
                        title=f'{performance_name}\n{seed = }\n{ind_or_pooled = }\n{level}\n')
        
        # save
        pdf.savefig(bbox_inches="tight")
        plt.show()
        plt.close()
        
    pass


def get_vmin_vmax(performance_dict):
    """Get min and max Kendall's values from permutations of validation in `performance_dict`."""
    vals = []
    for *keys, corrs in unwrap_dictionary(performance_dict):
        vals.append(corrs.tolist())

    all_vals = flatten(vals)

    vmin = min(all_vals)
    vmax = max(all_vals)

    return vmin, vmax


def create_heatmap_subplots(performance_dict, performance_name, pdf, samppop, locations,
                            cmap='viridis', use_vmin_vmax=True, marker_sets=['FALSE', 'CAUSAL', 'NEUTRAL', 'TRUE']):
    """Create a panel of heatmaps for performance of either garden or source (within `performance_dict`).
    
    Parameters
    ----------
    performance_dict - nested dictionary
        - contstructed as - performance_dict[ind_or_pooled][marker_set][which_traits][structcorr]
        - the final value is a pandas.Series of Kendall's tau validation scores (cor between offset
            and prediction)
    performance_name - str
        - the type of performance measured, used for figure label and file; either 'garden performance'
            or 'source performance'
    pdf
        - file opened with from PdfPages
    samppop
        - dict with key=samp, val=subpopID
    locations - pd.DataFrame
        - rows for subpopID, columns for x, y coordinates
    cmap
        - the type of color map to signify the sign and magnitude of Kendall's tau
    use_vmin_vmax - bool
        - whether to use the full range (-1 to 1) for heatmap colors or the min and max of performance_dict
    marker_sets - list
        - a list of marker sets - see `label_dict`
    """
    print(ColorText(f'\nCreating heatmap subplots for {performance_name} ...').bold().custom('gold'))
    
    # differences between ntraits=1 and ntraits=2
    total_traits = ['ntraits-1', 'ntraits-2'] if ntraits == 1 else ['ntraits-2']
    
    if use_vmin_vmax is True:
        vmin, vmax = get_vmin_vmax(performance_dict)
#         print(f'{vmin = } {vmax = }')
    else:
        vmin = -1
        vmax = 1
    
    # create heatmap subplots
    heatmaps = wrap_defaultdict(dict, 3)
    for ind_or_pooled in ['ind', 'pooled']:
        # create subplots to fill in
        fig, axes = plt.subplots(len(marker_sets), len(total_traits) * 2,
                                 sharey='all', layout='tight',
                                 figsize=(10, 10) if ntraits == 1 else (5, 10))
    
        # fill in subplots with histograms
        for row, marker_set in enumerate(marker_sets):
            row_axes = axes[row]

            col = 0  # column counter
            for which_traits in total_traits:
                for structcorr in ['nocorr', 'structcorr']:
                    ax = row_axes[col]
                    
                    try:
                        corrs = performance_dict[ind_or_pooled][marker_set][which_traits][structcorr].copy()
                    except KeyError as e:
                        print(ind_or_pooled, marker_set, which_traits, structcorr)
                        raise e

                    if ind_or_pooled == 'ind' and performance_name != 'garden_performance':
                        # average across individuals for each population
                        ind_corrs = pd.DataFrame(corrs, columns=['performance'])
                        ind_corrs['subpopID'] = ind_corrs.index.map(samppop)
                        corrs = ind_corrs.groupby('subpopID')['performance'].apply(np.mean)

                    # fill in heatmap
                    df = mvp03.blank_dataframe()
                    for garden, corr in corrs.items():
                        x, y = locations.loc[int(garden)]
                        df.loc[y, x] = corr
                    heatmaps[ind_or_pooled][marker_set][which_traits][structcorr] = df.copy()

                    # plot heatmap
                    _ = sns.heatmap(df,
                                    cmap='viridis', cbar=False,
                                    ax=ax, vmin=vmin, vmax=vmax
                                   )
                    
                    # add in some labels
                    if row == 0:
                        structure_label = label_dict[structcorr]
                        ax.set_title(f'{structure_label}')

                    if marker_set == marker_sets[-1]:  # if the last row
                        ax.set_xlabel("longitude")
                        
                    if col == 0:  # if the first column
                        ax.set_ylabel('latitude')
                    
                    col += 1
        
        # add labels, title, etc
        decorate_figure(marker_sets, fig, axes, title=f'{performance_name}\n{seed = }\n{ind_or_pooled = }\n{level}\n',
                        vmin=vmin, vmax=vmax, program='RDA')
        
        # save fig
        pdf.savefig(bbox_inches="tight")
        plt.show()
        plt.close()
        
        # save heatmaps
        pkl = op.join(heat_dir, f'{seed}_{performance_name}_heatmaps.pkl')
        pkldump(heatmaps, pkl)
        print(f'\n\tSaved {performance_name} heatmaps to: {pkl}')
        
    pass


def fill_slope_heatmaps(marker_sets, heatmaps, vmin, vmax, total_traits=['sal_opt', 'temp_opt']):
    """Fill in heatmap subplots to display slope of relationship between fitness and offset."""
    
    # create subplots to fill in
    fig, axes = plt.subplots(len(marker_sets), len(total_traits) * 2,
                             sharey='row',
                             sharex='col',
                             figsize=(10, 10) if ntraits == 1 else (5, 10))

    # fill in subplots with histograms
    for row, marker_set in enumerate(marker_sets):
        row_axes = axes[row]

        col = 0  # column counter
        for which_traits in total_traits:
            for structcorr in ['nocorr', 'structcorr']:
                ax = row_axes[col]

                try:
                    heatmap = heatmaps[marker_set][which_traits][structcorr].copy()
                except KeyError as e:
                    print(marker_set, which_traits)
                    raise e
                    
                assert isinstance(heatmap, pd.DataFrame)

                # plot the heatmap
                _ = sns.heatmap(heatmap,
                                cmap='viridis',
                                cbar=False,
                                vmin=vmin,
                                vmax=vmax,
                                ax=ax)
                
                # add in some labels
                if row == 0:
                    structure_label = label_dict[structcorr]
                    ax.set_title(f'{structure_label}')

                if marker_set == marker_sets[-1]:
                    ax.set_xlabel('Longitude (x)')

                if col == 0:
                    ax.set_ylabel('Latitude (y)')
                
                col += 1

    return fig, axes


def create_slope_heatmap_subplots(performance_name, slope_dict, locations, pdf,
                                  marker_sets=['FALSE', 'CAUSAL', 'NEUTRAL', 'TRUE'], 
                                  total_traits=['sal_opt', 'temp_opt']):
    """Fill in heatmap subplots to display slope of relationship between fitness and offset."""
    print(ColorText(f'\nCreating slope heatmaps for {performance_name}').bold().custom('gold'))
    
    heatmaps = wrap_defaultdict(dict, 3)
    for ind_or_pooled in ['ind', 'pooled']:
        # determine vmin and vmax across all heatmaps
        vmin = math.inf
        vmax = -math.inf
        for (marker_set, which_traits, structcorr), garden_slopes in unwrap_dictionary(slope_dict[ind_or_pooled]):

            # get slopes and fill in the heatmap
            heatmap = mvp03.blank_dataframe()
            for garden, slope in garden_slopes.items():
                x, y = locations.loc[garden]
                heatmap.loc[y, x] = slope
            heatmaps[ind_or_pooled][marker_set][which_traits][structcorr] = heatmap.copy()

            # determine min and max
            hmin = heatmap.min().min()
            hmax = heatmap.max().max()
            if hmin < vmin:
                vmin = hmin
            if hmax > vmax:
                vmax = hmax

        # fill in subplots with histograms
        fig, axes = fill_slope_heatmaps(marker_sets, heatmaps[ind_or_pooled], vmin=vmin, vmax=vmax, total_traits=total_traits)

        decorate_figure(marker_sets, fig, axes, cmap='viridis', vmin=vmin, vmax=vmax,
                        cbar_label='slope of fitness ~ RDA offset',
                        title=f'{performance_name} slope\n{seed = }\n{ind_or_pooled = }\n{level}\n')

        # save figure
        pdf.savefig(bbox_inches="tight")
        plt.show()
        plt.close()

    # save objects
    pkl = op.join(heat_dir, f'{seed}_{performance_name}_slope_heatmaps.pkl')
    pkldump(heatmaps, pkl)
    print(f'\n\twrote heatmaps to : {pkl}')
    
    pass


def scatter_wrapper(offset_dfs, fitness_mat, envdata, locations, samppop, popsamps, pdf, total_traits=None):
    """Wrapper for `mvp06.performance_scatter`."""
    print(ColorText('\nCreating scatter plots ...').bold().custom('gold'))
    for (ind_or_pooled, marker_set, which_traits, structcorr), offset in unwrap_dictionary(offset_dfs):
        fitness = fitness_mat[ind_or_pooled].copy()
        
        desc = f'{ind_or_pooled} {marker_set} {which_traits} {structcorr}'
        for home_env in pbar(['sal_opt', 'temp_opt'], desc=desc):  # environment to color pops

            # color for the environment (temp_opt) of source_pop
            colormap = 'Reds' if home_env=='temp_opt' else 'Blues_r'
            cmap = plt.cm.get_cmap(colormap)
            
            # determine colors for scatter plot
            if ind_or_pooled == 'ind':
                indcolors = fitness.columns.map(samppop).map(
                    envdata[home_env]).to_series(index=fitness.columns.astype(int)).apply(mvp06.color,
                                                                                   cmap=cmap,
                                                                                   norm=norm).to_dict()
            gardencolors = fitness.index.map(
                envdata[home_env]).to_series(index=fitness.index).apply(mvp06.color,
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
                                          f'{ind_or_pooled} {label_dict[marker_set]} {which_traits}',
                                          locations,
                                          colors,
                                          pdf,
                                          popsamps=popsamps, norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir,
                                          home_env=home_env,
                                          program='RDA',
                                          garden_or_source=garden_or_source,
                                          ind_or_pooled=ind_or_pooled)

    pass


def fig_wrapper(performance_dicts, samppop, popsamps, locations, offset_dfs, fitness_mat, envdata, fig_dir=None):
    """Wrapper for create_histo_subplots(), create_heatmap_subplots."""
    # how many envs were selective?
    total_traits = ['ntraits-1', 'ntraits-2'] if ntraits == 1 else ['ntraits-2']
    
    saveloc = op.join(fig_dir, f'{seed}_RDA_figures.pdf')
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
            create_slope_heatmap_subplots(
                performance_name, performance_dicts[f'{slope_group}_slopes'].copy(), locations, pdf,
                total_traits=total_traits
            )
            
    print(ColorText(f'\nsaved fig to: {saveloc}').bold())
    
    # save scatterplots separately so computers don't get slow trying to display everything
    saveloc = op.join(fig_dir, f'{seed}_RDA_figures_scatter.pdf')
    with PdfPages(saveloc) as pdf:  # save all figures to one pdf
        scatter_wrapper(offset_dfs, fitness_mat, envdata, locations, samppop, popsamps, pdf, total_traits=total_traits)
        
    print(ColorText(f'\nsaved scatter fig to: {saveloc}').bold())
        
    pass


def main():
    # get predicted offset files
    offset_dfs = retrieve_offset_data(seed)
    
    # get a list of subsampled individuals, map samp top subpopID, and get population locations for each subpopID
    subset, locations, envdata = mvp06.get_pop_data(slimdir, seed)

    # map subpopID to list of samps - key = subpopID val = list of individual sample names
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()

    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))
    
    # get fitness matrices for individuals and pops (pops are mean fitness)
    fitness_mat = retrieve_fitness_data(slimdir, seed, subset)
    
    # calculate validation scores
    performance_dicts = calculate_performance(offset_dfs, fitness_mat, popsamps, samppop)
    
    # create figs
    fig_wrapper(performance_dicts, samppop, popsamps, locations, offset_dfs, fitness_mat, envdata, fig_dir=fig_dir)
    
    # DONE!
    print(ColorText('\nDONE!!').bold().green())
    print(ColorText(f'\ttime to complete: {formatclock(dt.now() - t1, exact=True)}\n'))
    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, outerdir = sys.argv
    
    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))
    
    # set up timer
    t1 = dt.now()
    
    # details about demography and selection
    params = mvp10.read_params_file(slimdir)
    ntraits, level = params.loc[seed, ['N_traits', 'level']]
    
    # set globally
    norm = Normalize(vmin=-1.0, vmax=1.0)
    rda_dir = op.join(outerdir, 'rda')
    rda_outdir = op.join(rda_dir, 'offset_outfiles')
    fig_dir = makedir(op.join(rda_dir, 'validation/figs'))
    heat_dir = makedir(op.join(rda_dir, 'validation/heatmap_textfiles'))
    pkl_dir = makedir(op.join(rda_dir, 'validation/pkl_files'))
    corr_dir = makedir(op.join(rda_dir, 'validation/corrs'))
    offset_dir = makedir(op.join(rda_dir, 'validation/offset_dfs'))
    
    # dict for pretty labels in figures
    label_dict = {
        'TRUE' : 'RDA outliers',
        'FALSE' : 'all loci',
        'CAUSAL' : 'causal loci',
        'NEUTRAL' : 'neutral loci',
        'nocorr' : 'no correction',
        'structcorr' : 'structure-corrected',
        'sal_opt' : 'sal',
        'temp_opt' : 'temp'
    }

    # background color for figures
    background_cmap = create_cmap(['white', 'gold'], grain=1000)
    
    # add to namespaces
    mvp06.label_dict = label_dict
    mvp06.level = level
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    main()
