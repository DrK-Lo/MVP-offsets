"""Validate RONA with mean individual fitness per pop from the simulation data.

Validation is done by correlating population mean fitness with RONA (for each env).

TODO
----
save heatmaps as csv

Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon completion of MVP_04_train_RONA.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from myfigs import save_pdf

import MVP_01_train_gradient_forests as mvp01

import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress


def load_pooled_fitness_matrix(slimdir, seed):
    """Load fitness matrix output from simulations."""
    # an n_deme x n_deme table that indicates the mean fitness of individuals 
        # from the source deme (in columns) in the transplant deme (in rows) 
        
    fitness = pd.read_table(op.join(slimdir, f'{seed}_fitnessmat.txt'),
                            delim_whitespace=True,
                            header=None)
    
    assert fitness.shape == (100, 100)

    # set column names for popID
    fitness.columns = range(1, 101, 1)
    fitness.index = range(1, 101, 1)

    return fitness


def color(val, cmap=None, norm=None):
    """Return rgb on colormap `cmap`."""
    return cmap(norm(val))[:3]


def fig_setup(locations):
    """Get figure position (order) of each population on a 10x10 subplot."""
    count = 0
    figpos = {}
    for y in reversed(range(1,11,1)):
        for x in range(1,11,1):
            pop = locations[(locations['lon']==x) & (locations['lat']==y)].index[0]
            figpos[count] = pop
            count += 1
            
    # set up big fig
    fig, axes = plt.subplots(10, 10,
                             sharex='all',
                             sharey='all',
                             figsize=(15, 10))
    return figpos, fig, axes


def garden_performance_scatter(
    offset, fitness, env, locations, envdata, cols, cmap=None, norm=None, seed=None, fig_dir=None, program='RONA'
):
    """Create a map of pops using coords, show relationsip between RONA and fitness."""
    figpos, fig, axes = fig_setup(locations)
    
    # create each of the population subfigures in the order matplotlib puts them into the figure
    for subplot,ax in enumerate(axes.flat):
        garden = figpos[subplot]  # which garden now?
        ax.scatter(offset.loc[garden],
                   fitness.loc[garden],
                   c=cols)
        # decide if I need to label longitude (x) or latitude (y) axes
        x,y = locations.loc[garden]  
        if subplot in range(0, 110, 10):
            ax.set_ylabel(int(y))
        if subplot in range(90, 101, 1):
            ax.set_xlabel(int(x))
            
    # set colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:,:])
    cbar.ax.set_title(env)
    
    fig.supylabel('fitness')
    fig.supxlabel('predicted offset')
    fig.suptitle(f'{seed}\n{program} garden performance for {env}\ntransplanted pops colored by home environment')
#     print('\tgarden_performance_scatter()')
    if fig_dir is not None:
        save_pdf(
            op.join(fig_dir, f'{seed}_{env}_{program}_garden_performance_scatter.pdf')
        )
    
    plt.show()
    plt.close()
    
    del figpos, fig, axes
    
    pass


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


def create_scatter_map(rona, fitness, locations, envdata, marker_set):
    """Wrapper for garden_performance_scatter()."""
    print(ColorText("\nCreating scatter maps and calculating performance ...").bold().custom('gold'))
    garden_performance = defaultdict(dict)  # performance within gardens across transplant pops
    source_performance = defaultdict(dict)  # performance for transplant pops across gardens

    garden_slopes = defaultdict(dict)  # slope of relationship between fitness and RONA
    source_slopes = defaultdict(dict)  # slope of relationship between fitness and RONA

    for env,rona_dict in rona.items():
        # format rona predictions in the same format as fitness dataframe
            # pd.DataFrame(rona[env]).T is necessary so that source deme is col and garden is row
        rona_offset = pd.DataFrame(rona_dict).T
        # color for the environment (temp_opt) of source_pop - TODO: infer selected envs from data
        colormap = 'Reds' if env=='temp_opt' else 'Blues_r'
        cmap = plt.cm.get_cmap(colormap)
        cols = rona_offset.columns.map(envdata[env]).to_series().apply(color, cmap=cmap, norm=norm).tolist()
        garden_performance_scatter(
            rona_offset, fitness, f'{marker_set}_{env}', locations, envdata, cols,
            norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir
        )
        
        
        # garden performance
        for garden in pbar(fitness.index, desc=f'{env} garden performance'):
            # retrieve rona for this garden
            ronadata = pd.Series(rona_dict[garden], dtype=float)
            
            # record squared spearman's rho
            garden_performance[env][garden] = ronadata.corr(fitness.loc[garden],
                                                            method='spearman')
            # record slope
            garden_slopes[env][garden] = linregress(ronadata, fitness.loc[garden]).slope


        # population performance
        for source_pop in pbar(fitness.columns, desc=f'{env} population performance'):
            # retrieve rona for this pop
            ronadata = pd.Series(dtype=float)
            for garden in rona_dict:
                ronadata.loc[garden] = rona_dict[garden][source_pop]
            
            # record spearman's rho
            source_performance[env][source_pop] = ronadata.corr(fitness[source_pop],
                                                                method='spearman')
            
            # record slope
            source_slopes[env][source_pop] = linregress(ronadata, fitness[source_pop]).slope

    return garden_performance, source_performance, garden_slopes, source_slopes


def blank_dataframe():
    """Create a defaultdict with default being a blankblank dataframe (landscape map) filled with NaN,
    columns and index are subpopIDs.
    
    Notes
    -----
    instantiating with dtype=float is necessary for sns.heatmap (otherwise sns.heatmap(df.astype(float)))
    """
    df_dict = defaultdict(lambda: pd.DataFrame(columns=range(1, 11, 1),
                                               index=reversed(range(1,11,1)),  # so that x=1,y=10 is in top left
                                               dtype=float))
    return df_dict


def fill_heatmaps(garden_performance, source_performance, garden_slopes, source_slopes, locations, marker_set):
    """Create dataframe maps to hold information about each garden or source pop using pop coordinates."""
    print(ColorText(f'\nCreating heatmaps for {marker_set} loci ...').bold().custom('gold'))
    # create empty dataframes to fill in as heatmaps
    garden_heat = blank_dataframe()
    garden_slope_heat = blank_dataframe()
    source_heat = blank_dataframe()
    source_slope_heat = blank_dataframe()

    # fill out heat maps
    print('\tfilling out heatmaps ...')
    for env in source_performance.keys():
        for source_pop,performance in source_performance[env].items():
            x,y = locations.loc[source_pop]
            
            # fill out source_heat
            source_heat[env].loc[y, x] = performance
            
            # fill out garden_heat
            garden_heat[env].loc[y, x] = garden_performance[env][source_pop]
            
            # fill out source_slope
            source_slope_heat[env].loc[y, x] = source_slopes[env][source_pop]
            
            # fill out garden slope
            garden_slope_heat[env].loc[y, x] = garden_slopes[env][source_pop]

    # create figs that show the performance across pops for each garden
    print('\tsaving heatmaps for performance of transplanted pops within garden ...')
    for env,heatmap in garden_heat.items():
        _ = sns.heatmap(heatmap,
                        cmap='viridis',
                        cbar_kws={'label': "Spearman's $\\rho$"})
        
        plt.title(f'performance in garden for {env = }')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        
        save_pdf(op.join(fig_dir, f'{seed}_{marker_set}_garden_performance_heatmap-{env}.pdf'))
        
        plt.show()

    # create figs that show the performance across gardens for each source
    print('\n\tsaving heatmaps for the performance across gardens for each source ...')
    for env,heatmap in source_heat.items():
        _ = sns.heatmap(heatmap,
                        cmap='viridis',
                        cbar_kws={'label': "Spearman's $\\rho$"})
        
        plt.title(f'performance for each source across gardens\nfor {env = }')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        
        save_pdf(op.join(fig_dir, f'{seed}_{marker_set}_source_performance_heatmap-{env}.pdf'))
        
        plt.show()

    # create figs for slope of relationship between fitness ~ RONA at each garden for each env
    print(
        '\n\tsaving heatmaps for the slope of relationship between fitness ~ RONA at each garden for each env ...'
    )
    for env,heatmap in garden_slope_heat.items():
        _ = sns.heatmap(heatmap,
                        cmap='viridis',
                        cbar_kws={'label': "slope of fitness ~ RONA"})
        plt.title(f'slope in garden for {env = }')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        
        save_pdf(op.join(fig_dir, f'{seed}_{marker_set}_garden_slope_heatmap-{env}.pdf'))
        
        plt.show()

    # create figs for slope of performance for each source pop across gardens
    print(
        '\n\tsaving heatmaps for the slope of relationship between fitness ~ RONA for each source pop across gardens for each env ...'
    )
    for env,heatmap in source_slope_heat.items():
        _ = sns.heatmap(heatmap,
                        cmap='viridis',
                        cbar_kws={'label': "slope of fitness ~ RONA"})
        plt.title(f'slope across garden for {env = }')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        
        save_pdf(op.join(fig_dir, f'{seed}_{marker_set}_source_slope_heatmap-{env}.pdf'))
        
        plt.show()

    return garden_heat, source_heat, garden_slope_heat, source_slope_heat

def save_objects(garden_heat, source_heat, garden_slope_heat, source_slope_heat, marker_set):
    """Save heatmap objects."""
    print(ColorText(f'\nSaving heatmap objects for {marker_set} loci ...').bold().custom('gold'))
    for env,heatmap in source_heat.items():
        heatfile = op.join(heat_dir, f'{seed}_{marker_set}_source_performance_heatmap-{env}.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True)
        print('\t', heatfile)
        
    print(' ')
        
    for env,heatmap in garden_heat.items():
        heatfile = op.join(heat_dir, f'{seed}_{marker_set}_garden_performance_heatmap-{env}.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True)
        print('\t', heatfile)
        
    print(' ')
        
    for env,heatmap in source_slope_heat.items():
        heatfile = op.join(heat_dir, f'{seed}_{marker_set}_source_slope_heatmap-{env}.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True)
        print('\t', heatfile)
        
    print(' ')
        
    for env,heatmap in garden_slope_heat.items():
        heatfile = op.join(heat_dir, f'{seed}_{marker_set}_garden_slope_heatmap-{env}.txt')
        heatmap.to_csv(heatfile, sep='\t', index=True)
        print('\t', heatfile)
    
    pass


def main():
    # get pop data
    print(ColorText('\nGetting population information ...').bold().custom('gold'))
    
    fitness = load_pooled_fitness_matrix(slimdir, seed)
    subset, locations, envdata = get_pop_data(slimdir, seed)
    
    for marker_set in ['all', 'adaptive']:
        print(ColorText(f'\nValidating RONA using {marker_set} loci ...').bold().custom('gold'))
        
        # load RONA estimates
        rona = pklload(op.join(rona_outdir, f'{seed}_{marker_set}_RONA_results.pkl'))

        # plot relationship between fitness and RONA for each subpopID on a map of subpops
        data = create_scatter_map(rona, fitness, locations, envdata, marker_set)

        # create heatmaps for performance and slope of relationship between fitness and RONA
        heatmap_objects = fill_heatmaps(*data, locations, marker_set)

        # save the objects in case we ever want them
        save_objects(*heatmap_objects, marker_set)

    # done
    print(ColorText(f'\ntime to complete: {formatclock(dt.now() - t1, exact=True)}'))
    print(ColorText('\nDONE!!').bold().green(), '\n')

    pass


if __name__ == '__main__':
    # get input arguments
    thisfile, seed, slimdir, rona_outdir = sys.argv

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()

    # create dirs
    rona_dir = op.dirname(op.dirname(rona_outdir))
    fig_dir = makedir(op.join(rona_dir, 'validation/figs'))
    heat_dir = makedir(op.join(op.dirname(fig_dir), 'heatmap_objects'))

    # set globally
    norm = Normalize(vmin=-1.0, vmax=1.0)

    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)

    main()
