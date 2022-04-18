"""
Dependencies
------------
- dependent upon completion of MVP_01_train_gradient_forests.py
- dependent upon completion of MVP_02_fit_gradient_forests.py
- dependent upon code from github.com/brandonlind/pythonimports
"""
from pythonimports import *
from myfigs import save_pdf
import MVP_05_validate_RONA as mvp5

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
    fitness.columns = ('i' + fitness.columns.astype(str)).tolist()
    # reduce to only the subsampled individuals
    fitness = fitness[subset.index]
    
    return fitness


def get_offset_predictions(fitness_mat):
    """Get offset predictions output by MVP_02_fit_gradient_forests.py."""
    print(ColorText('\nRetrieving predicted offsets ...').bold().custom('gold'))
    # get the predicted offset from files output from fitting created in ../02_fit_gradient_forests.ipynb
    files = fs(fitting_dir, 'offset')

    # make sure just as many RDS files were created from fitting script (ie that all fitting finished)
    rdsfiles = fs(fitting_dir, endswith='.RDS')
    assert len(files) == len(rdsfiles)

    outfiles = wrap_defaultdict(dict, 3)
    for outfile in files:
        seed, ind_or_pooled, adaptive_or_all, garden_ID, *suffix = op.basename(outfile).split("_")
        outfiles[ind_or_pooled][adaptive_or_all][int(garden_ID)] = outfile

    # gather the predicted offset values for each individual in each garden
    offset_series = wrap_defaultdict(list, 2)  # for gathering in next loop
    for (ind_or_pooled, adaptive_or_all, garden_ID), outfile in unwrap_dictionary(outfiles):
        # get the appropriate fitness matrix
        fitness = fitness_mat[ind_or_pooled].copy()
        
        # read in offset projections
        offset = pd.read_table(outfile, index_col=0)
        offset_series[ind_or_pooled][adaptive_or_all].append(
            pd.Series(offset['offset'],
                    name=garden_ID)
        )

    # collapse the predicted offset values for each individual in each garden into one data frame
        # - use for correlation calcs in next cell
    offset_dfs = wrap_defaultdict(None, 2)
    for (ind_or_pooled, adaptive_or_all), series_list in unwrap_dictionary(offset_series):
        # collapse all of the offset values from each garden into a single dataframe
        df = pd.concat(series_list,
                    axis=1,
                    keys=[series.name for series in series_list])
        
        # sort by garden_ID, transpose to conform to convention of `fitness_mat`
        offset_dfs[ind_or_pooled][adaptive_or_all] = df[sorted(df.columns)].T
        
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


def create_heatmap(corrs, ind_or_pooled, adaptive_or_all, title, locations, samppop, performance='garden', save=True):
    """Create a heatmap of the landscape performance - npop by npop.
    
    Notes
    -----
    - if individual data is input, data will be average per population
    
    """

    if ind_or_pooled == 'ind' and performance != 'garden':
        # average across individuals for each population
        ind_corrs = pd.DataFrame(corrs, columns=['performance'])
        ind_corrs['subpopID'] = ind_corrs.index.map(samppop)
        corrs = ind_corrs.groupby('subpopID')['performance'].apply(np.mean)
        
    # fill in heatmap
    df = blank_dataframe()
    for garden,corr in corrs.items():
        x, y = locations.loc[garden]
        df.loc[y, x] = corr

    # plot heatmap
    _ = sns.heatmap(df,
                    cmap='viridis',
                    cbar_kws={'label': "Spearman's $\\rho^2$"})
    plt.title(title)
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')

    if save is True:
        save_pdf(
            op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_GF_{performance}_performance_heatmap.pdf')
        )
    
    plt.show()
    plt.close()
    
    pass


def print_dataset(dataset, length=46):
    """Print three lines, first and third are repetead ticks, second is centered dataset name."""
    ticks = '-' * length
    
    nchars = len(dataset)  # add 1 for space
    before = math.ceil((length - nchars) / 2)
    after = length - before - nchars
    
    # first divider
    print(
        ColorText(ticks).bold().custom('red')
    )
    # dataset
    print(
        ColorText(' ' * before + \
                  dataset + \
                  ' ' * after).bold().custom('red')
    )
    # second divider
    print(
        ColorText(ticks).bold().custom('red')
    )
    
    pass


def sample_performance(offset, fitness, ind_or_pooled, adaptive_or_all, locations):
    # set up figure
    figpos, fig, axes = mvp5.fig_setup(locations)
    
    # color for the environment (temp_opt) of source_pop - TODO: infer selected envs from data
    colormap = 'Reds' if env=='temp_opt' else 'Blues_r'
    cmap = plt.cm.get_cmap(colormap)
    if ind_or_pooled == 'ind':
        cols = offset.columns.map(samppop).map(envdata[env]).to_series().apply(mvp5.color, cmap=cmap, norm=norm)
    else:
        cols = offset.columns.map(envdata[env]).to_series().apply(mvp5.color, cmap=cmap, norm=norm)
        
    # create each of the population figures in the order matplotlib puts them into the figure
    for subplot,ax in enumerate(axes.flat):
        sample = figpos[subplot]  # which pop now?
        ax.scatter(offset[sample],
                   fitness[sample],
                   c=cols)
        # decide if I need to label longitude (x) or latitude (y) axes
        x,y = locations.loc[samppop[sample]]
        if subplot in range(0, 110, 10):
            ax.set_ylabel(int(y))
        if subplot in range(90, 101, 1):
            ax.set_xlabel(int(x))
    
    pass


def source_performance_slope_heatmap_and_boxplot(offset, fitness, ind_or_pooled, adaptive_or_all, popsamps, locations):
    """Create a heatmap for each source population that displays average slope of regression: fitness ~ offset."""
    if ind_or_pooled == 'ind':
        # 1. HEATMAP
        # 1.1 get abs slopes
        all_slopes = defaultdict(list)  # for boxplot
        average_slopes = {}  # for heatmap
        for source_pop in fitness.index:
            samps = popsamps[source_pop]

            for samp in samps:
                all_slopes[source_pop].append(
                    abs(
                        linregress(offset[samp], fitness[samp]).slope
                    )
                )
            average_slopes[source_pop] = np.mean(all_slopes[source_pop])
            
        # 1.2 create heatmap
        heatmap = blank_dataframe()
        for source_pop,slope in average_slopes.items():
            x,y = locations.loc[source_pop]
            heatmap.loc[y, x] = slope
        plt.close()
        _ = sns.heatmap(heatmap,
                        cmap='viridis',
                        cbar_kws={'label': "abs slope of fitness ~ offset"})
        plt.title(f'average slope per source pop for {ind_or_pooled} {adaptive_or_all}')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')

        save_pdf(op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_source_slope_heatmap.pdf'))
        plt.show()
        plt.close()
        
        # 2. boxplot
        # 2.1 create fig
        figpos, fig, axes = mvp5.fig_setup(locations)
        for subplot,ax in enumerate(axes.flat):
            source_pop = figpos[subplot]
            ax.boxplot(all_slopes[source_pop])
            ax.set_xlabel(None)
            # decide if I need to label longitude (x) or latitude (y) axes
            x,y = locations.loc[source_pop]  
            if subplot in range(0, 110, 10):
                ax.set_ylabel(int(y))
            if subplot in range(90, 101, 1):
                ax.set_xlabel(int(x))
        # 2.2 decorate fig
        fig.supylabel('fitness')
        fig.supxlabel('predicted offset')
        fig.suptitle(f'{seed}\nGF {ind_or_pooled} source performance distribution')

        save_pdf(
            op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_GF_source_performance_boxplot.pdf')
        )

        plt.show()
        plt.close()
        
    else:  # pooled
        # 1. calculate slopes
        slopes = {}
        for source_pop in fitness.index:
            slopes[source_pop] = linregress(offset[source_pop], fitness[source_pop]).slope
        # 2. fill heatmap
        heatmap = blank_dataframe()
        for source_pop,slope in slopes.items():
            x,y = locations.loc[source_pop]
            heatmap.loc[y, x] = slope
        # 3. show heatmap
        _ = sns.heatmap(heatmap.abs(),
                        cmap='viridis',
                        cbar_kws={'label': "abs slope of fitness ~ offset"})
        plt.title(f'slope per source pop for {ind_or_pooled} {adaptive_or_all}')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')

        save_pdf(op.join(fig_dir, f'{seed}_source_slope_heatmap-{ind_or_pooled}_{adaptive_or_all}.pdf'))
        plt.show()
        plt.close()
        
    pass


def garden_performance_slope_heatmap(offset, fitness, ind_or_pooled, adaptive_or_all, locations):
    """Create a heatmap for each common garden that displays slope of regression: fitness ~ offset."""
    # get slopes and fill in the heatmap
    heatmap = blank_dataframe()
    for garden in fitness.index:
        x,y = locations.loc[garden]
        heatmap.loc[y, x] = linregress(offset.loc[garden], fitness.loc[garden]).slope

    # plot the heatmap
    _ = sns.heatmap(heatmap.abs(),
                    cmap='viridis',
                    cbar_kws={'label': "abs slope of fitness ~ GF offset"})
    plt.title(f'slope in garden for {ind_or_pooled} {adaptive_or_all}')
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')

    print('\tgarden_performance_slope_heatmap()')
    save_pdf(op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_GF_garden_slope_heatmap.pdf'))

    plt.show()
    plt.close()

    pass


def fig_wrapper(offset_dfs, fitness_mat, locations, samppop, envdata, popsamps):
    """Create a bunch o' figs."""
    print(ColorText('\nCreating figs ...').bold().custom('gold'))
    # performance (spearman's rho) and slopes of relationship from fitness ~ GF offset
    garden_performance = wrap_defaultdict(None, 2)
    garden_slopes = wrap_defaultdict(None, 2)
    individual_performance = wrap_defaultdict(None, 2)
    individual_slopes = wrap_defaultdict(None, 2)
    for (ind_or_pooled, adaptive_or_all), offset in unwrap_dictionary(offset_dfs):
        # 1 - GET DATA
        # 1.1 divide figures by dataset when printing
        print_dataset(f'{ind_or_pooled} {adaptive_or_all}')
        
        # 1.2 get the appropriate fitness matrix
        fitness = fitness_mat[ind_or_pooled].copy()
        
        
        # 2 - GARDEN PERFORMANCE - how well offset was predicted at the common garden location across samples
        # 2.1 squared spearman's correlation coefficient (val) for each garden (key)
        garden_performance[ind_or_pooled][adaptive_or_all] = offset.corrwith(fitness,
                                                                             axis='columns',  # across columns for each row
                                                                             method='spearman') ** 2
        # 2.2 plot histogram
        garden_performance[ind_or_pooled][adaptive_or_all].hist()
        title = f'{seed}\ngarden performance\ndata={ind_or_pooled} loci={adaptive_or_all}'
        plt.title(title)
        plt.ylabel('count')
        plt.xlabel("Spearman's $\\rho^2$")
        save_pdf(
            op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_GF_garden_performance_histogram.pdf')
        )
        plt.show()
        plt.close()
        
        # 2.3 create heatmap
        create_heatmap(garden_performance[ind_or_pooled][adaptive_or_all],
                       ind_or_pooled,
                       adaptive_or_all,
                       title,
                       locations,
                       samppop)
        
        # 2.4 calculate and plot slope of relationship between fitness ~ offset at each garden
        # color for the environment (temp_opt, sal_opt) of source_pop - TODO: infer selected envs from data
        for env,env_series in envdata.items():
            colormap = 'Reds' if env=='temp_opt' else 'Blues_r'
            cmap = plt.cm.get_cmap(colormap)
            if ind_or_pooled == 'ind':
                cols = offset.columns.map(samppop).map(env_series).to_series().apply(mvp5.color, cmap=cmap, norm=norm)
            else:
                cols = offset.columns.map(env_series).to_series().apply(mvp5.color, cmap=cmap, norm=norm)
            mvp5.garden_performance_scatter(offset, fitness, f'{ind_or_pooled}_{adaptive_or_all}_{env}', locations, env_series, cols,
                                            norm=norm, cmap=cmap, seed=seed, fig_dir=fig_dir,
                                            program=f'GF')
            plt.close()  # needed so that garden_performance_slope_heatmap doesn't create fig on top of this
        
        # 2.5 calculate the slope of the linear model between fitness ~ offset at each garden
        garden_performance_slope_heatmap(offset, fitness, ind_or_pooled, adaptive_or_all, locations); plt.close()
        
        # 3 - SAMPLE-LEVEL PERFORMACE - how well performace was predicted for the sample across gardens
        # 3.1 squared spearman's correlation coefficient for each individual or pool
        individual_performance[ind_or_pooled][adaptive_or_all] = offset.corrwith(fitness,                                                                                axis='index', # across rows for each column
                                                                                 method='spearman') ** 2
        
        # 3.2 plot histogram
        individual_performance[ind_or_pooled][adaptive_or_all].hist()
        name = 'source pool' if ind_or_pooled=='pooled' else 'individual'
        title = f'{seed}\n{name} performance\ndata={ind_or_pooled} loci={adaptive_or_all}'
        plt.title(title)
        save_pdf(
            op.join(fig_dir, f'{seed}_{ind_or_pooled}_{adaptive_or_all}_{name}_GF_performance_histogram.pdf')
        )
        plt.show()
        plt.close()

        # 3.3 create heatmap
        pkldump(
            individual_performance[ind_or_pooled][adaptive_or_all],
            op.join(validation_dir, 
                    f'{seed}_{ind_or_pooled}_{adaptive_or_all}_corrs.pkl')
        )
#         pkldump(samppop, '/work/lotterhos/MVP-Offsets/mypractice_20210308/gradient_forests/samppop.pkl')
        create_heatmap(individual_performance[ind_or_pooled][adaptive_or_all],
                       ind_or_pooled,
                       adaptive_or_all,
                       title.replace('individual', 'average individual'),  # replace only happens when 'individual' is in title
                       locations,
                       samppop,
                       performance='individual')
        
        # 3.4 calculate and plot slope of relationship between fitness ~ offset for each ind/pool across gardens
        source_performance_slope_heatmap_and_boxplot(offset, fitness, ind_or_pooled, adaptive_or_all, popsamps, locations)


def main():
    # get a list of subsampled individuals, map samp top subpopID, and get population locations for each subpopID
    subset, locations, envdata = mvp5.get_pop_data(slimdir, seed)

    # map subpopID to list of samps - key = subpopID val = list of individual sample names
    popsamps = subset.groupby('subpopID')['sample_name'].apply(list).to_dict()

    # map samp to subpopID
    samppop = dict(zip(subset.index, subset['subpopID']))
    
    # get fitness matrices for individuals and pops (pops are mean fitness)
    fitness_mat = {'ind': load_ind_fitness_matrix(slimdir, seed, subset),
                   'pooled': mvp5.load_pooled_fitness_matrix(slimdir, seed)}
    
    # get predicted ofset
    offset_dfs = get_offset_predictions(fitness_mat)
    
    fig_wrapper(offset_dfs, fitness_mat, locations, samppop, envdata, popsamps)
    
    # DONE!
    print(ColorText('\nDONE!!').bold().green())
    print(ColorText(f'\ttime to complete: {formatclock(dt.now() - t1, exact=True)}\n'))
    pass


if __name__ == '__main__':
    # get input args
    thisfile, seed, slimdir, gf_parentdir = sys.argv

    print(ColorText(f'\nStarting {op.basename(thisfile)} ...').bold().custom('gold'))

    # set up timer
    t1 = dt.now()
    
    # print versions of packages and environment
    print(ColorText('\nEnvironment info :').bold().custom('gold'))
    latest_commit()
    session_info.show(html=False, dependencies=True)
    
    # get dirs
    fitting_dir = op.join(gf_parentdir, 'fitting/fitting_outfiles')
    training_outdir = op.join(gf_parentdir, 'training/training_outfiles')
    fig_dir = makedir(op.join(gf_parentdir, 'validation/figs'))
    validation_dir = op.dirname(fig_dir)
    
    # set global variables needed for `mvp5.color` and `mvp5.garden_performance_scatter`
    norm=Normalize(vmin=-1.0, vmax=1.0)
    
    main()
