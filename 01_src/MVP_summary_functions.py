"""Functions used to create figs across scripts that summarize output of MVP offset validation pipeline."""
from pythonimports import *
from myfigs import adjust_box_widths, save_pdf
import builtins

# DICTIONARIES
boxplot_kwargs = {  # kwargs for seaborn.catplot (boxplot) properties
    'palette': {'no pleiotropy': '#3e4a89',
                'pleiotropy': '#35b779',
                'SS-Clines': '#440c54',
                'SS-Mtn': '#22908c',
                'Est-Clines': '#fde624',
                'edgecolor': '#bebebe',
                'equal-S': 'k',
                'unequal-S': 'darkgray',
                'N-equal': '#0b559f',  # popsizes are sns.color_palette('Blues_r') using matplotlib.colors.rgb2hex
                'N-variable': '#2b7bba',
                'N-cline-center-to-edge': '#539ecd',
                'N-cline-N-to-S': '#89bedc',
                'm-constant': '#572c92',       # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[0])
                'm-variable': '#705eaa',       # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[1])
                'm-breaks': '#8d89c0',         # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[2])
                'all causal': '#05712f',       # matplotlib.colors.rgb2hex(sns.color_palette('Greens_r')[0])
                'one noncausal': '#56b567',    # matplotlib.colors.rgb2hex(sns.color_palette('Greens_r')[2])
                'no noncausal': '#bce4b5',     # matplotlib.colors.rgb2hex(sns.color_palette('Greens_r')[4])
                'oligogenic': '#aa1016',       # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[0])
                'mod-polygenic': '#f44f39',    # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[2])
                'highly-polygenic': '#fcaf93',  # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[4])
                'RONA': 'tab:blue',
                'RONA-sal_opt': 'tab:blue',
                'RONA-temp_opt': 'tab:orange',
                'lfmm2': 'tab:green',
                'GF': 'tab:red',
                'rda': 'tab:purple'
               },
    
    'whiskerprops': {'color': '#bebebe'},
    
    'medianprops': {'color': '#bebebe',
                    'alpha': 1.0},
    
    'boxprops': {'edgecolor': '#bebebe'},
    
    'flierprops': {'color': '#bebebe',
                   'markeredgecolor': '#bebebe',
                   'markerfacecolor': '#bebebe'},
    
    'capprops': {'color': '#bebebe'}
}


hue_orders = {
    'landscape' : ['SS-Clines', 'SS-Mtn', 'Est-Clines'],
    'glevel' : ['oligogenic', 'mod-polygenic', 'highly-polygenic'],
    'pleio' : ['no pleiotropy', 'pleiotropy'],
    'slevel' : ['equal-S', 'unequal-S'],
    'popsize' : ['N-equal', 'N-variable', 'N-cline-center-to-edge', 'N-cline-N-to-S'],
    'migration' : ['m-constant', 'm-variable', 'm-breaks'],
}


factor_names = {  # used in eg anova models to convert column name to human-readable label for figure
    'architecture': 'architecture',
    'demography': 'demography',
    'glevel' : 'architecture',
    'landscape' : 'landscape',
    'slevel' : 'slevel',
    'popsize_migration' : 'demography',
    'plevel_pleio' : 'pleiotropy',
    'cor_TPR_temp' : "$\^p_{clinal, temp}$",
    'cor_TPR_sal' : "$\^p_{clinal, Env2}$",
    'final_LA' : 'Degree of\nlocal adaptation',
    'Residual' : 'Residual',
    'C(garden)' : 'Garden ID',
    'final_LA:architecture' : 'final_LA:architecture',
    'final_LA:landscape' : 'final_LA:landscape',
    'final_LA:C(garden)' : 'final_LA:C(garden)',
    'final_LA:slevel' : 'final_LA:slevel'
}



# FUNCTIONS
def display_level_dfs(level_scores=None):
    """Display first 5 lines of dataframes created in 02.01.00_save_level_scores.ipynb."""
    for program in keys(level_scores):
        print(program)
        display(level_scores[program]['garden_performance'].head())
    pass


def pretty_facetgrid(g, program, num_seeds, saveloc, add_title, center=0.47):
    """Make the FacetGrid look nice.
    
    Parameters
    ----------
    g - seaborn.axisgrid.FacetGrid (e.g. as returned from seaborn.catplot)
    program - str
        the offset method name
    num_seeds - int
        the unique number of seeds used to create the fig represented by *g*
    saveloc - str
        path for where to save
    add_title - str
        string to add to title of figure
    center - float
        where to center title and 'Marker Set' x-axis label
    """
    # line at tau = 0
    for ax in g.axes[0]:
        ax.axhline(0, linestyle='--', color='gainsboro', linewidth=0.5, zorder=0)

    # make it pretty
    g.set_titles('{col_name}')
    g.set_axis_labels("", "Validation Score (Kendall's $\\tau$)", fontsize=12)
    adjust_box_widths(list(g.axes[0]), 0.85)
    g.set(ylim=(-1, 0.6))
    
    # add labels
    title = f'{program} (num seeds = {num_seeds}){add_title}'
    g.fig.suptitle(title, x=center)
    g.fig.text(y=-0.05, x=center, s='Marker Set', fontsize=12, ha='center')
    
    g.tight_layout()
    
    # save
    save_pdf(saveloc)

    plt.show()

    pass


def run_facetgrid_figs(fxn, function_name, level_scores=None, savedir=None):
    """Apply function *fxn* to see how parameters affect validation scores from 2-trait sims.
    
    See notebooks in 02.01_summary_figs folder.

    Parameters
    ----------
    fxn
        an executable function that takes args=(program, data) and kwargs={hue, hue_order, add_title, filename}
            (see generic functions in eg 01_glevel_vs_other.ipynb or 02_landscape_vs_other.ipynb)
    function_name - str
        name of *fxn* (used because functools.partial(generic_fxn) will remove __doc__ and __name__)
    level_scores - dict
        nested dict of the type: level_scores[program]['garden_performance'] = pd.DataFrame
    """
    print(ColorText(function_name).bold().blue(), '\n\n')

    # RONA temp-only
    data = level_scores['RONA']['garden_performance'].copy()
    data = data[(data['plevel'] != '1-trait') & (data['env']=='temp_opt')].copy()
    fxn('RONA', data, add_title='\n2-trait sims\ntemp only',
        filename=f'{function_name}_RONA_garden_performance_2-trait_temp_only.pdf')

    # RONA sal-only
    data = level_scores['RONA']['garden_performance'].copy()
    data = data[(data['plevel'] != '1-trait') & (data['env']=='sal_opt')].copy()
    fxn('RONA', data, add_title='\n2-trait sims\nsal only',
        filename=f'{function_name}_RONA_garden_performance_2-trait_sal_only.pdf')

    # all programs (combine rona envs and rda structcorr)
    for program in keys(level_scores):
        data = level_scores[program]['garden_performance'].copy()

        data = data[data['plevel'] != '1-trait']

        add_title = '\n2-trait sims'
        add = ''
        if program == 'RONA':
            add = 'both envs'
            add_title += f'\n{add}'

        elif program == 'rda':
            add = 'structure corr + uncorr'
            add_title += f'\n{add}'

        add = '_'.join(add.split())
        if add != '':
            add = f'_{add}'

        fxn(program, data, add_title=add_title, filename=f'{function_name}_{program}_garden_performance_2-trait{add}.pdf')

    # rda uncorrected for population structure
    data = level_scores['rda']['garden_performance'].copy()
    data = data[(data['plevel'] != '1-trait') & (data['structcrxn']=='nocorr')].copy()
    fxn('rda', data, add_title='\n2-trait sims\nuncorrected for structure',
        filename=f'{function_name}_rda_garden_performance_2-trait_nocorr.pdf')

    # rda corrected for population structure
    data = level_scores['rda']['garden_performance'].copy()
    data = data[(data['plevel'] != '1-trait') & (data['structcrxn']=='structcorr')].copy()
    fxn('rda', data, add_title='\n2-trait sims\ncorrected for structure',
        filename=f'{function_name}_rda_garden_performance_2-trait_structcorr.pdf')

    pass


def get_summary_data():
    df = pd.read_table('/work/lotterhos/MVP-NonClinalAF/summary_20220428_20220726.txt', delim_whitespace=True)
    df.index = df.seed.astype(str).tolist()

    return df


def get_bcs_data(level_scores):
    """Get validation data for 'best case scenario'.

    Notes
    -----
    2-trait sims evaluated using all markers and all envs, no structure correction in RDA
    """
    bcs = {}
    for program in level_scores:
        # reduce to bcs
        data = level_scores[program]['garden_performance'].copy()
        data = data[(data['marker_set']=='all') & (data['plevel'] == '2-trait')]

        if program == 'rda':
            data = data[data['structcrxn']=='nocorr']

        # combine some columns
        data['architecture'] = data['glevel'].str.cat(data[['plevel', 'pleio', 'slevel']], sep='_').str.replace(' ', '-')
        data['plevel_pleio'] = data['plevel'] + '_' + data['pleio']
        data['demography'] = data['popsize'] + '_' + data['migration']
        data['seed_garden'] = data['garden'].astype(str) + '_' + data['seed'].astype(str)

        bcs[program] = data
        
    # get data that contains level of local adaptation
    summary = get_summary_data()

    # add LA and proportion of clinal alleles to the bcs data
    for df in bcs.values():
        for col in ['final_LA', 'cor_TPR_temp', 'cor_TPR_sal']:
            df[col] = df.seed.map(summary[col]) 

    return bcs


repdirs = ['/work/lotterhos/MVP-Offsets/run_20220919',
           '/work/lotterhos/MVP-Offsets/run_20220919_225-450',
           '/work/lotterhos/MVP-Offsets/run_20220919_450-675',
           '/work/lotterhos/MVP-Offsets/run_20220919_675-900',
           '/work/lotterhos/MVP-Offsets/run_20220919_900-1125',
#            '/work/lotterhos/MVP-Offsets/run_20220919_1125-1350',  # not done yet
          ]

def combine_level_dicts(use_bc_data=True, display_df=False):
    """Across pipline batches, combine garden performance scores into one pd.DataFrame.
    
    Parameters
    ----------
    use_bc_data bool
        whether to use best-case-scenario data; defined in `get_bcs_data()` above
    """
    summary = get_summary_data()
    
    level_dicts = defaultdict(dict)
    for repdir in pbar(repdirs, desc='reading reps'):
        rep = op.basename(repdir).split('_')[-1]

        if rep == '20220919':
            rep = '0-225'

        pkl = op.join(repdir, 'summaries/all_performance_dicts/level_scores.pkl')

        level_scores = pklload(pkl)

        if use_bc_data is True:
            data = get_bcs_data(level_scores)
        else:
            data = {}
            for program in level_scores.keys():
                data[program] = level_scores[program]['garden_performance']

        level_dicts[rep] = data
        
    # read in garden performance
    all_dicts = defaultdict(list)
    for rep, level_dict in level_dicts.items():
        print(ColorText(rep).bold().blue())
        for program, performance_df in level_dict.items():
            print('\t', ColorText(program).bold(), luni(performance_df['seed']))
            all_dicts[program].append(performance_df)
        print('\n')

    # concat data frame lists, overwrite `all_dicts`
    for program in keys(all_dicts):
        df = pd.concat(all_dicts[program])
        num_seeds = luni(df['seed'])

        print('\n', ColorText(program).bold(), f'{df.shape = }', f'{num_seeds = }')

        # add a column for easier grouping
        df['simulation_garden'] = df['simulation_level'] + '-' + df['garden'].astype(str)

        # add column for final_LA
        df['final_LA'] = df.seed.map(summary.final_LA)

        if display_df is True:
            display(df.head())

        all_dicts[program] = df.copy()

    return level_dicts
