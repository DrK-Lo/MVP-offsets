"""Functions used to create figs across scripts that summarize output of MVP offset validation pipeline.

TODO
----
- LA into combine: 02_analysis/04_outlier_climate_runs/05_visualize_validation_of_outlier_predictions.ipynb
    02_analysis/02_main_questions/02_Q1_performance_vs_local-adaptation_figures.ipynb

"""
from pythonimports import *
from myfigs import adjust_box_widths, save_pdf

import MVP_10_train_lfmm2_offset as mvp10

import builtins
import inspect
import seaborn as sns
from matplotlib.lines import Line2D

# DICTIONARIES FOR FIGURE / TABLE MAKING
boxplot_kwargs = {  # kwargs for seaborn.catplot (boxplot) properties
    'palette' : {'no pleiotropy' : '#3e4a89',
                 'pleiotropy' : '#35b779',
                 
                 'SS-Clines' : '#440c54',
                 'SS-Mtn' : '#22908c',
                 'Est-Clines' : '#fde624',
                 
                 'edgecolor' : '#bebebe',
                 
                 'equal-S' : 'k',
                 'unequal-S' : 'darkgray',
                 
                 'N-equal' : '#0b559f',  # popsizes are sns.color_palette('Blues_r') using matplotlib.colors.rgb2hex
                 'N-variable' : '#2b7bba',
                 'N-cline-center-to-edge' : '#539ecd',
                 'N-cline-N-to-S' : '#89bedc',
                 
                 'm-constant' : '#572c92',        # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[0])
                 'm-variable' : '#705eaa',        # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[1])
                 'm-breaks' : '#8d89c0',          # matplotlib.colors.rgb2hex(sns.color_palette('Purples_r')[2])

                 '1-trait 0-nuisance' : '#e9f7e5',
                 '1-trait 1-nuisance' : '#d3eecd',
                 '1-trait 3-nuisance' : '#b8e3b2',
                 '1-trait 4-nuisance' : '#98d594',
                 '2-trait 0-nuisance' : '#73c476',
                 '2-trait 2-nuisance' : '#4bb062',
                 '2-trait 3-nuisance' : '#2f974e',
                 '6-trait 0-nuisance' : '#157f3b',
                 '6-trait 3-nuisance' : '#006428',
                 
                 'N-equal_m-constant' : '#7dba91',  # rgb2hex(sns.color_palette('crest')[0])
                 'N-equal_m-breaks' : '#59a590',  # rgb2hex(sns.color_palette('crest')[1])
                 'N-cline-center-to-edge_m-constant' : '#40908e',  # rgb2hex(sns.color_palette('crest')[2])
                 'N-cline-N-to-S_m-constant' : '#287a8c',  # rgb2hex(sns.color_palette('crest')[3])
                 'N-variable_m-variable' : '#1c6488',  # rgb2hex(sns.color_palette('crest')[4])

                 'oligogenic' : '#aa1016',        # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[0])
                 'mod-polygenic' : '#f44f39',     # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[2])
                 'highly-polygenic' : '#fcaf93',  # matplotlib.colors.rgb2hex(sns.color_palette('Reds_r')[4])
                 
                 'RONA' : sns.color_palette("Paired")[1],
                 'RONA-sal_opt' : 'k',
                 'RONA-temp_opt' :  (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),  # grayish
                 'lfmm2' : sns.color_palette("Paired")[-5],
                 'GF' : sns.color_palette("Paired")[9],
                 'rda' : sns.color_palette("viridis")[-1],
                 'rda-nocorr' : 'darkgreen',
                 'rda-structcorr' : 'lightgreen',
                 
                 'adaptive' : sns.color_palette('magma', n_colors=11)[-5],
                 'all' : sns.color_palette('magma', n_colors=11)[-3],
                 'neutral' : sns.color_palette('magma', n_colors=11)[-1],
                 'rda outliers' : 'white',
                 
                 'northwest' : (0.38407269378943537, 0.46139018782416635, 0.7309466543290268),
                 'rangecenter' : (0.18488035509396164, 0.07942573027972388, 0.21307651648984993),
                 'southeast' : (0.6980608153581771, 0.3382897632604862, 0.3220747885521809),

                 'Est-Clines_1-trait': 'orange',
                 'Est-Clines_equal-S': 'k',
                 'Est-Clines_unequal-S': 'darkgray',
                 'SS-Clines_1-trait': 'orange',
                 'SS-Clines_equal-S': 'k',
                 'SS-Clines_unequal-S': 'darkgray',
                 'SS-Mtn_1-trait': 'orange',
                 'SS-Mtn_equal-S': 'k',
                 'SS-Mtn_unequal-S': 'darkgray',
                 '1-trait single-S' : 'orange',
                 '2-trait equal-S' : 'k',
                 '2-trait unequal-S' : 'darkgray'
                },
    
    'whiskerprops' : {'color' : '#bebebe'},
    
    'medianprops' : {'color' : '#bebebe',
                     'alpha' : 1.0},
    
    'boxprops' : {'edgecolor' : '#bebebe'},
    
    'flierprops' : {'color' : '#bebebe',
                    'markeredgecolor' : '#bebebe',
                    'markerfacecolor' : 'none',
                    'markeredgewidth' : 0.5,
                    'markersize' : 4,
                    'marker' : '.'},
    
    'capprops' : {'color' : '#bebebe'}
}


markers = {
    'RONA-sal_opt' : '<',  # left-facing triangle
    'RONA-temp_opt' : '^',  # upward-facing triangle
    'lfmm2' : 'p',  # pentagon
    'GF' : 'o',  # circle
    'rda-nocorr' : 's',  # square
    'rda-structcorr' : 'D',  # diamond,
    'Est-Clines' : "3",  # 3 tri_left (not triangle)
    'SS-Clines' : "+",  # + plus
    'SS-Mtn' : "x",  # x X
    'Est-Clines_1-trait' : "3",
    'Est-Clines_equal-S' : "3",
    'Est-Clines_unequal-S' : "3",
    'SS-Clines_1-trait' : "+",
    'SS-Clines_equal-S' : "+",
    'SS-Clines_unequal-S' : "+",
    'SS-Mtn_1-trait' : "x",
    'SS-Mtn_equal-S' : "x",
    'SS-Mtn_unequal-S' : "x"
}


hue_order = {
    'landscape' : ['SS-Clines', 'SS-Mtn', 'Est-Clines'],
    'glevel' : ['oligogenic', 'mod-polygenic', 'highly-polygenic'],
    'pleio' : ['no pleiotropy', 'pleiotropy'],
    'slevel' : ['equal-S', 'unequal-S'],
    'popsize' : ['N-equal', 'N-variable', 'N-cline-center-to-edge', 'N-cline-N-to-S'],
    'migration' : ['m-constant', 'm-variable', 'm-breaks'],
    'noncausal_env' : ['1-trait 1-env', '1-trait 2-envs', '2-trait 2-envs'],
    'marker_set' : ['adaptive', 'all', 'neutral'],
    'program' : ['RONA-sal_opt', 'RONA-temp_opt', 'lfmm2', 'rda-nocorr', 'rda-structcorr', 'GF'],
    'demography' : ['N-equal_m-constant', 'N-equal_m-breaks', 'N-cline-center-to-edge_m-constant',
                    'N-cline-N-to-S_m-constant', 'N-variable_m-variable']
}


# convert environmental names to number of traits and envs
nuis_dict = {
    'ISO-PSsd' : '2-trait 4-envs',
    'ISO-TSsd-PSsd' : '2-trait 5-envs',
    'sal-ISO-PSsd' : '1-trait 4-envs',
    'sal-ISO-TSsd-PSsd' : '1-trait 5-envs'
}

# for pretty figures:
factor_names = {level : level for (k, levels) in hue_order.items() for level in levels}  # add default names to factor_names
factor_names.update({  # update/overwrite factor_names with what I really want
    # used in eg anova models to convert column name to human-readable label for figure
    # also used to make pretty legend/axis labels/titles
    'architecture' : 'architecture',
    'demography' : 'demography',
    'glevel' : 'Polygenicity',
    'landscape' : 'Landscape',
    'slevel' : 'Selection\nStrength',
    'popsize_migration' : 'demography',
    'plevel_pleio' : 'pleiotropy',
    'cor_TPR_temp' : "$p_{cQTN, temp}$",
    'cor_TPR_sal' : "$p_{cQTN, Env2}$",
    'final_LA' : 'Degree of\nlocal adaptation',
    'Residual' : 'Residual',
    'C(garden)' : 'Garden ID',
    'final_LA:architecture' : 'final_LA:architecture',
    'final_LA:landscape' : 'final_LA:landscape',
    'final_LA:C(garden)' : 'final_LA:C(garden)',
    'final_LA:slevel' : 'final_LA:slevel',
    'cor_FPR_temp_neutSNPs' : "$p_{cNeut, temp}$",
    'cor_FPR_sal_neutSNPs' : "$p_{cNeut, Env2}$",
    'Intercept' : 'Intercept',
    'pleio' : 'Pleiotropy',
    'popsize' : 'Population Size',
    'migration' : 'Migration',
    'noncausal_env' : 'Nuisance\nLevel',
    'nuis_envs' : 'Nuisance\nLevel',
    'marker_set' : 'Marker Set',
    'program' : 'Method',
    'SS-Clines' : 'Stepping Stone - Clines',
    'SS-Mtn' : 'Stepping Stone - Mountain',
    'Est-Clines' : 'Estuary - Clines',
    'rda-structcorr' : 'RDA-corrected',
    'rda-nocorr' : 'RDA-uncorrected',
    'GF' : 'Gradient Forests',
    'lfmm2' : 'LFMM2',
    'rda' : 'RDA',
    '1-trait' : 'Nuisance\nLevel',
    '2-trait' : 'Nuisance\nLevel',
    '1-trait 2-envs' : '1-trait 1-nuisance',
    '2-trait 2-envs' : '2-trait 0-nuisance',
    '1-trait 1-env' : '1-trait 0-nuisance',
    'sal-ISO-PSsd' : '1-trait 3-nuisance',  # 1-trait 4-envs
    'ISO-PSsd' : '2-trait 2-nuisance',  # 2-trait 4-envs
    'sal-ISO-TSsd-PSsd' : '1-trait 4-nuisance',  # 1-trait 5-envs
    'ISO-TSsd-PSsd' : '2-trait 3-nuisance',  # 2-trait 5-envs
    'RONA-sal_opt' : 'RONA (Env2)',
    'RONA-temp_opt' : 'RONA (temp)',
    'block' : 'Population\nBlock',
    'mod-polygenic' : 'moderately polygenic',
    'highly-polygenic' : 'highly polygenic',
    'rda outliers' : 'rda outliers',
    'N-equal_m-constant' : 'N-equal m-constant',
    'N-equal_m-breaks' : 'N-equal m-breaks',
    'N-cline-center-to-edge_m-constant' : 'N-cline-center-to-edge m-constant',
    'N-cline-N-to-S_m-constant' : 'N-cline-N-to-S m-constant',
    'N-variable_m-variable' : 'N-variable m-variable',
    'demography' : 'Demography',
    'northwest' : 'northwest',
    'rangecenter' : 'range center',
    'southeast' : 'southeast',
    'landscape-slevel' : 'Landscape x Selection Strength',
    'Est-Clines_1-trait': 'Estuary - Clines single-trait',
    'SS-Clines_1-trait': 'Stepping Stone - Clines single-trait',
    'SS-Mtn_1-trait': 'Stepping Stone - Mountain single-trait',
    'Est-Clines_equal-S': 'Estuary - Clines equal-S',
    'SS-Clines_equal-S': 'Stepping Stone - Clines equal-S',
    'SS-Mtn_equal-S': 'Stepping Stone - Mountain equal-S',
    'Est-Clines_unequal-S': 'Estuary - Clines unequal-S',
    'SS-Clines_unequal-S': 'Stepping Stone - Clines unequal-S',
    'SS-Mtn_unequal-S': 'Stepping Stone - Mountain unequal-S',
    'slevel_plus1' : 'Trait x Selection Strength',
    '1-trait single-S' : '1-trait single-S',
    '2-trait equal-S' : '2-trait equal-S',
    '2-trait unequal-S' : '2-trait unequal-S'
#     'Est-Clines' : "3",  # tri_left (not triangle)
#     'SS-Clines' : "+",  # plus
#     'SS-Mtn' : "x",  # vertical line
#     '1-trait 1-env' : '1-causal 0-nuisance',
#     '1-trait 2-envs' : '1-causal 1-nuisance',
#     '2-trait 2-envs' : '2-causal 0-nuisance',
#     '1-trait 4-envs' : '1-causal 3-nuisance',
#     '1-trait 5-envs' : '1-causal 4-nuisance',
#     '2-trait 4-envs' : '2-causal 2-nuisance',
#     '2-trait 5-envs' : '2-causal 3-nuisance',
})

ytick_labels = [i/100 for i in range(-100, 66, 20)][::-1]

# climate outlier scenarios for temp (sal = temp * -1) (raw climate values)
new_envs = [0, 1.10, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75]

slimdir = '/work/lotterhos/MVP-NonClinalAF/sim_output_20220428'

# for validating climate outlier scenarios - determined in 02_analysis/04_outlier_climate_runs/02_define_pop_groups.ipynb
block_pops = {
    'northwest': [91, 92, 93, 81, 82, 83, 71, 72, 73],
    'rangecenter': [54, 55, 56, 44, 45, 46, 34, 35, 36],
    'southeast': [28, 29, 30, 18, 19, 20, 8, 9, 10]
}


# FUNCTIONS
def display_level_dfs(level_scores=None):
    """Display first 5 lines of dataframes created in 02.01.00_save_level_scores.ipynb."""
    for program in keys(level_scores):
        print(program)
        display(level_scores[program]['garden_performance'].head())
    pass


def jitter_fliers(g=None, axes=None, jitter_axis='x', jit=0.05):
    """Add jitter to seaborn boxplot outliers.
    
    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid
        eg returned from seaborn.catplot etc
        
    Notes
    -----
    Thanks - https://stackoverflow.com/questions/61638303/how-to-jitter-the-outliers-of-a-boxplot
    """
    if g is not None:
        axes = g.axes.flat
    
    for ax in axes:
        for artist in ax.get_lines():
            if artist.get_linestyle() == "None":
                if jitter_axis == 'x' :
                    pos = artist.get_xdata()
                    artist.set_xdata(pos + np.random.uniform(-jit, jit, len(pos)))
                else:
                    assert jitter_axis == 'y'
                    pos = artist.get_ydata()
                    artist.set_ydata(pos + np.random.uniform(-jit, jit, len(pos)))
                    
    pass


def pretty_facetgrid(g, program, num_levels, num_reps, saveloc, add_title, center=0.5):
    """Make the FacetGrid look nice.
    
    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid (e.g. as returned from seaborn.catplot)
    program : str
        the offset method name
    num_levels : int
        the unique number of seeds used to create the fig represented by *g*
    saveloc : str
        path for where to save
    add_title : str
        string to add to title of figure
    center : float
        where to center title and 'Marker Set' x-axis label
    """
    # line at tau = 0
    for ax in g.axes[0]:
        ax.axhline(0, linestyle='--', color='gainsboro', linewidth=0.5, zorder=0)

    # make it pretty
#     g.set_titles('{col_name}')
#     print('g.get_titles() = ', g.get_titles())
#     g.set_titles([factor_names[title.get_text()] for title in g.get_titles()])
    [ax.set_title(factor_names[ax.get_title().split()[-1]], ha='center') for ax in g.axes.flatten()]
    g.set_axis_labels("", "Performance (Kendall's $\\tau$)", fontsize=12)
    adjust_box_widths(list(g.axes[0]), 0.85)
    g.set(ylim=(0.65, -1),
          yticks=ytick_labels,
          yticklabels=ytick_labels)
    jitter_fliers(g)
    
    
    # add labels
    title = f'{program}\n{num_levels = } {num_reps = }{add_title}'
    print(title)
#     g.fig.suptitle(title, x=center)
    g.fig.text(y=-0.05, x=center, s='Marker Set', fontsize=12, ha='center')
    
    g.tight_layout()
    
    # save
    save_pdf(saveloc)

    plt.show()

    pass


def run_facetgrid_figs(fxn, function_name, all_scores=None, plevel='2-trait'):
    """Apply function *fxn* to see how parameters affect validation scores from sims.
    
    See notebooks in 02.01_summary_figs folder.

    Parameters
    ----------
    fxn : Callable
        an executable function that takes args=(program, data) and kwargs={hue, hue_order, add_title, filename}
            (see generic functions in eg 01_glevel_vs_other.ipynb or 02_landscape_vs_other.ipynb)
    function_name : str
        name of *fxn* (used because functools.partial(generic_fxn) will remove __doc__ and __name__)
    all_scores : dict
        nested dict of the type: all_scores[program]['garden_performance'] = pd.DataFrame
        
    Notes
    -----
    - *fxn* will take filename as basename and add directory for saving
    """
    print(ColorText(function_name).bold().blue(), '\n\n')

    # RONA temp-only
    data = all_scores['RONA'].copy()
    data = data[(data['plevel'] == plevel) & (data['env']=='temp_opt')].copy()
    fxn('RONA', data, add_title=f'\n{plevel} sims\ntemp only',
        filename=f'{function_name}_RONA_garden_performance_{plevel}_temp_only.pdf')

    # RONA sal-only
    data = all_scores['RONA'].copy()
    data = data[(data['plevel'] == plevel) & (data['env']=='sal_opt')].copy()
    fxn('RONA', data, add_title=f'\n{plevel} sims\nsal only',
        filename=f'{function_name}_RONA_garden_performance_{plevel}_sal_only.pdf')

    # all programs (combine rona envs and rda structcorr)
    for program in keys(all_scores):
        data = all_scores[program].copy()

        data = data[data['plevel'] == plevel]

        add_title = f'\n{plevel} sims'
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

        fxn(program, data, add_title=add_title, filename=f'{function_name}_{program}_garden_performance_{plevel}{add}.pdf')

    # rda uncorrected for population structure
    data = all_scores['rda'].copy()
    data = data[(data['plevel'] == plevel) & (data['structcrxn']=='nocorr')].copy()
    fxn('rda', data, add_title=f'\n{plevel} sims\nuncorrected for structure',
        filename=f'{function_name}_rda_garden_performance_{plevel}_nocorr.pdf')

    # rda corrected for population structure
    data = all_scores['rda'].copy()
    data = data[(data['plevel'] == plevel) & (data['structcrxn']=='structcorr')].copy()
    fxn('rda', data, add_title=f'\n{plevel} sims\ncorrected for structure',
        filename=f'{function_name}_rda_garden_performance_{plevel}_structcorr.pdf')

    pass


def get_summary_data():
    """Read in all simulation metadata."""
    df = pd.read_table('/work/lotterhos/MVP-NonClinalAF/summary_20220428_20220726.txt', delim_whitespace=True)
    df.index = df.seed.astype(str).tolist()

    return df


def subset_dataframe(df, num_traits=None, ntraits=None, marker_set=None, remove_structcrxn=False,
                     keep_nuisance=True, remove_rda_outliers=True, bcs=False):
    """Subset performance dataframe to kwargs.
    
    Parameters
    ----------
    num_traits : int
        the number of traits under selection in the sims
    ntraits : int
        the number of traits used to train offset methods
        ie if num_traits == 2 then ntraits = 2; if num_traits == 1 then ntraits in {1, 2}    
    marker_set : str
        one of {'all', 'adaptive', 'neutral'}
    """
    if bcs is True:
        marker_set = 'all'
        num_traits = 2
        keep_nuisance = False
        remove_rda_outliers = True
        
    df = df.copy()  # avoid slice warnings

    if num_traits is not None:
        df = df[df['plevel'] == f'{num_traits}-trait']

    if marker_set is not None:
        df = df[df['marker_set'] == marker_set]

    if 'ntraits' in df.columns.tolist() and ntraits is not None:
        df = df[df['ntraits'] == f'ntraits-{ntraits}']

    if 'structcrxn' in df.columns.tolist() and remove_structcrxn is True:
        df = df[df['structcrxn'] == 'nocorr']

    if keep_nuisance is False:
        if 'noncausal_env' in df.columns.tolist():
            df = df[(df.noncausal_env != '1-trait 2-envs') | (df.noncausal_env.isnull())]  # allows RONA

        rona_df = df[df.program.str.contains('RONA')].copy()  # allows for RONA-sal_opt and RONA-temp_opt
        original_df = df[~df.program.str.contains('RONA')]
        if nrow(rona_df) > 0:
            trait_env = rona_df.plevel + '-' + rona_df.env
            rona_df = rona_df[trait_env != '1-trait-sal_opt']  # keep 1-trait-temp_opt, 2-trait-temp_opt, 2-trait-sal_opt

            df = pd.concat([original_df, rona_df])

    if remove_rda_outliers is True:  # remove rda outlier marker set
        df = df[~df.marker_set.str.contains('outliers')]
    
    return df


def get_1trait_data(df, ntraits=None, marker_set=None, remove_structcrxn=False, remove_rda_outliers=True):
    """Subset performance dataframe for either 1-trait sims with no nuisance envs.
    
    Parameters
    ----------
    all passed to subset_dataframe (kwargs are set to default except for those that are hard-coded below)
    """
    data = subset_dataframe(df, num_traits=1, keep_nuisance=False, marker_set=marker_set, remove_structcrxn=remove_structcrxn,
                            remove_rda_outliers=remove_rda_outliers)
        
    return data
    

def get_bcs_data(level_scores, performance='garden_performance'):
    """Get validation data for 'best case scenario'.

    Notes
    -----
    2-trait sims evaluated using all markers and all envs
    """
    bcs = {}
    for program in level_scores:
        # reduce to bcs b
        data = level_scores[program][performance].copy()
        
        bcs[program] = subset_dataframe(data, bcs=True)

    return bcs

# where I keep outputs of 1- and 2-trait sims; each dir is 1 rep of 225 unique levels (10 reps)
repdirs = ['/work/lotterhos/MVP-Offsets/run_20220919_0-225',
           '/work/lotterhos/MVP-Offsets/run_20220919_225-450',
           '/work/lotterhos/MVP-Offsets/run_20220919_450-675',
           '/work/lotterhos/MVP-Offsets/run_20220919_675-900',
           '/work/lotterhos/MVP-Offsets/run_20220919_900-1125',
           '/work/lotterhos/MVP-Offsets/run_20220919_1125-1350',
           '/work/lotterhos/MVP-Offsets/run_20220919_1350-1575',
           '/work/lotterhos/MVP-Offsets/run_20220919_1575-1800',
           '/work/lotterhos/MVP-Offsets/run_20220919_1800-2025',
           '/work/lotterhos/MVP-Offsets/run_20220919_2025-2250',
          ]

@timer
def combine_level_dicts(use_bcs_data=True, display_df=False, performance='garden_performance', repdirs=None,
                        add_1_trait=None, **kwargs):
    """Across pipeline batches, combine garden performance scores into one performance pd.DataFrame.

    Parameters
    ----------
    use_bcs_data : bool
        whether to use best-case-scenario data; defined in `get_bcs_data()` above
    display_df : bool
        print a df.head() or not
    performance : str
        one of 'garden_performance', or 'source_performance'
    rep_dirs : list
        a list of directories to retrieve offset performance info (default None will use all `repdirs` from above)
    add_1_trait : bool
        whether to include GF 1-trait 0-nuisance in performance output
    kwargs : dict
        passed to subset_dataframe()
    """
    import MVP_summary_functions as mvp

    if repdirs is None:
        repdirs = mvp.repdirs  # this is repdirs from above
        
    if 'num_traits' in keys(kwargs) and kwargs['num_traits'] == 2:
        add_1_trait = False
    else:
        add_1_trait = True

    summary = get_summary_data()    

    # print filtering criteria
    print(ColorText('filtering criteria:\n').bold().green().__str__() + 
          ColorText('\tuse_bcs_data = ').bold().__str__() + f'{use_bcs_data}\n' +
          ColorText('\tperformance = ').bold().__str__() + f'{performance}')
    signature = inspect.signature(subset_dataframe)
    for kwarg in signature.parameters.keys():
        if kwarg in ['df', 'bcs']:
            continue

        if kwarg in keys(kwargs):
            default = kwargs[kwarg]
        else:
            default = signature.parameters[kwarg]._default

        if use_bcs_data is True:  # the following kwargs are set in `subset_dataframe`
            if kwarg == 'marker_set':
                default = 'all'
            if kwarg == 'num_traits':
                default = 2
            if kwarg == 'keep_nuisance':
                default = False
            if kwarg == 'remove_rda_outliers':
                default = True

        print(ColorText(f'\t{kwarg} = ').bold().__str__() + f'{default}')

    level_dicts = defaultdict(dict)
    for repdir in pbar(repdirs, desc=f'reading reps'):
        rep = op.basename(repdir).split('_')[-1]

        pkl = op.join(repdir, 'summaries/all_performance_dicts/level_scores.pkl')

        level_scores = pklload(pkl)

        # add program name
        for program in level_scores.keys():
            df = level_scores[program][performance]  # do not copy
            if program == 'RONA':
                df['program'] = 'RONA' + '-' + df['env']
            elif program == 'rda':
                df['program'] = 'rda' + '-' + df['structcrxn']
            else:
                df['program'] = program

        if use_bcs_data is True:
            data = get_bcs_data(level_scores, performance=performance)
        else:
            data = {}
            for program in level_scores.keys():
                df = level_scores[program][performance].copy()

                df = subset_dataframe(df, **kwargs)

                if program == 'GF' and add_1_trait is True:                  
                    # add in the 1-trait 0-nuisance runs
                    pkl = pkl.replace('run_', '1-trait_run_')  # change directory name
                    onetrait_df = pklload(pkl)[program][performance]

                    # add program name
                    onetrait_df['program'] = program

                    df = pd.concat([df, onetrait_df])

                data[program] = df

        level_dicts[rep] = data

    # make a list of dataframes for each program, one df for each rep
    all_scores = defaultdict(list)
    for rep, level_dict in level_dicts.items():
        print(ColorText(rep).bold().blue())

        for program, performance_df in level_dict.items():
            print('\t', ColorText(program).bold(), 'num seeds = ', luni(performance_df['seed']))

            performance_df['rep'] = rep  # add rep for counting reps in figs etc

            all_scores[program].append(performance_df)

        print('\n')

    # concat data frame lists, overwrite `all_scores`
    for program in keys(all_scores):
        df = pd.concat(all_scores[program])
        num_seeds = luni(df['seed'])
        num_reps = luni(df['rep'])

        print('\n', ColorText(program).bold(), f'{df.shape = }', f'{num_seeds = }', f'{num_reps = }')

        # add a column for easier grouping
        df['simulation_garden'] = df['simulation_level'] + '-' + df['garden'].astype(str)

        # add LA and proportion of clinal alleles to the bcs data
        for col in ['final_LA', 'cor_TPR_temp', 'cor_TPR_sal', 'cor_FPR_temp_neutSNPs', 'cor_FPR_sal_neutSNPs']:
            df[col] = df.seed.map(summary[col])

        # combine some columns
        df['plevel_pleio'] = df['plevel'] + '_' + df['pleio']
        df['demography'] = df['popsize'] + '_' + df['migration']

        if display_df is True:
            display(df.head())

        all_scores[program] = df.copy()

    return all_scores


def combine_program_data(scores) -> pd.DataFrame:
    """Combine validation score dataframes across programs into one dataframe.

    Parameters
    ----------
    scores : dict
        output from combine_level_dicts
        key = program, value = pd.DataFrame

    Example
    -------
    >>> program_data = combine_program_data(combine_level_dicts())
    """
    dfs = []
    for program, df in scores.items():
        dfs.append(df.copy())

    return pd.concat(dfs)


def subset_data_scores(scores, apply_median=True, **kwargs):
    """Subset scores to kwargs.
    
    Parameters
    ----------
    scores : dict
        eg object returned from combine_level_dicts
        note that if used on object returned from get_bcs_data, `num_traits` and `marker_set` are irrelevant
    apply_median : bool
        whether to apply a median score per seed
    """
    data = {}
    for program in scores.keys():
        df = subset_dataframe(scores[program].copy(), **kwargs)

#         seed = df.index.tolist()[0]
        print(program, df.seed.value_counts().unique())  # Counter(df.seed)[seed])

        if program == 'RONA':
            for env in ['sal_opt', 'temp_opt']:
                dfenv = df[df['env']==env].copy()

                if apply_median is True:
                    data[f'RONA-{env}'] = dfenv.groupby('seed')['score'].apply(np.median)
                else:
                    data[f'RONA-{env}'] = dfenv['score']
                    
        elif program == 'rda':
            for structcorr in df.program.unique():
                structdf = df[df.program == structcorr].copy()
                
                if apply_median is True:
                    data[structcorr] = structdf.groupby('seed')['score'].apply(np.median)
                else:
                    data[structcorr] = structdf['score']
        else:
            if apply_median is True:
                data[program] = df.groupby('seed')['score'].apply(np.median)  # median score for each seed
            else:
                data[program] = df['score']
            
    return data


def latest_commit():
    """Print today's date, author info, and commit hashes of pythonimports and MVP_offsets."""
    import pythonimports as pyimp

    pyimp_info = pyimp._git_pretty(pyimp._find_pythonimports())
    mvp_info = pyimp._git_pretty('/home/b.lind/code/MVP-offsets')
    current_datetime = "Today:\t" + time.strftime("%B %d, %Y - %H:%M:%S %Z") + "\n"
    version = "python version: " + sys.version.split()[0] + "\n"

    width = max([len(x) for x in flatten([pyimp_info.split('\n'), mvp_info.split('\n'), current_datetime])])
    hashes = '#########################################################\n'
    
    try:
        env = 'conda env: %s\n' % os.environ['CONDA_DEFAULT_ENV']
    except KeyError as e:
        env = ''

    print(
        hashes
        + current_datetime
        + version + f'{env}\n'
        + f"Current commit of %s:\n" % ColorText("pythonimports").bold()
        + pyimp_info + '\n'
        + "Current commit of %s:\n" % ColorText('MVP_offsets').bold().blue()
        + mvp_info
        + hashes
    )
    
    pass


def add_legend(fig_object, fontsize=11, color_by='marker_set', loc='upper left', bbox_to_anchor=None, ncol=1,
               only_rda=False, use_markers=False, legendmarkerfacecolor='fill', exclude_RONA=False,
               one_trait=False, markeredgecolor=None):
    """Add pretty legend to figure, `fig_object`.
    
    Parameters
    ----------
    fig_object : [matplotlib.figure.Figure or matplotlib.axes._subplots.AxesSubplot]
        which object of the figure to add the legend
    fontsize : int
        the fontsize of the legend title and elements
    color_by : str
        key to mvp.hue_order; used to determine how scatter points are colored from boxplot_kwargs['palette']
    loc : str
        location of legend with respect to bbox_to_anchor location
    bbox_to_anchor : tuple
        Box that is used to position the legend in conjunction with *loc*.
    ncol : int
        the number of columns in the legend
    only_rda : bool
        whether this is only RDA and to ignore other programs for legend
    use_markers : bool
    
    legendmarkerfacecolor : str
    
    exclude_RONA : bool
    
    one_trait : bool
        whether to exclude RONA_sal-opt when color_by == 'program'
    markeredgecolor : str | tuple(R, G, B)
        color of shape edge in legend
    """
       
    # i often iterate hue_order, so I don't want to go back and have to exclude these when iterating in other notebooks
    if color_by in ['1-trait', '2-trait', '6-trait']:
        hue_order[color_by] = [
            nuis_level for nuis_level in boxplot_kwargs['palette'] if nuis_level.startswith(color_by)
        ]
    elif color_by == 'block':
        hue_order[color_by] = ['northwest', 'rangecenter', 'southeast']
    elif color_by == 'landscape-slevel':
        hue_order[color_by] = ['Est-Clines_1-trait', 'Est-Clines_equal-S', 'Est-Clines_unequal-S', 'SS-Clines_1-trait',
                               'SS-Clines_equal-S', 'SS-Clines_unequal-S', 'SS-Mtn_1-trait', 'SS-Mtn_equal-S',
                               'SS-Mtn_unequal-S']
    elif color_by == 'slevel_plus1':
        hue_order[color_by] = ['1-trait single-S', '2-trait equal-S', '2-trait unequal-S']
        
    # legend attributes
    legend_title = factor_names[color_by]
    if ncol == 'auto':
        ncol = len(hue_order[color_by])
        legend_title = legend_title.replace('\n', ' ')

    # specific ordering of 'programs' in legend when legend is in upper center
    if ncol == 6 and color_by == 'program':
        h_order = ['RONA-sal_opt', 'RONA-temp_opt', 'rda-nocorr', 'rda-structcorr', 'lfmm2', 'GF']
        ncol = 3  # overwrite ncol
    else:
        h_order = hue_order[color_by]
    
    legend_kws = dict(title=legend_title, fancybox=True, shadow=False,
                      facecolor='whitesmoke', loc=loc, bbox_to_anchor=bbox_to_anchor,
                      prop=dict(family='serif', size=fontsize))
    
    # get the things that go into the legend
    handles = []
    for level in h_order:
        if all([color_by == 'program',  # skip other program names if I'm only looking at RDA
                only_rda is True,
                level not in ['rda-nocorr', 'rda-structcorr']]):
            continue
        if exclude_RONA is True and 'RONA' in level:
            continue
        if one_trait is True and color_by == 'program' and level == 'RONA-sal_opt':
            continue

        color = boxplot_kwargs['palette'][level]
        marker = 'o' if use_markers is False else markers[level]
        if legendmarkerfacecolor == 'fill':
            markerfacecolor = color
        else:
            markerfacecolor = legendmarkerfacecolor

        handles.append(
            Line2D([0], [0], marker=marker, color='none', markerfacecolor=markerfacecolor,
                   markeredgecolor=markeredgecolor if markeredgecolor else color if level != 'rda outliers' else 'black',
                   label=factor_names[level], markersize=fontsize)
        )

    # create a legend
    leg1 = fig_object.legend(handles=handles, ncol=ncol, **legend_kws)
    fig_object.add_artist(leg1)
    plt.setp(leg1.get_title(), family='serif', fontsize=fontsize+1)
    leg1.get_title().set_multialignment('center')

    # remove from dict
    if color_by.endswith('-trait') or color_by in ['landscape-slevel', 'block', 'slevel_plus1']:
        hue_order.pop(color_by)

    return leg1


def update_ticklabels(ax, update='x', replace=None, fontsize=None, **kwargs):
    """Use factor names to make pretty tick labels.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        fig object to change axis labels
    update : str
        either 'x' or 'y' for axis target
    replace : tuple
        use to replace things like spaces or hyphens - eg replace=('-', '\n')
    """
    if replace is not None:
        if isinstance(replace, tuple):
            replace = [replace]
    
    if update == 'x' or update == 'both':
        new_xlabels = [factor_names[label.get_text()] for label in ax.get_xticklabels()]
        if replace is not None:
            for rep in replace:
                new_xlabels = [label.replace(*rep) for label in new_xlabels]
        ax.set_xticklabels(new_xlabels, fontsize=fontsize, **kwargs)

    if update == 'y' or update == 'both':
        new_ylabels = [factor_names[label.get_text()] for label in ax.get_yticklabels()]
        if replace is not None:
            for rep in replace:
                new_ylabels = [label.replace(*rep) for label in new_ylabels]
        ax.set_yticklabels(new_ylabels, fontsize=fontsize, **kwargs)

    pass


def read_params_file():
    """Annotate params file.
    
    Notes
    -----
    - code based on processing notebooks in 02_analysis/01_summary_figs_and_pipeline_output_processing
    """
    summary = get_summary_data()
    params = mvp10.read_params_file('/home/b.lind/offsets/run_20220919_0-225/slimdir')
    params['final_LA'] = params.index.map(summary.final_LA)
    
    for seed in pbar(params.index):
        glevel, plevel, _blank_, landscape, popsize, *migration = params.loc[seed, 'level'].split("_")
        migration = '-'.join(migration)  # for m_breaks to m-breaks (otherwise m-constant)

        if plevel != '1-trait':
            num, trait_str, *pleio, equality, S_str = plevel.split('-')
            plevel = '2-trait'
            pleio = ' '.join(pleio)
            slevel = f'{equality}-S'
        else:
            pleio = 'no pleiotropy'
            slevel = np.nan

        # set single value for each of the locations
        for column, val in zip(
            ['glevel', 'plevel', 'pleio', 'slevel', 'landscape', 'popsize', 'migration'],
            [glevel,    plevel,   pleio,   slevel,   landscape,   popsize,   migration]
        ):
            params.loc[seed, column] = val

    params['landscape-slevel'] = params.landscape + '_' + params.slevel.fillna('1-trait')
            
    return params
