"""Fit trained GF models to climate outlier climates.

Usage
-----
python MVP_climate_outlier_fitting.py argpkl

Parameters
----------
argpkl path to a .pkl file that contains a list, where each list is a tuple of input args. The input args are:

gfOut_trainingfile - file path to trained gradient forest object, saved as .RDS
garden_file - tab delimited .txt file path with columns for `lat`, `lon`, and each env used in training
   - data can be current, future, or uniform (eg for common garden) climate
predfile - path to RDS of object saved from interpolating gradient forest model across landscape
   for the same time period as the data that went into training (created in gradient_training.R)
basename - the basename prefix wanted for saving files in `save_dir`
save_dir - the directory where files should be saved

Notes
-----
this effectively runs the commandline command: Rscript MVP_gf_fitting_script.R arg1 arg2 arg3 arg4 arg5
"""
import MVP_02_fit_gradient_forests as mvp02
import sys
from pythonimports import start_engines, pklload, watch_async
from myclasses import ColorText
from os import path as op


def main():
    print(ColorText('\nStarting parallel fitting of climate outliers ...'))
    
    # load commands
    cmd_args = pklload(argpkl)
    
    # run commands in parallel
    jobs = []
    for args in cmd_args:
        jobs.append(
            lview.apply_async(
                mvp02.fit_gradient_forests, *((rscript_exe, fitting_file) + args)
            )
        )

    # wait until they complete
    watch_async(jobs)
    
    # make sure I redo any jobs that died, probably due to mem
    mvp02.handle_dead_jobs(jobs, cmd_args)
    
    
    print(ColorText('\nShutting down engines ...').bold().custom('gold'))
    print(ColorText('\nDONE!!').bold().green())

    
    pass


if __name__ == '__main__':
    thisfile, argpkl = sys.argv
    
    num = op.basename(argpkl).split("_")[-1].rstrip('.txt')
    
    # start cluster
    print(ColorText('\nStarting engines ...').bold().custom('gold'))
    lview, dview, cluster_id = start_engines(n=7, profile=f'climate_outlier_{num}')
    mvp02.lview = lview
    
    # load objects to cluster
    rscript_exe = '/home/b.lind/anaconda3/envs/r35/lib/R/bin/Rscript'
    mvp02.rscript_exe = rscript_exe
    dview['rscript_exe'] = rscript_exe
    
    fitting_file = op.join(op.dirname(thisfile), 'MVP_gf_fitting_script.R')
    mvp02.fitting_file = fitting_file
    dview['fitting_file'] = fitting_file

    main()
