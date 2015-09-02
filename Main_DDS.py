import DDS                  
import toolkit as util
import os, glob, shutil
import numpy as np
import math as m
import time


#======================== Brief Description ====================================
# This is the main calling script for the Dynamically Dimensioned Search (DDS) 
# algorithm by Bryan Tolson, Department of Civil & Environmental Engineering
# University of Waterloo    
# The algorithm was originally coded on Nov 2005 by Bryan Tolson
# This Python-distribution DDS-Py was coded on Aug 2015 by Thouheed Abdul Gaffoor

# DDS is an n-dimensional stochastic heuristic global optimization algorithm.
# It is coded as a minimizer but is coded with a transformation to make it 
# capable of solving a maximization problem.
# In a maximization problem, the algorithm minimizes 
# the negative of the objective function F (-1*F).  User specifies in inputs 
# whether it is a max or a min problem.

# REFERENCE FOR THIS ALGORITHM:
# Tolson, B. A., and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm 
# for computationally efficient watershed model calibration, Water Resour. Res., 43, 
# W01413, doi:10.1029/2005WR004723.
#===============================================================================
# Definitions

# objfunc_name     - Name of objective function (i.e. Griewank)
# runname          - Compact name to append to all algorithm output files
# num_trials       - Number of optimisation trials
# num_iters        -  Number of iterations per processing slave 
# user_seed        - Random number seed provided by user
# out_print        - Flag to enabled compressed outputs (0 = full output, 1 = summary)
# ini_name         - Name of text file storing user supplied initial solutions
# modeldir         - Subdirectory name where model files are stored
# obj_flag         -   Flag for optimisation problem type: -1 = max problem, 1 = min problem
# num_slaves       - Number of parallel processing slaves used
# pre_empt_flag    - Flag to enable model preemption (0 = disabled, 1 = enabled)

#===============================================================================
# 1.0   Read DDS Input Files ( 1- main control file, 2 - decision variable bounds)

DDS_inp = util.read_DDS_inp('DDS_inp.txt')          # read 1

bounds_file = DDS_inp['objfunc_name'] + '.txt'
DV_bounds = util.read_param_file(bounds_file)       # read 2
num_dec = DV_bounds['S_min'].shape[0]               # number of dec variables
#===============================================================================
# 2.0   Input verification

# Ensure valid entry for parallel processing slaves 
# n= 1: serial run, n = 0: optimised auto slaves, n > 1 = 'n' user specified slaves       
assert DDS_inp['num_slaves'] >= 0, 'For a parallel run, please enter a valid number (> 1) of processing slaves! Try program again.'

# Determine if Parallel or serial execution:
if DDS_inp['num_slaves'] > 1:
    # parallel run with 'n' user specified slaves
    parallel_run = True
    # total iters = n slaves * evaluations per slave                                                         
    DDS_inp['num_iters'] = DDS_inp['num_iters']*DDS_inp['num_slaves']
    # number of initial solution iters = number of slaves (each slave gets one eval)           
    its = DDS_inp['num_slaves']                                                 
elif DDS_inp['num_slaves'] == 0:
    parallel_run = True
    # total iters = n slaves * evaluations per slave 
    DDS_inp['num_iters'] = DDS_inp['num_iters']*DDS_inp['num_slaves']
    # number of initial solution iters = number of slaves (each slave gets one eval) 
    its = DDS_inp['num_slaves']
elif DDS_inp['num_slaves'] == 1:
    parallel_run = False
    # number of initial solution iters = max of 5 and 0.5% of total iterations
    its=max(5,np.around(0.005*DDS_inp['num_iters']))


assert DDS_inp['obj_flag'] == -1 or DDS_inp['obj_flag'] == 1, 'Please enter -1 or 1 for objective function flag!  Try program again.'

assert DDS_inp['num_trials'] >= 1 and DDS_inp['num_trials'] <= 1000, 'Please enter 1 to 1000 optimization trials!  Try program again.'

assert DDS_inp['num_iters'] >= 7 and DDS_inp['num_iters'] <= 1000000, 'Please enter 7 to 1000000 for max # function evaluations!  Try program again.'

assert DDS_inp['out_print'] ==0 or DDS_inp['out_print'] ==1, 'Please enter 0 or 1 for output printing flag! Try program again.'

# Set random seed
np.random.seed(DDS_inp['user_seed'])

# Initial Solution Set-up
if DDS_inp['ini_name'] != '0':      # Case where initial sols file is provided
    its = 1
    Init_Mat = np.loadtxt(DDS_inp['ini_name'],dtype=float,comments = '#',skiprows =2)
    assert DDS_inp['num_trials'] == Init_Mat.shape[0], 'Number of initial solutions does not match # trials selected. Try program again.'
    assert num_dec == Init_Mat.shape[1], 'Number of dec vars in S_min & initial solution matrix not consistent.'

#===============================================================================
# 3.0   Definition of directory and model subdirectory structure 

# Set script directory - location of this script
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# Get objective function file type by splitting ext. (i.e. 'myfunc'.'exe')
ext_name = DDS_inp['objfunc_name'].split('.')

if len(ext_name) == 1:
    # Python objective function (*.py)
    exe_name = np.array([])
    # sets executable file name variable to null
else: # case where .exe or .bat file is called
    exe_name = DDS_inp['objfunc_name']
    # indicates that toolkit function ext_function will run to handle .exe file
    DDS_inp['objfunc_name'] ='ext_function'

# If subdirectory for model is not specified:
if DDS_inp['modeldir'] == '0':
    Modeldir = script_dir
# If relative path specified and parallel run
elif DDS_inp['modeldir'] != 0 and parallel_run is True:
    # Set Model subdirectory
    Modeldir = os.path.join(script_dir, DDS_inp['modeldir'])
    #Generate copies of base-model for slave-acess
    util.generate_dir(DDS_inp['num_slaves'],Modeldir)
# Else relative path and serial run
else:  
    Modeldir = os.path.join(script_dir, DDS_inp['modeldir'])
#===============================================================================
# 4.0   Define Output files and arrays:

filenam1=DDS_inp['runname'] +'_ini.out'         # initial sol'n file name
filenam2=DDS_inp['runname'] + '_AVG.out'        # avg trial outputs
filenam3=DDS_inp['runname'] + '_sbest.out'      # output best DV solutions per trial
filenam4=DDS_inp['runname'] + '_trials.out'     # output Jbest per iteration number per trial

# Master output solution Matrix - Initial iteration = initial solution eval
master_output = np.empty((DDS_inp['num_iters'] + its, 3 + num_dec),dtype= float)
initial_sols = np.empty((its, 3 + num_dec),dtype = float)
output = np.empty((DDS_inp['num_iters'],3),dtype = float)
# tracks only Jbest but for all trials in one file
Jbest_trials=np.empty((DDS_inp['num_iters'],DDS_inp['num_trials']),dtype = float) 
# Matrix holding the best sets of decision variables
Sbest_trials=np.empty((DDS_inp['num_trials'],num_dec),dtype = float)  
sum_output = np.empty((DDS_inp['num_iters']-its,3),dtype =float)
# Matrix holding averages
MAT_avg = np.empty_like(sum_output)
#===============================================================================
# 5.0   Main Algorithm Calling Loop

for j in range(0,DDS_inp['num_trials']):
    
    # Output to console:
    print('Trial number %s executing ... '%(j+1))
    
    # Start timer:
    t_0 = time.time()
    
    # Feed initial solutions:
    if DDS_inp['ini_name'] == '0':
        sinitial = np.array([])
    else:
        sinitial = Init_Mat[j,:]
    
    # Call either Serial or MPI DDS Algorithm:
    if parallel_run == False:
        output = DDS.DDS_serial(DDS_inp['objfunc_name'],exe_name,Modeldir,DDS_inp['obj_flag'],DV_bounds,sinitial,its,DDS_inp['num_iters'])
    #else:
        #output = DDS.DDS_MPI()
        
    # store initial solution results
    initial_sols = output['Master'][0:its,:]
    
    # store truncated outputs - Columns: 0 -> iter #; 1 -> Jbest; 2 -> Jtest
    trunc_out = output['Master'][its:,0:3]
    output_ALL = output['Master'][its:,:]

    # accumlate only Jbest for each trial:
    Jbest_trials[:,j]= output['Master'][:,1]
    # accumulate only Sbest for each trial:
    Sbest_trials[j,:] = output['Best_sol']
    
    if DDS_inp['out_print'] == 0:
        # Write Master Output Matrix at every trial (i.e. - 'Ex1_trial_1.out'):
        # ---------------------------------------------------------------------
        master_file = DDS_inp['runname']+'_trial_' + str(j+1) +'.out'
        np.savetxt(master_file,output['Master']) 

        # Write Dec. Var. best solutions at every trial (i.e. - 'sbest_trial_1.out'):
        # --------------------------------------------------------------------------
        sbest_file = 'sbest'+ '_trial_' + str(j+1) +'.out'
        np.savetxt(sbest_file,output['Best_sol']) 

        # Write Initial Solution to 'Ex1_ini_1.out':
        # ----------------------------------------
        ini_file = DDS_inp['runname'] + '_ini_' + str(j+1) + '.out'
        np.savetxt(ini_file,initial_sols) 

        # Write Jbest compressed output file (i.e. - 'Jbest_trial_1.out'):
        # ----------------------------------------------------------------
        Jbest_file = 'Jbest_trial_' + str(j+1) + '.out'
        np.savetxt(Jbest_file,Jbest_trials) 
    
    # Prepare matrix for average performance evaluation:
    sum_output = sum_output + trunc_out 
    
    # Stop trial timer
    t_1 = time.time() 
    runtime = t_1 - t_0
    
    # Output to console
    print('Best objective function value of %f found at Iteration %i \n'%(output['F_Best'], output['Best_iter']))
    print('Time of execution for Trial %i was %f seconds or %f hours. \n\n' %(j+1,runtime,runtime/3600))
#============================================================================
# 6.0   Post Processing

# Generate average results from all trials
Jbest_avg = np.divide(sum_output[:,1], DDS_inp['num_trials'])
Jtest_avg = np.divide(sum_output[:,2], DDS_inp['num_trials'])
MAT_avg[:,0] = range(DDS_inp['num_iters']-its)
MAT_avg[:,1] = Jbest_avg
MAT_avg[:,2] = Jtest_avg

# Write averages to output file
avg_file = DDS_inp['runname'] + '_trial_avgs' + '.out'
np.savetxt( avg_file,MAT_avg)

# Generate output directory
outpath = os.path.join(script_dir,(DDS_inp['runname'] + '_Output'))
if os.path.exists(outpath):         # If output directory exists - empty it
    exis_files = glob.glob(os.path.join(outpath, "*.out"))
    for f in exis_files:
        os.remove(f)
else:
    os.makedirs(outpath)            # Else nonexistent - make new output directory 

# Move output files to a new directory
out_files = glob.iglob(os.path.join(script_dir, "*.out"))   #return an iterator of files with ext of *.out in script directory
for file in out_files:
    if os.path.isfile(file):
        shutil.move(file, outpath)  # move to output file directory
#============================================================================