# =============================================================================
# This module contains a collection of utility functions aimed at I/O handling,
# directory management and external simulation model control
# =============================================================================

import numpy as np
import os
import fitness_func as of

def read_param_file(filename):
#===========================================================================
# This function will read the initial parameter range file set by the user 
# The user should set this file manually
# This file should contain 4 columns
# column 0: name of parameter; col 1: lower bound; col 2: upper bound 
# column 3: discrete dec variable? 0 = no; 1 = yes
#===========================================================================
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    bounds_dat = np.loadtxt(filename,dtype={'names':('S_name','S_min','S_max','Discrete_flag'),'formats':('S3','f12','f12','i4')},skiprows = 1)
    return bounds_dat

def read_DDS_inp(filename):
#===========================================================================
# This function will read the DDS Main Input Control File - see file for
# instructions on how to populate
#===========================================================================    
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    A = np.loadtxt(filename, dtype='str', comments = '#',skiprows =2)
    return {'objfunc_name':A[0],'runname':A[1],'num_trials':int(A[2]),'num_iters':int(A[3]),'user_seed':int(A[4]), \
    'out_print':int(A[5]),'ini_name':A[6],'modeldir':A[7],'obj_flag':int(A[8]),'num_slaves':int(A[9]),'pre_empt_flag':int(A[10])}


def ext_function(x,modeldir,exe_name):
#============================================================================ 
# This function enables DDS to optimize external simulation models
# Function interfaces with simulation model or external exe function with DDS
# Follow this coding framework to link DDS with any general *.exe file OR a
# batch file (*.bat) needed to compute your objective function.
#============================================================================
    # STEP 1: switch to model directory
    os.chdir(modeldir)
    
    # STEP 2: write model input file with current decision variables
    np.savetxt('variables_in.txt', x)
    
    # STEP 3: execute model 
    os.system(exe_name)
    
    # STEP 4: read model output
    #  - assumes that model reads 'variables_in', runs and 
    #  - then outputs objective function value to 'function_out'
    y = np.loadtxt('function_out.txt')
    
    # STEP 5: return back to main code directory
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    
    return y

def get_objfunc(x,modeldir,objfunc_name,exe_name,slave_index):
#============================================================================
# Function to handle calls to external objective functions
#============================================================================    
    # Establish script directory
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    # Determine if model subdirectory is available
    if script_dir == modeldir:
        # if objective function is in script directory
        feval = getattr(of, objfunc_name)
        y = feval(x)
    else:
        # else - check if serial or parallel code is running
        if slave_index == 0:   # serial code
            os.chdir(modeldir)
        else:# if parallel code - launch slave directory
            modeldir = modeldir + '_' + str(slave_index)
            os.chdir(modeldir)
        # if objective function is .py script
        if exe_name.size == 0:
            feval = getattr(of, objfunc_name)
            y = feval(x)
            os.chdir(script_dir)
            # else - launch ext_function to handle *.exe/*.bat
        else:
            y = ext_function(x,modeldir,exe_name)
    return y 



class solution:
    def __init__(self, decnum, objnum):
        self.dv = zeros(decnum,float)
        self.f = zeros(objnum,float)
        self.z = 0

   