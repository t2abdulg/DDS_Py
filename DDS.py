import numpy as np
import math as m
import neighbor as nval
import toolkit as util
#from mpi4py import MPI

def DDS_serial(objfunc_name,exe_name,modeldir,to_max,DV,sinitial,its,maxiter):
    # ==========================================================================
    # Definitions
    # ==========================================================================
    num_dec = DV['S_min'].shape[0]                             # number of DVs 
    #DV['S_min']                                               # array of DV lower bounds
    #DV['S_max']                                               # array of DV upper bounds
    stest = np.empty(num_dec,dtype=float)                      # candidate solution array
    sbest = np.empty_like(stest)                               # best solution array
    Discrete_flag = DV['Discrete_flag']                        # array of DV types(flt/int)
    S_range = DV['S_max'] - DV['S_min']                        # array of DV ranges
    ileft = maxiter - its                                      # number of iterations
    # its = number of function evaluations to initialize the DDS algorithm
    solution = np.empty((maxiter,num_dec+3),dtype=float)       # solution storage array
    
    # ==========================================================================
    # Initial Solution Processing
    # ==========================================================================  
    for i in range(its):
        if its>1:
            if Discrete_flag.all() == 0:# handling continuous variables  
                # return continuous uniform random samples 
                stest = DV['S_min'] +S_range*np.random.random(num_dec)  
            else: # handling discrete case
                for j in range(0,num_dec):
                    # return random integers from the discrete uniform dist'n
                    stest[j] = np.random.randit([DV['S_min'][j], DV['S_max'][j]],size =1) 
        else: # know its=1, using a user supplied initial solution.
            # get initial solution from the input file
            stest=sinitial

        # Call obj function 
        Jtest = to_max*util.get_objfunc(stest,modeldir,objfunc_name,exe_name,0)  
            
        # Update current best
        if i==0:
            Jbest = Jtest 
        if Jtest <= Jbest:
            Jbest = Jtest 
            np.copyto(sbest,stest) 
                
        # Store initial sol. data in Master solution array    
        solution[i,0] = i 
        solution[i,1] = to_max*Jbest 
        solution[i,2] = to_max*Jtest
        solution[i,3:3+num_dec] = stest
   
    # ==========================================================================
    # Main Algorithm Loop
    # ==========================================================================
    for i in range(ileft):
        # probability of being selected as neighbour
        Pn=1.0-m.log1p(i)/m.log(ileft)  
        # counter for how many decision variables vary in neighbour
        dvn_count=0 
        # define stest initially as current (sbest for greedy)
        np.copyto(stest,sbest)
        # Generate array of random uniformly distributed numbers for neighborhood inclusion
        randnums=np.random.random(num_dec)
        
        for j in range(num_dec):
            # then j th DV selected to vary in neighbour
            if randnums[j]< Pn: 
                dvn_count=dvn_count+1
                stest[j] = nval.perturb_type(sbest[j], DV['S_min'][j], DV['S_max'][j],DV['Discrete_flag'][j])
     
        # no DVs selected at random, so select ONE   
        if dvn_count==0: 
            # which dec var to modify for neighbour 
            dec_var=int(m.floor((num_dec)*np.random.random(1)))
            stest[dec_var] = nval.perturb_type(sbest[dec_var], DV['S_min'][dec_var], DV['S_max'][dec_var],DV['Discrete_flag'][dec_var])
    
        # Get ojective function value
        Jtest = to_max*util.get_objfunc(stest,modeldir,objfunc_name,exe_name,0)

        # Update current best
        if Jtest<=Jbest:
            Jbest = Jtest
            np.copyto(sbest,stest) 
            # iteration number best solution found
            it_sbest=i+its 
        
        # accumulate results in Master output matrix 
        # [col 0: iter # col 1: Fbest col 2: Ftest col 3: param set (xtest)]
        solution[i+its,0]=i+its
        solution[i+its,1]=to_max*Jbest
        solution[i+its,2]=to_max*Jtest
        solution[i+its,3:3+num_dec]=stest

    # Return dict: {Master, best iteration #, best solution, best param set}
    return {'Master':solution,'Best_iter':it_sbest,'Best_sol':sbest,'F_Best':Jbest}


def DDS_MPI(objfunc_name,exe_name,modeldir,to_max,DV,sinitial,its,maxiter,num_slaves):

    return 