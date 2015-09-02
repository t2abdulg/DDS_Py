# ======================================================================
# Functions to perturb neighborhoud of decision variables to generate 
# new candidate solutions. Perturbation magnitudes are randomly sampled 
# from the standard normal distribution (mean = zero) 
# ======================================================================
import numpy as np
import math as m

def perturb_type(s,s_min,s_max,discrete_flag):
    
    if discrete_flag == 0:
        # perturb continuous variable
        s_new = perturb_cont(s,s_min,s_max)
    else:
        # perturb discrete variable
        s_new = perturb_disc(s,s_min,s_max)
    return s_new

def perturb_cont(s,s_min,s_max):
        
        # Define parameter range
        s_range = s_max - s_min
        # Scalar neighbourhood size perturbation parameter (r) 
        # NOTE: this value is proven to be robust. **DO NOT CHANGE**
        r = 0.2
        # Perturb variable
        z_value = stand_norm() 
        delta = s_range*r*z_value 
        #delta = s_range*r*np.random.randn(1)
        s_new = s + delta
        
        # Handle perturbations outside of decision variable range:
        # Reflect and absorb decision variable at bounds
        
        # probability of absorbing or reflecting at boundary
        P_Abs_or_Ref = np.random.random()
        
        # Case 1) New variable is below lower bound
        if s_new < s_min: # works for any pos or neg s_min
            if P_Abs_or_Ref <= 0.5: # with 50% chance reflect
                s_new = s_min + (s_min - s_new) 
            else: # with 50% chance absorb
                np.copy(s_new,s_min)           
            # if reflection goes past s_max then value should be s_min since without reflection
            # the approach goes way past lower bound.  This keeps X close to lower bound when X current
            # is close to lower bound:
            if s_new > s_max:
                np.copy(s_new,s_min) 

        # Case 2) New variable is above upper bound
        elif s_new > s_max:  #works for any pos or neg s_max
            if P_Abs_or_Ref <= 0.5:  #with 50% chance reflect
                s_new = s_max - (s_new - s_max) 
            else:  # with 50% chance absorb
                np.copy(s_new,s_max)
            # if reflection goes past s_min then value should be s_max for same reasons as above
            if s_new < s_min:
                np.copy(s_new,s_max)

        return s_new

def perturb_disc(s,s_min,s_max):
    # ==================================================== 
    # Function for discrete decision variable perturbation
    # ====================================================        
    # Define parameter range
    s_range = s_max - s_min
    # Scalar neighbourhood size perturbation parameter (r) 
    # NOTE: this value is proven to be robust. **DO NOT CHANGE**
    r = 0.2;
    # Perturb variable
    z_value = stand_norm 
    delta = s_range*r*z_value
    s_new = s + delta
    
    # Handle perturbations outside of decision variable range:
    # Reflect and absorb decision variable at bounds
    
    # probability of absorbing or reflecting at boundary
    P_Abs_or_Ref = np.random.rand(1)
    
    # Case 1) New variable is below lower bound
    if s_new < s_min - 0.5: # works for any pos or neg s_min
        if P_Abs_or_Ref <= 0.5: # with 50% chance reflect
            s_new = (s_min-0.5) + ((s_min-0.5) - s_new) 
        else: # with 50% chance absorb
            s_new = s_min          
        # if reflection goes past s_max+0.5 then value should be s_min since without reflection
        # the approach goes way past lower bound.  This keeps X close to lower bound when X current
        # is close to lower bound:
        if s_new > s_max + 0.5:
            s_new = s_min

    # Case 2) New variable is above upper bound
    elif s_new > s_max + 0.5:  #works for any pos or neg s_max
        if P_Abs_or_Ref <= 0.5:  #with 50% chance reflect
            s_new = (s_max+0.5) - (s_new - (s_max+0.5))
        else:  # with 50% chance absorb
            s_new = s_max
        # if reflection goes past s_min -0.5 then value should be s_max for same reasons as above
        if s_new < s_min - 0.5:
            s_new = s_max
            
    # Round new value to nearest integer
    s_new = np.around(s_new)
    
    # Handle case where new value is the same as current: sample from 
    # uniform distribution
    if s_new == s:
        samp = s_min - 1 + np.ceil(s_range)*np.random.rand()
        if samp < s:
            s_new = samp
        else:
            s_new = samp+1

    return s_new  

def stand_norm():
    # Function returns a standard Gaussian random number (zvalue)  
    # based upon Numerical recipes gasdev and Marsagalia-Bray Algorithm
    Work3=2.0 
    while( (Work3>=1.0) or (Work3==0.0) ):
    # call random_number(ranval) # get one uniform random number
        ranval = np.random.random() #harvest(ign)
        Work1 = 2.0 * ranval - 1.0  #2.0 * DBLE(ranval) - 1.0
    # call random_number(ranval) # get one uniform random number
        ranval = np.random.random() #harvest(ign+1)
        Work2 = 2.0 * ranval - 1.0 # 2.0 * DBLE(ranval) - 1.0
        Work3 = Work1 * Work1 + Work2 * Work2
        # ign = ign + 2

    Work3 = ((-2.0 * m.log(Work3)) / Work3)**0.5  # natural log
        
    # pick one of two deviates at random (don't worry about trying to use both):
    # call random_number(ranval) # get one uniform random number
    ranval = np.random.random() #harvest(ign)
    # ign = ign + 1

    if (ranval < 0.5) : 
        zvalue = Work1 * Work3
    else :
        zvalue = Work2 * Work3

    return zvalue
    