# fitness_func.py
# =========================================================================================================================
# This module contains a catalogue of generalised D- dimensional well-known, cited optimisation benchmark fitness functions
# to evaluate the optimizer algorithm performance.
# =========================================================================================================================
import math


def Rastrigin(param_space):
	# =====================================================================================================================
	# The Rastrigin function is a multimodal, non-convex function used as a performance test problem for optimisation 
	# algorithms. 
	# It was first proposed by Rastrigin as a 2-dimensional function and has been generalized by Muhlenbein et al.
	# INPUT: param_space = vector of current Decision Variable values; TYPE = list
	# =====================================================================================================================
	fitness = 0
	for x in param_space:
		fitness += float(x**2) - (math.cos(2*math.pi*float(x)))

	return fitness

def Griewank(x):
	# =====================================================================================================================
	# The Griewank function [1981] is a non-linear multimodal function widely used to test the convergence of optimisation 
	# functions
	# INPUT: param_space = vector of current Decision Variable values; TYPE = list
	# =====================================================================================================================
	dimens=len(x)
	term1 = 0
	term2 = 1
	term3 = 1
	for i in range(dimens):
	   term1 += x[i]**2
    	   term2 *= math.cos(x[i]/math.sqrt(i+1))

	fitness = (float(term1)/4000.0)- float(term2) + term3
	return fitness

def Ackley(param_space):
	# =====================================================================================================================
	# The Ackley function [1987] is a non-linear multimodal function widely used to test the convergence of optimisation 
	# functions
	# INPUT: param_space = vector of current Decision Variable values; TYPE = list
	# =====================================================================================================================
	dimens = len(param_space)

	sum1 = 0
	sum2 = 0

	for x in param_space:
		sum1 += x**2
		sum2 += math.cos(2*math.pi*float(x))
	fitness = -20*math.exp(-0.2*math.sqrt(sum1/float(dimens)))-math.exp(sum2/dimens)
	return fitness


