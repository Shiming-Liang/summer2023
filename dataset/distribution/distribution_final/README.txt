=================================================================================
 README.txt: instructions on how to use the software distribution accompanying
             the tutorial on performance assessment of stochastic multiobjective
             optimizers by C. Fonseca, J. Knowles, L. Thiele, and E. Zitzler
             given at EMO 2005 in Mexico. The corresponding document is
             available as
 
             @TechReport{KTZ2005a,
  
               author = {J. Knowles and L. Thiele and E. Zitzler},
  
               title = {A Tutorial on the Performance Assessment of Stochastic
 
                       Multiobjective Optimizers},
  
               institution = {Computer Engineering and Networks Laboratory (TIK),

                       Swiss Federal Institute of Technology (ETH) Zurich},
  
               year = {2005},
  
               type = {214},
  
               month = jul
 }

 Eckart Zitzler, 3/2005
 Lothar Thiele, 8/2005
=================================================================================

0. Quick Start
==============

Linux:
  Edit the file run.ksh and set the variable os to "linux" (os="linux");
  run the script run.ksh: ./run.ksh

Solaris:
  Edit the file run.ksh and set the variable os to "solaris" (os="solaris");
  run the script run.ksh: ./run.ksh

Windows:
  Run the batch file run.bat

The script/batch file automatically runs IBEA, NSGA2, and SPEA2 on three
test problems (DTLZ2, KNAPSACK, ZDT6) and writes the output to the
directory runs. Moreover, the epsilon indicator, the hypervolume indicator,
and the R indicator are applied to the resulting outcomes, and the indicator
value samples are analyzed using different statistical tests (see directory
tests). In addition the attainment surfaces are computed for all selectors
on all test problems. Finally, all paris of selectors are compared on all 
test problems using attainment surfaces and dominance ranks. 


1. Contents
===========

This distribution contains the following classes of programs:

- All available tools for testing and comparing different optimizers 
  (selectors) on different test problems.

- Examples of batch files for SOLARIS, LINUX and WINDOWS that call
  selecors and variators for the optimization runs and that call 
  the tools for testing and comparing. These batch files are mainly
  for demonstration purpose and need to be adapted.

- A few optimizers (selectors) and test problems (variators) 
  for demonstration purposes only. Additional ones can be downloaded 
  from the PISA web site.

In particular, the following files and directories are part of the
distribution:

Sample Runs
-----------
runs_sample, tests_sample:
  These directories contain a sample run (runs_sample) and
  the corresponding test results (tests_sample). They are
  produced using the provided batch files where the variable
  'generations' is set to 200 (instead of 2).

Batch Files
-----------
run, compute:
  The batch files run.bat/compute.bat (for Windows) and 
  run.ksh/compute.ksh (for SOLARIS/LINUX) call the programs and
  store the results into the respective files. These batch files are
  examples only and show the main capabilities of the distribution.
  A short discription of their functionality is given in run.bat and
  compute.bat.

Selectors
---------
ibea:
  indicator-based evolutionary algorithm
nsga2:
  nondominated sorting genetic algorithm II
spea2:
  strength pareto evolutionary algorithm 2

Variators
---------
dtlz2:
  test problem by Deb, Thiele, Laumanns, and Zitzler
knapsack:
  multiobjective knapsack problem
zdt6:
  test problem by Zitzler, Deb, and Thiele

Monitor
-------
monitor:
  program that automatically performs multiple runs and stores the outcomes

Testing and Comparing
---------------------
tools:
  programs for postprocessing of generated fronts:
  bound     = calculates different bounds for a file of objective vectors
  normalize = transforms all objective values to the interval [1,2]
  filter    = computes the nondominated front for a file of objective vectors
indicators:
  computation for three different quality indicators:
  eps_ind = unary epsilon indicator
  hyp_ind = unary hypervolume indicator
  r_ind   = R indicators
  dominance-rank = implements the dominance ranking method
statistics:
  statistical testing procedures for comparing indicator value samples:
  fisher-indep  
  fisher-matched 
  kruskal-wallis
  mann-whit
  wilcoxon-sign
attainment:
  programs related the attainment function approach:
  eaf = computes the attainment surface for a set of runs
  eaf-test = realizes the statistical test for two attainment surface distributions

For each of these packages, there exist four directories, namely the source
files (*_c_source) and the binaries for Linux (*_linux), Solaris (*_solaris),
and Windows (*_win). 

The batch files are constructed such that the output of the optimization
runs is written to the directory 'runs', and the output of the indicator programs
and the statistical testing procedures is written to the directory 'tests'.

2. The Statistical Testing Procedures
=====================================

The whole distribution is based on the assumption that SMALLER IS BETTER. This
holds as well for the implementation of the variators, the indicators and
the statistical (one-tailed) non-parametric tests. 

fisher-indep
------------
Implements a nonparametric test for differences between precisely two independent 
samples, as described in W.J.Conover (1999) "Practical Nonparametric Statistics 
(3rd Edition)", Wiley.

This version accepts multiple (more than two) samples. In that case, the test is
carried out for each pair and a warning, which advises the p-values are not
accurate because the samples are no longer independent random samples, is issued.

The output is the p-value of the one-tailed test for rejecting the null 
hypothesis of no difference between each pair of sample populations.

A warning message is output if there are multiple (more than two) samples.

fisher-matched
--------------
Implements a nonparametric test for differences between two paired (or matched) 
samples, as described in W.J.Conover (1999) "Practical Nonparametric Statistics 
(3rd Edition)", Wiley.

This version accepts multiple (more than two) samples. In that case, the test is
carried out for each pair and a warning, which advises the p-values are not
accurate because the samples are no longer independent random samples, is issued.

The output is the one-tailed p-value that the null hypothesis is true for each pair.
If the normal approximation has been used then this is indicated in brackets.

A warning message is output if there are multiple (more than two) samples.

kruskal-wallis
--------------
Implements a nonparametric test for differences between multiple independent 
samples, as described in W.J.Conover (1999) "Practical Nonparametric Statistics 
(3rd Edition)", Wiley.

If and only if a first test for significance of any differences between the
samples is passed, at the given alpha value, then the output will be the 
one-taileed p-values for rejecting a null hypothesis of no significant 
difference between two samples, for each pair-wise combination.

In the case that the first test fails, the output is simply, "H0".

mann-whit
---------
Implements a nonparametric test for differences between precisely two 
independent samples, as described in W.J.Conover (1999) "Practical 
Nonparametric Statistics (3rd Edition)", Wiley. The version here uses 
T1 - the equation to correct for ties in the inputs. However, it does
not compute corrections for small samples. Instead, a warning is 
issued if either sample is smaller than 20.

This version accepts multiple (more than two) samples. In that case, 
the test is carried out for each pair and a warning, which advises 
the p-values are not accurate because the samples are no longer 
independent random samples, is issued.

The output is the p-value of the two-tailed test for rejecting 
the null hypothesis of no difference between each pair of sample populations.

A warning message is output if there are multiple (more than two) samples.

wilcoxon-sign
-------------
Implements a nonparametric test for differences between two paired 
(or matched) samples, as described in W.J.Conover (1999) "Practical 
Nonparametric Statisticsn (3rd Edition)", Wiley.
If the sample size n>=50 then the normal approximation is used. 
If n<=50, the exact quantiles from a table are used. In this case, 
only the lowest from a set of critical p-values that exceeds the ranksum 
is returned: no interpolation of values in the table is used.

This version accepts multiple (more than two) samples. In that case, the 
test is carried out for each pair and a warning, which advises the p-values are not
accurate because the samples are no longer independent random samples, is issued.

The output is the one-tailed p-value that the null hypothesis is true for each pair.
If the normal approximation has been used then this is indicated in brackets.

A warning message is output if there are multiple (more than two) samples.

3. Attainment Surfaces and Tests
================================

eaf
---
Computes all attainment surfaces for a set of runs stored in the data file

and writes them one after the other to the output file. The implementation
works for two-dimensional objective spaces only!

eaf-test
--------
To compare the outcomes of two algorithms A and B, first
 the attainment surfaces

are computed and then a Kolmogorov-Smirnov like test is applied. Again, the
implementation works for two dimensions only.


4. Dominance Ranking
====================

Nondominated ranking testing of two samples, A and B,
of N and M approximation sets are implemented.

The procedure applied uses the program dominance-rank and
the Mann-Whitney rank sum test:

Label the sets A_i, i=1,...,N as X_i, i=1,...,N and the
sets B_j, j=1,...,M as X_i, i=N+1,..., N+M.
Apply the binary epsilon indicator to each possible
ordered pairing of approximation sets <X_i, X_j> and <X_j, X_i>,
for all i,j in 0,..., N+M.

If an ordered pair yields a negative epsilon value (assuming 
use of the additive epsilon indicator), then the second
approximation set is dominated. The dominance rank of every
set is computed in this way and these values are sorted.
The Mann-Whitney rank sum test is then applied to the ranks
(meaning the positions in the sorted list) of the A sample and
the B sample to test if there is a significant difference in the
two samples. The null hypothesis is that the A's and the B's
have the same distribution of dominance ranks. The alternative 
hypethesis is that the A's tend to have a better dominance 
rank than the B's (one-tailed test).

5. Naming Conventions of the Generated Test Files
=================================================

variator.gen 
  The accumulated generated objective vectors for the problem 
  'variator' in generation 'gen' from all selectors and from all runs
variator_bound.gen 
  Different characteristic bounds for the objective
  vectors taken from 'variator.gen'
variator_norm.gen 
  Accumulated generated objective vectors from
  'variator.gen', but normalized to [1,2].
variator_ref.gen 
  Non-dominated front calculated from 'variator_norm.gen',
  i.e. the reference front for the calculuation of indicators

variator_selector_norm.gen 
  Generated objective vectors from all runs of 'selector' 
  with problem 'variator' at generation 'gen',
  but normalized to [1,2].
variator_selector_indicator.gen 
  Indicator values of the 'selector' on problem 'variator' in 
  generation 'gen'. The 'indicator' may be r (for the R indicator), 
  eps (for the epsilon indicator) and hyp (for the hypervolume 
  indicator).
variator_indicator.gen 
  Indicator values of the all selectors on
  problem 'variator', combined in one single file. The 'indicator' 
  may be r (for the R indicator), eps (for the epsilon indicator)
  and hyp (for the hypervolume indicator).
variator_indicator_test.gen 
  Results of the statistical test 'test'
  applied to all combinations of selectors for problem 'variator'.
  The 'indicator' may be r (for the R indicator), 
  eps (for the epsilon indicator) and hyp (for the hypervolume 
  indicator). The 'test' may be  wilcoxon (wilcoxon-sign), 
  kruskal (kruskal-wallis), fisheri (fisher independent) and 
  fisherm (fisher matched) and mann (mann-whit).

variator_selector_eaf.gen 
  All attainment functions of the 'selector' on problem 'variator' in 
  generation 'gen'. There are as many attainment functions as there
  are runs. Each attainment function is characterized by the corresponding
  points.
variator_selectora_selectorb_eaftest.gen
  Results of the statistical test applied to the attainment functions
  of 'selectora' and 'selectorb' on problem 'variator'.
variator_selectora_selectorb_ranktest.gen
  Results of the statistical test applied to the dominance ranking
  of 'selectora' and 'selectorb' on problem 'variator'.
 
6. Your Own Test Bench
======================

a. Adding other algorithms:
   Download the algorithm from the PISA Website (http:www.tik.ee.ethz.ch/pisa/)
   and create a corresponding directory <alg>_<os>, e.g., femo_linux. Add the
   corresponding name, e.g., femo, to the variable selectors in run.bat resp. run.ksh.
    
b. Adding further test problems:
   Download the test problem from the PISA Website (http:www.tik.ee.ethz.ch/pisa/)
   and create a corresponding directory <problem>_<os>, e.g., expo_win. Add the
   corresponding name, e.g., expo, to the variable variators in run.bat resp. run.ksh.
   Create a file PISA_cfg in the directory of the test problem that specifies
   population size and number of objectives.

c. Setting parameters:
   - number of runs: change file monitor_param.txt in directory monitor_* accordingly
   - number of generations: change files monitor_param.txt in directory monitor_* and
     run.bat resp. run.ksh (variable generation) accordingly
   - population size: change file PISA_cfg in the directory of the test problem
     (mu = number of parents, lambda = number of offspring, alpha = population size)
     accordingly

d. Selecting algorithms and test problems:
   Set the variables selectors and variators in the file run.ksh resp. run.bat

e. Run the scripts:
   Execute run.bat resp. run.ksh to perform all runs and statistics. If you only
   want to do the statistics, use "./run.ksh compare" resp. "run.bat compare"
