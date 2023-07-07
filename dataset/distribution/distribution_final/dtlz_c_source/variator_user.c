/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Functions the user must implement
  
  C file.
  
  file: variator_user.h
  author: Marco Laumanns, laumanns@tik.ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
   
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "variator.h"
#include "variator_user.h"

/*--------------------| global variable definitions |-------------------*/

/*==== declared in variator_user.h used in other files as well ====*/

char *log_file = "dtlz_error.log"; /**** Changed for DTLZ.*/

char paramfile[FILE_NAME_LENGTH]; /* file with local parameters */


/*==== only used in this file ====*/

/* local parameters from paramfile*/
char problem[FILE_NAME_LENGTH]; 
int seed;   /* seed for random number generator */
int number_decision_variables; /* length of the binary string */
int maxgen; /* maximum number of generations (stop criterion) */
int gen;
char outfile[FILE_NAME_LENGTH]; /* output file for last population */
double individual_mutation_probability;
double individual_recombination_probability;
double variable_mutation_probability;
double variable_swap_probability;
double variable_recombination_probability;
double eta_mutation;
double eta_recombination;

/*-------------------------| individual |-------------------------------*/

void free_individual(individual *ind) 
/* Frees the memory for one indiviual.

   post: memory for ind is freed
*/
{
     /**********| added for DTLZ |**************/
    
     if (ind == NULL)
          return;
     
     free(ind->x);
     free(ind->f);
     
     /**********| addition for DTLZ end |*******/
     
     free(ind);
}

double get_objective_value(int identity, int i)
/* Gets objective value of an individual.

   pre: 0 <= i <= dimension - 1 (dimension is the number of objectives)

   post: Return value == the objective value number 'i' in individual '*ind'.
         If no individual with ID 'identity' return value == -1. 
*/   
{
     /**********| added for DTLZ |**************/
     individual *temp_ind;
     /**********| addition for DTLZ end |*******/
     
     double objective_value = -1.0;

     assert(0 <= i && i < dimension); /* asserting the pre-condition */
     
     /**********| added for DTLZ |**************/
    
     if (i < 0 || i > (dimension - 1))
	  return(-1);
     
     temp_ind = get_individual(identity);
     if (temp_ind == NULL)
	 return(-1);
     
     objective_value = temp_ind->f[i];
     
     /**********| addition for DTLZ end |*******/
  
     return (objective_value);
}

/*-------------------------| statemachine functions |-------------------*/

int state0() 
/* Do what needs to be done in state 0.

   pre: The global variable 'paramfile' contains the name of the
        parameter file specified on the commandline.
        The global variable 'alpha' contains the number of indiviuals
        you need to generate for the initial population.
                
   post: Optionally read parameter specific for the module.
         Optionally do some initialization.
         Initial population created.
         Information about initial population written to the ini file
         using write_ini().
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for DTLZ |**************/
     int i;
     /**********| addition for DTLZ end |*******/

     
     int result; /* stores return values of called functions */
     int *initial_population; /* storing the IDs of the individuals */
     initial_population = (int *) malloc(alpha * sizeof(int)); 
     if (initial_population == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }

     /**********| added for DTLZ |**************/
     result = read_local_parameters();
     
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't read local parameters");
          return (1);
     }

     /* initializing the first alpha individuals */
     for(i = 0; i < alpha; i++)
     {
 	 individual *ind = new_individual();
	 eval(ind);
	 initial_population[i] = add_individual(ind);
         if(initial_population[i] == -1)
            return(1);
     } 

     gen = 1;
     
     /**********| addition for DTLZ end |*******/

     result = write_ini(initial_population);
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't write ini");
          free(initial_population);
          return (1);
     }

     free(initial_population);
     return (0);
}



int state2()
/* Do what needs to be done in state 2.

   pre: The global variable 'mu' contains the number of indiviuals
        you need to read using 'read_sel()'.
        The global variable 'lambda' contains the number of individuals
        you need to create by variation of the individuals specified the
        'sel' file.
        
   post: Optionally call read_arc() in order to delete old uncessary
         individuals from the global population.
         read_sel() called
         'lambda' children generated from the 'mu' parents
         Children added to the global population using add_individual().
         Information about children written to the 'var' file using
         write_var().
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     int *parent_identities, *offspring_identities; /* array of identities */
     int result; /* stores return values of called functions */

     parent_identities = (int *) malloc(mu * sizeof(int)); 
     if (parent_identities == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }

     offspring_identities = (int *) malloc(lambda * sizeof(int)); 
     if (offspring_identities == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }
     
     result = read_sel(parent_identities);
     if (result != 0) /* if some file reading error occurs, return 2 */
          return (2);

     result = read_arc(); 
     if (result != 0) /* if some file reading error occurs, return 2 */
          return (2);

     /**********| added for DTLZ |**************/

     result = variate(parent_identities, offspring_identities);
     if (result != 0)
          return (1);
          
     gen++;

     /**********| addition for DTLZ end |*******/


     result = write_var(offspring_identities);
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't write var");
          free(offspring_identities);
          free(parent_identities);
          return (1);
     }

     free(offspring_identities);
     free(parent_identities);
     return (0);
}
 

int state4() 
/* Do what needs to be done in state 4.

   pre: State 4 means the variator has to terminate.

   post: Free all memory.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for DTLZ |**************/
     
     int result;
     result = read_arc();

     if (0 == result) /* arc file correctly read
                         this means it was not read before,
                         e.g., in a reset. */
     {
        write_output_file();
     }
     
     /**********| addition for DTLZ end |*******/
     
     return (0);
}


int state7()
/* Do what needs to be done in state 7.

   pre: State 7 means that the selector has just terminated.

   post: You probably don't need to do anything, just return 0.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     return(0);  
}


int state8()
/* Do what needs to be done in state 8.

   pre: State 8 means that the variator needs to reset and get ready to
        start again in state 0.

   post: Get ready to start again in state 0. 
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for DTLZ |**************/

   int result;
   
   gen = 1;
     
   result = read_arc();

   if (0 == result) /* arc file correctly read
                       this means it was not read before */
   {
      write_output_file();
   }
   
     /**********| addition for DTLZ end |*******/
     
     return (0);
}


int state11()
/* Do what needs to be done in state 11.

   pre: State 11 means that the selector has just reset and is ready
        to start again in state 1.

   post: You probably don't need to do anything, just return 0.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     return (0);  
}


int is_finished()
/* Tests if ending criterion of your algorithm applies.

   post: return value == 1 if optimization should stop
         return value == 0 if optimization should continue
*/
{
     /**********| added for DTLZ |**************/
     return (gen >= maxgen);
     /**********| addition for DTLZ end |*******/
}


/**********| added for DTLZ |**************/

int read_local_parameters()
{
     FILE *fp;
     char str[CFG_NAME_LENGTH];

     /* reading parameter file with parameters for selection */
     fp = fopen(paramfile, "r"); 
     assert(fp != NULL);

     if(dimension < 0)
     {
          log_to_file(log_file, __FILE__, 
                      __LINE__, "can't handle that dimension");
          return(1);
     } 

     if(mu != lambda)
     {
          log_to_file(log_file, __FILE__, 
                      __LINE__, "can't handle mu != lambda");
          return(1);
     }

     fscanf(fp, "%s", str);
     assert(strcmp(str, "problem") == 0);
     fscanf(fp, "%s", problem); /* fscanf() returns EOF if
                                   reading failed. */

     fscanf(fp, "%s", str);
     assert(strcmp(str, "seed") == 0);
     fscanf(fp, "%d", &seed);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "number_decision_variables") == 0);
     fscanf(fp, "%d", &number_decision_variables);
     assert(number_decision_variables >= dimension);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "maxgen") == 0);
     fscanf(fp, "%d", &maxgen);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "outputfile") == 0);
     fscanf(fp, "%s", outfile); /* fscanf() returns EOF if
                                   reading failed. */

     fscanf(fp, "%s", str);
     assert(strcmp(str, "individual_mutation_probability") == 0);
     fscanf(fp, "%le", &individual_mutation_probability);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "individual_recombination_probability") == 0);
     fscanf(fp, "%le", &individual_recombination_probability);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "variable_mutation_probability") == 0);
     fscanf(fp, "%le", &variable_mutation_probability);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "variable_swap_probability") == 0);
     fscanf(fp, "%le", &variable_swap_probability);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "variable_recombination_probability") == 0);
     fscanf(fp, "%le", &variable_recombination_probability);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "eta_mutation") == 0);
     fscanf(fp, "%le", &eta_mutation);
     assert(eta_mutation >= 0);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "eta_recombination") == 0);
     fscanf(fp, "%le", &eta_recombination);

     srand(seed); /* seeding random number generator */

     fclose(fp);

     return (0);
}


/* Performs variation. */
int variate(int *selected, int *result_ids)
{
     int result, i, k;

     result = 1;

     /* copying all individuals from selected */
     for(i = 0; i < mu; i++)
     {
          result_ids[i] = 
               add_individual(copy_individual(get_individual(selected[i])));
          if(result_ids[i] == -1)
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "copying + adding failed");
               return (1);
          }
     }
 
     /* if odd number of individuals, last one is
        left as is */
     if((((double)mu/2) - (int)(mu/2)) != 0) k = mu - 1; 
     else k = mu;

     /* do recombination */
     for(i = 0; i < k; i+= 2)
     {  
          if (drand(1) <= individual_recombination_probability)
          {
               if (variable_swap_probability > 0)
               {
                    result = uniform_crossover
                      (get_individual(result_ids[i]),
                       get_individual(result_ids[i + 1]));
                    if (result != 0)
                         log_to_file(log_file, __FILE__, 
                                     __LINE__, "recombination failed!");
               }

               if (variable_recombination_probability > 0)
               {
                    result = sbx
                      (get_individual(result_ids[i]), 
                       get_individual(result_ids[i + 1]));
                    if (result != 0)
                         log_to_file(log_file, __FILE__, 
                                     __LINE__, "recombination failed!");
               }
          }
     }
     
     /* do mutation */
     for(i = 0; i < mu; i++)
     {
          if (drand(1) <= individual_mutation_probability)
          { 
               if (variable_mutation_probability > 0)
               {
                    result = mutation(get_individual(result_ids[i]));
                    if(result != 0)
                      log_to_file(log_file, __FILE__, __LINE__,
                                  "mutation failed!");
               }
          }
     }
     
     /* do evaluation */
     for(i = 0; i < mu; i++)
     {
          int result;
          result = eval(get_individual(result_ids[i]));
     }
     
     return (0);
}


int mutation(individual *ind)
{
     int i;

     if (ind == NULL)
     {
	 return (1);
     }
     
     for (i = 0; i < ind->n; i++)
     {
	 if (drand(1) <= variable_mutation_probability)
	 {
	     double eta = eta_mutation;
	     double u = drand(1.0);
	     double delta = 0;
	     double x = ind->x[i];
	     double lb = 0;    /* lower bound of variable i */
	     double ub = 1;    /* upper bound of variable i */
	     double diff = ub - lb;  /* range of variable i */
	     double maxmut0 = x - lb;
	     double maxmut = ub - x;
	     double delta_max = maxmut0 / diff;
	     if (maxmut0 > maxmut)
	     {
		 delta_max = maxmut / diff;
	     }
	     
	     if (u < 0.5)
	     {
		 double b =  2*u + (1-2*u)*(pow(1-delta_max,(eta+1)));
		 delta = pow(b,(1.0/(eta+1))) - 1.0;
	     }
	     else
	     {
		 double b = 2*(1-u) + 2*(u-0.5)*(pow(1-delta_max,(eta+1)));
		 delta = 1.0 - pow(b,(1.0/(eta+1)));
	     }
	     if (delta > delta_max)  /* machine accuracy problem */
		 delta = delta_max;
	     else if (delta < -delta_max)
		 delta = -delta_max;
	     
	     ind->x[i] = x + delta * diff;
	 }
     }
     
     return (0);
}



int uniform_crossover(individual *ind1, individual *ind2)
{
     int i;
   
     for (i = 0; i < ind2->n; i++)
     {
	 if (drand(1) <= variable_swap_probability) /* switch variable */
	 {
	     double x = ind2->x[i];
	     ind2->x[i] = ind1->x[i];
	     ind1->x[i] = x;
          } 
     }  

     return (0);
}



int sbx(individual *ind1, individual *ind2)
{
     int i;
   
     for (i = 0; i < ind2->n; i++)
     {
	 if (drand(1) <= variable_recombination_probability)  
	 {
	     double di = eta_recombination; /* distribution index */
	     int bounded = 1;
	     double lb = 0;    /* lower bound of variable i */
	     double ub = 1;    /* upper bound of variable i */	     
	     double u = drand(1);
	     double b0=0, b1=0;   /* spread factors */
	     double x0 = ind1->x[i];
	     double x1 = ind2->x[i];
	     double bl=0, bu=0, p_bl=0, p_bu=0, bll=0, buu=0, blll=0, buuu=0;
	     double dx = 0;
	     double u0=0, u1=0;
             
            /* calculate spread factor(s) b0, b1 */ 
            if (bounded == 1)
            {
                dx = fabs(x1-x0);   /* difference of x values */
                if (dx > 0)
                {
                    bl = (x0 + x1 - 2*lb) / dx;
                    bu = (2*ub - x0 - x1) / dx;
                    bll = (x0 + x1 - 2*(x0-lb)) / dx;
                    buu = (2*(ub-x1)-x0-x1) / dx;
                    if (x0 < x1)
                    {
                        blll = 1 + 2 * (x0 - lb) / dx;
                        buuu = 1 + 2 * (ub - x1) / dx;
                    }
                    else
                    {
                        blll = 1 + 2 * (x1 - lb) / dx;
                        buuu = 1 + 2 * (ub-x0) / dx;
		    }
		    
		    bl = blll; /* take Deb's version (numerically stable) */
                    bu = buuu;
                    if (bl < bu)  /* symmetric distribution, like Deb */
                        bu = bl;
                    else
                        bl = bu;
                    assert(bl > 0 && bu > 0);
                    p_bl = 1 - 1/(2*pow(bl,di+1));
                    p_bu = 1 - 1/(2*pow(bu,di+1));
		}
                else
                {
                    p_bl = 1;
                    p_bu = 1;
                }
                u0 = u*p_bl;
                u1 = u*p_bu;
                if (u0<=0.5)
                    b0 = pow(2*u0,1/(di+1));
                else
                    b0 = pow(0.5/(1-u0),1/(di+1));
                if (u1<=0.5)
                    b1 = pow(2*u1,1/(di+1));
                else
                    b1 = pow(0.5/(1-u1),1/(di+1));
                assert(dx==0 || (b0<=bl && b1<=bu)); /* machine accuracy */
            }
            else
            {
                if (u<=0.5)
                    b0 = pow(2*u,1/(di+1));
                else
                    b0 = pow(0.5/(1-u),1/(di+1));
                b1 = b0;
            }
            assert (b0 == b1);
            if (x0 < x1)
            {
                ind1->x[i] = 0.5*(x0+x1 + b0*(x0-x1));
                ind2->x[i] = 0.5*(x0+x1 + b1*(x1-x0));
            }
            else
            {
                ind1->x[i] = 0.5*(x0+x1 + b1*(x0-x1));
                ind2->x[i] = 0.5*(x0+x1 + b0*(x1-x0));
            }
	 }
     }  
     
     return (0);
}



/* Generate a random integer. */
int irand(int range)
{
     int j;
     j=(int) ((double) range * (double) rand() / (RAND_MAX + 1.0));
     return (j);
}


/* Generate a random double. */
double drand(double range)
{
     double j;
     j=(range * (double) rand() / (RAND_MAX + 1.0));
     return (j);
}

/* Determines the objective value based on DTLZ */
int eval(individual *ind)
{
    if (strcmp(problem, "DTLZ1") == 0)
    {
	return (eval_DTLZ1(ind));
    }
    
    if (strcmp(problem, "DTLZ2") == 0)
    {
	return (eval_DTLZ2(ind));
    }
    
    if (strcmp(problem, "DTLZ3") == 0)
    {
	return (eval_DTLZ3(ind));
    }
    
    if (strcmp(problem, "DTLZ4") == 0)
    {
	return (eval_DTLZ4(ind));
    }

    if (strcmp(problem, "DTLZ5") == 0)
    {
	return (eval_DTLZ5(ind));
    }

    if (strcmp(problem, "DTLZ6") == 0)
    {
	return (eval_DTLZ6(ind));
    }

    if (strcmp(problem, "DTLZ7") == 0)
    {
	return (eval_DTLZ7(ind));
    }

    if (strcmp(problem, "COMET") == 0)
    {
	return (eval_COMET(ind));
    }

    if (strcmp(problem, "ZDT1") == 0)
    {
	return (eval_ZDT1(ind));
    }

    if (strcmp(problem, "ZDT2") == 0)
    {
	return (eval_ZDT2(ind));
    }

    if (strcmp(problem, "ZDT3") == 0)
    {
	return (eval_ZDT3(ind));
    }

    if (strcmp(problem, "ZDT4") == 0)
    {
	return (eval_ZDT4(ind));
    }

    if (strcmp(problem, "ZDT6") == 0)
    {
	return (eval_ZDT6(ind));
    }
    
    if (strcmp(problem, "SPHERE") == 0)
    {
	return (eval_SPHERE(ind));
    }
    
    if (strcmp(problem, "KUR") == 0)
    {
	return (eval_KUR(ind));
    }
    
    if (strcmp(problem, "QV") == 0)
    {
	return (eval_QV(ind));
    }

    log_to_file(log_file, __FILE__, __LINE__, "unknown problem specified");
    return (1);
}

int eval_DTLZ1(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    
    double g = 0;
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1]-0.5,2) - cos(20 * PISA_PI * (ind->x[i-1]-0.5));
    }
    g = 100 * (k + g);
    
    for (i = 1; i <= dimension; i++)
    {
	double f = 0.5 * (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= ind->x[j-1];
	}
	if (i > 1)
	{
	    f *= 1 - ind->x[(dimension - i + 1) - 1];
	}

	ind->f[i-1] = f;
    }

    return(0);
}

int eval_DTLZ2(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    
    double g = 0;
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1]-0.5,2);
    }
    
    for (i = 1; i <= dimension; i++)
    {
	double f = (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= cos(ind->x[j-1] * PISA_PI / 2);
	}
	if (i > 1)
	{
	    f *= sin(ind->x[(dimension - i + 1) - 1] * PISA_PI / 2);
	}

	ind->f[i-1] = f;
    }

    return(0);
}



int eval_DTLZ3(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    
    double g = 0;
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1]-0.5,2) - cos(20 * PISA_PI * (ind->x[i-1]-0.5));
    }
    g = 100 * (k + g);
    
    for (i = 1; i <= dimension; i++)
    {
	double f = (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= cos(ind->x[j-1] * PISA_PI / 2);
	}
	if (i > 1)
	{
	    f *= sin(ind->x[(dimension - i + 1) - 1] * PISA_PI / 2);
	}

	ind->f[i-1] = f;
    }

    return(0);
}

int eval_DTLZ4(individual *ind)
{    
    int i = 0;
    int j = 0;
    double alpha = 100;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    
    double g = 0;
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1]-0.5,2);
    }
    
    for (i = 1; i <= dimension; i++)
    {
	double f = (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= cos(pow(ind->x[j-1],alpha) * PISA_PI / 2);
	}
	if (i > 1)
	{
	    f *= sin(pow(ind->x[(dimension - i + 1) - 1],alpha) * PISA_PI / 2);
	}

	ind->f[i-1] = f;
    }

    return(0);
}

int eval_DTLZ5(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    double *theta = malloc(dimension * sizeof(double));
    double t = 0;
    double g = 0;
    
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1] - 0.5, 2);
    }

    t = PISA_PI / (4 * (1 + g));
    theta[0] = ind->x[0] * PISA_PI / 2;
    for (i = 2; i <= dimension - 1; i++)
    {
	theta[i-1] = t * (1 + 2 * g * ind->x[i-1]);
    }
    
    for (i = 1; i <= dimension; i++)
    {
	double f = (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= cos(theta[j-1]);
	}
	if (i > 1)
	{
	    f *= sin(theta[(dimension - i + 1) - 1]);
	}

	ind->f[i-1] = f;
    }

    free(theta);
    return(0);
}

int eval_DTLZ6(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    double *theta = malloc(dimension * sizeof(double));
    double t = 0;
    double g = 0;
    
    for (i = n - k + 1; i <= n; i++)
    {
	g += pow(ind->x[i-1], 0.1);
    }

    t = PISA_PI / (4 * (1 + g));
    theta[0] = ind->x[0] * PISA_PI / 2;
    for (i = 2; i <= dimension - 1; i++)
    {
	theta[i-1] = t * (1 + 2 * g * ind->x[i-1]);
    }
    
    for (i = 1; i <= dimension; i++)
    {
	double f = (1 + g);
	for (j = dimension - i; j >= 1; j--)
	{
	    f *= cos(theta[j-1]);
	}
	if (i > 1)
	{
	    f *= sin(theta[(dimension - i + 1) - 1]);
	}

	ind->f[i-1] = f;
    }

    free(theta);
    return(0);
}

int eval_DTLZ7(individual *ind)
{    
    int i = 0;
    int j = 0;
    int n = number_decision_variables;
    int k = n - dimension + 1;
    double g = 0;
    double h = 0;
    
    for (i = n - k + 1; i <= n; i++)
    {
	g += ind->x[i-1];
    }
    g = 1 + 9 * g / k;

    for (i = 1; i <= dimension - 1; i++)
    {
	ind->f[i-1] = ind->x[i-1];
    }
	
    for (j = 1; j <= dimension - 1; j++)
    {
	h += ind->x[j-1] / (1 + g) * (1 + sin(3 * PISA_PI * ind->x[j-1]));
    }
    h = dimension - h;
    ind->f[dimension - 1] = (1 + g) * h;

    return(0);
}

int eval_COMET(individual *ind)
{
    double x1;
    double x2;
    double x3;
    double g;
    
    assert(number_decision_variables == 3);
    assert(dimension == 3);
    
    x1 = 1 + (ind->x[0] * 2.5);
    x2 = -2 + (ind->x[1] * 4);
    x3 = ind->x[2];

    g = x3;

    ind->f[0] = (1 + g) * (pow(x1,3) * pow(x2,2) - 10 * x1 - 4 * x2);
    ind->f[1] = (1 + g) * (pow(x1,3) * pow(x2,2) - 10 * x1 + 4 * x2);
    ind->f[2] = 3 * (1 + g) * pow(x1,2);

    ind->f[0] = ind->f[0] + 100;
    ind->f[1] = ind->f[1] + 100;
    ind->f[2] = ind->f[2];
    
    
    return(0);
}

int eval_ZDT1(individual *ind)
{    
    int i = 0;
    int n = number_decision_variables;
    double f1 = 0;
    double g = 0;
    double h = 0;

    assert(dimension == 2);
    assert(number_decision_variables >= 2);

    f1 = ind->x[0];

    for (i = 1; i < n; i++)
    {
	g += ind->x[i];
    }
    g = 1 + 9 * g / (n-1);
    h = 1 - sqrt(f1 / g);

    ind->f[0] = f1;
    ind->f[1] = g * h;

    return(0);
}

int eval_ZDT2(individual *ind)
{    
    int i = 0;
    int n = number_decision_variables;
    double f1 = 0;
    double g = 0;
    double h = 0;

    assert(dimension == 2);
    assert(number_decision_variables >= 2);

    f1 = ind->x[0];

    for (i = 1; i < n; i++)
    {
	g += ind->x[i];
    }
    g = 1 + 9 * g / (n-1);
    h = 1 - pow(f1 / g, 2);

    ind->f[0] = f1;
    ind->f[1] = g * h;

    return(0);
}

int eval_ZDT3(individual *ind)
{    
    int i = 0;
    int n = number_decision_variables;
    double f1 = 0;
    double g = 0;
    double h = 0;

    assert(dimension == 2);
    assert(number_decision_variables >= 2);

    f1 = ind->x[0];

    for (i = 1; i < n; i++)
    {
	g += ind->x[i];
    }
    g = 1 + 9 * g / (n-1);
    h = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * PISA_PI * f1);

    ind->f[0] = f1;
    ind->f[1] = g * h + 1;

    return(0);
}

int eval_ZDT4(individual *ind)
{    
    int i = 0;
    int n = number_decision_variables;
    double f1 = 0;
    double g = 0;
    double h = 0;

    assert(dimension == 2);
    assert(number_decision_variables >= 2);

    f1 = ind->x[0];

    for (i = 1; i < n; i++)
    {
	double x = ind->x[i];
	g += x * x - 10 * cos(4 * PISA_PI * x);
    }
    g = 1 + 10 * (n - 1) + g;
    h = 1 - sqrt(f1 / g);

    ind->f[0] = f1;
    ind->f[1] = g * h;

    return(0);
}

int eval_ZDT6(individual *ind)
{    
    int i = 0;
    int n = number_decision_variables;
    double f1 = 0;
    double g = 0;
    double h = 0;

    assert(dimension == 2);
    assert(number_decision_variables >= 2);

    f1 = 1 - exp(-4 * ind->x[0]) * pow(sin(6 * PISA_PI * ind->x[0]), 6);

    for (i = 1; i < n; i++)
    {
	g += ind->x[i];
    }
    g = 1 + 9 * pow(g / (n-1), 0.25);
    h = 1 - pow(f1 / g, 2);

    ind->f[0] = f1;
    ind->f[1] = g * h;

    return(0);
}

int eval_SPHERE(individual *ind)
{    
    int i, j;
    int n = number_decision_variables;
    int m = dimension;

    for (j = 0; j < m; j++)
    {
        double f = 0.0;
        for (i = 0; i < n; i++)
        {
            double x = -1000 + 2000 * ind->x[(i + j) % n];
            if (i == 0)
	    {
                x = x - 1;
	    }
            f += x * x;
        }
        ind->f[j] = f;
    }

    return(0);
}

int eval_KUR(individual *ind)
{    
    int i;
    int n = number_decision_variables;
    double f = 0;
    
    assert(dimension == 2);
    
    for (i = 0; i < n; i++)
    {
	double x = -10 + 20 * ind->x[i];
        f += pow(fabs(x), 0.8) + 5 * pow(sin(x), 3) + 3.5828;
    }

    ind->f[0] = f;
    
    f = 0;
    for (i = 0; i < n-1; i++)
    {
	double x = -10 + 20 * ind->x[i];
	double x1 = -10 + 20 * ind->x[i+1];
	f += 1 - exp(-0.2 * sqrt(pow(x, 2) + pow(x1, 2)));
    }

    ind->f[1] = f;

    return(0);
}

int eval_QV(individual *ind)
{    
    int i;
    int n = number_decision_variables;
    double F1 = 0;
    double F2 = 0;
    
    assert(dimension == 2);
    
    for (i = 0; i < n; i++)
    {
        double x = -5 + 10 * ind->x[i];
        F1 += (x)*(x) - 10*cos(2*PISA_PI*(x)) + 10;
        F2 += (x-1.5)*(x-1.5) - 10*cos(2*PISA_PI*(x-1.5)) + 10;
    }
    F1 = pow((F1/n),0.25);
    F2 = pow((F2/n),0.25);

    ind->f[0] = F1;
    ind->f[1] = F2;

    return(0);
}

/* create a random new individual and allocate memory for it,
   returns a pointer to the new individual */
individual *new_individual()
{
     individual *return_ind;
     int i;

     return_ind = (individual *) malloc(sizeof(individual));
     return_ind->x = (double *) malloc(sizeof(double) * number_decision_variables);
     return_ind->f = (double *) malloc(sizeof(double) * dimension);
 
     for (i = 0; i < number_decision_variables; i++)
     {
	 return_ind->x[i] = drand(1.0);
     }
     
     return_ind->n = number_decision_variables;

     for (i = 0; i < dimension; i++)
     {
	 return_ind->f[i] = 0;
     }
     
     return (return_ind);
}


/* copy an individual and return the pointer to it */
individual *copy_individual(individual *ind)
{
     individual *return_ind;
     int i;

     return_ind = (individual *) malloc(sizeof(individual));
     return_ind->x = (double *)malloc(sizeof(double) * number_decision_variables);
     return_ind->f = (double *) malloc(sizeof(double) * dimension);
     
     for (i = 0; i < number_decision_variables; i++)
          return_ind->x[i] = ind->x[i];

     for (i = 0; i < dimension; i++)
	  return_ind->f[i] = ind->f[i];

     return_ind->n = ind->n;

     return(return_ind);
}

/* Writes the index, objective values and bit string of
   all individuals in global_population to 'out_filename'. */
void write_output_file()
{
     int j, current_id;
     FILE *fp_out;
     individual *temp;
     
     fp_out = fopen(outfile, "w");
     assert(fp_out != NULL);

     current_id = get_first();

     while (current_id != -1)
     {       
	  temp = get_individual(current_id);
          fprintf(fp_out, "%d ", current_id); /* write index */
	  for (j = 0; j < dimension; j++)
	  {
	      fprintf(fp_out, "%f ", temp->f[j]);
	  }
          
	  for (j = 0; j < temp->n; j++)
          {
               fprintf(fp_out, "%f ", temp->x[j]);
          }
          fprintf(fp_out, "\n");
	  current_id = get_next(current_id);
     }

     fclose(fp_out);
}


/**********| addition for DTLZ end |*******/
