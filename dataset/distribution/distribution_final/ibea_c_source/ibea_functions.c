/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
  ========================================================================
  IBEA - Indicator-Based EA
  
  Implements most functions.
  
  file: ibea_functions.c
  authors: Eckart Zitzler, zitzler@tik.ee.ethz.ch
           Marco Laumanns, laumanns@tik.ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
  ========================================================================
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "ibea.h"

/* common parameters */
int alpha;  /* number of individuals in initial population */
int mu;     /* number of individuals selected as parents */
int lambda; /* number of offspring individuals */
int dim;    /* number of objectives */


/* local parameters from paramfile*/
int seed;   /* seed for random number generator */
int tournament;  /* parameter for tournament selection */
int indicator;  /* type of indicator used for fitness calculation */
double kappa;  /* scaling factor for fitness calculation */
double rho;  /* determines the reference point for the hypervolume indicator */


/* other variables */
char cfgfile[FILE_NAME_LENGTH];  /* 'cfg' file (common parameters) */
char inifile[FILE_NAME_LENGTH];  /* 'ini' file (initial population) */
char selfile[FILE_NAME_LENGTH];  /* 'sel' file (parents) */
char arcfile[FILE_NAME_LENGTH];  /* 'arc' file (archive) */
char varfile[FILE_NAME_LENGTH];  /* 'var' file (offspring) */


/* population containers */
pop *pp_all = NULL;
pop *pp_new = NULL;
pop *pp_sel = NULL;


/* IBEA internal global variables */
typedef struct range_st
{
    double min;
    double max;
} range;

range *bounds;
double **fitcomp;



/*-----------------------| initialization |------------------------------*/

void initialize(char *paramfile, char *filenamebase)
/* Performs the necessary initialization to start in state 0. */
{
    FILE *fp;
    int result;
    char str[CFG_ENTRY_LENGTH];
    
    /* reading parameter file with parameters for selection */
    fp = fopen(paramfile, "r");
    assert(fp != NULL);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "seed") == 0);
    result = fscanf(fp, "%d", &seed);

    fscanf(fp, "%s", str);
    assert(strcmp(str, "tournament") == 0);
    result = fscanf(fp, "%d", &tournament);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "indicator") == 0);
    result = fscanf(fp, "%d", &indicator);

    fscanf(fp, "%s", str);
    assert(strcmp(str, "kappa") == 0);
    result = fscanf(fp, "%lf", &kappa);

    fscanf(fp, "%s", str);
    assert(strcmp(str, "rho") == 0);
    result = fscanf(fp, "%lf", &rho);   /* fscanf() returns EOF
					   if reading fails. */  
    assert(result != EOF); /* no EOF, parameters correctly read */
    
    fclose(fp);
    
    srand(seed); /* seeding random number generator */
    
    sprintf(varfile, "%svar", filenamebase);
    sprintf(selfile, "%ssel", filenamebase);
    sprintf(cfgfile, "%scfg", filenamebase);
    sprintf(inifile, "%sini", filenamebase);
    sprintf(arcfile, "%sarc", filenamebase);
    
    /* reading cfg file with common configurations for both parts */
    fp = fopen(cfgfile, "r");
    assert(fp != NULL);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "alpha") == 0);
    fscanf(fp, "%d", &alpha);
    assert(alpha > 0);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "mu") == 0);
    fscanf(fp, "%d", &mu);
    assert(mu > 0);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "lambda") == 0);
    fscanf(fp, "%d", &lambda);
    assert(lambda > 0);
    
    fscanf(fp, "%s", str);
    assert(strcmp(str, "dim") == 0);
    result = fscanf(fp, "%d", &dim);
    assert(result != EOF); /* no EOF, 'dim' correctly read */
    assert(dim > 0);
    
    fclose(fp);
    
    /* create individual and archive pop */
    pp_all = create_pop(alpha + lambda, dim);
    pp_sel = create_pop(mu, dim);
}



/*-------------------| memory allocation functions |---------------------*/

void* chk_malloc(size_t size)
/* Wrapper function for malloc(). Checks for failed allocations. */
{
    void *return_value = malloc(size);
    if(return_value == NULL)
	PISA_ERROR("Selector: Out of memory.");
    return (return_value);
}


pop* create_pop(int maxsize, int dim)
/* Allocates memory for a population. */
{
    int i;
    pop *pp;
    
    assert(dim >= 0);
    assert(maxsize >= 0);
    
    pp = (pop*) chk_malloc(sizeof(pop));
    pp->size = 0;
    pp->maxsize = maxsize;
    pp->ind_array = (ind**) chk_malloc(maxsize * sizeof(ind*));
    
    for (i = 0; i < maxsize; i++)
	pp->ind_array[i] = NULL;
    
    return (pp);
}


ind* create_ind(int dim)
/* Allocates memory for one individual. */
{
    ind *p_ind;
    
    assert(dim >= 0);
    
    p_ind = (ind*) chk_malloc(sizeof(ind));
    
    p_ind->index = -1;
    p_ind->fitness = -1;
    p_ind->f = (double*) chk_malloc(dim * sizeof(double));
    return (p_ind);
}


void free_memory()
/* Frees all memory. */
{
    free_pop(pp_sel);
    complete_free_pop(pp_all);
    free_pop(pp_new);
    pp_sel = NULL;
    pp_all = NULL;
    pp_new = NULL;
}

void free_pop(pop *pp)
/* Frees memory for given population. */
{
   if (pp != NULL)
   {
      free(pp->ind_array);
      free(pp);
   }
}

void complete_free_pop(pop *pp)
/* Frees memory for given population and for all individuals in the
   population. */
{
   int i = 0;
   if (pp != NULL)
   {
      if(pp->ind_array != NULL)
      {
         for (i = 0; i < pp->size; i++)
         {
            if (pp->ind_array[i] != NULL)
            {
               free_ind(pp->ind_array[i]);
               pp->ind_array[i] = NULL;
            }
         }
         
         free(pp->ind_array);
      }
   
      free(pp);
   }
}

void free_ind(ind *p_ind)
/* Frees memory for given individual. */
{
   assert(p_ind != NULL);
   
   free(p_ind->f);
   free(p_ind);
}



/*-----------------------| selection functions|--------------------------*/

void selection()
{
    int i;
    int size;

    /* Join offspring individuals from variator to population */
    mergeOffspring();
    
    size = pp_all->size;

    /* Matrices */
    bounds = (range*) chk_malloc(dim * sizeof(range));
    fitcomp = (double**) chk_malloc(size * sizeof(double*));
    for (i = 0; i < size; i++)
	fitcomp[i] = (double*) chk_malloc(size * sizeof(double));	  
    

    /* Calculates fitness values */
    calcBounds();
    calcFitnessComponents();
    calcFitnesses();

    /* Performs environmental selection
       (truncates 'pp_all' to size 'alpha') */
    environmentalSelection();

    /* Performs mating selection
       (fills mating pool / offspring population pp_sel */
    matingSelection();
    
    /* Frees memory of internal data structures */    
    free(bounds);
    bounds = NULL;
    for (i = 0; i < size; i++)
    {
       if (NULL != fitcomp)
          free(fitcomp[i]);
    }
    free(fitcomp);
    fitcomp = NULL;
    
    return;
}


void mergeOffspring()
{
    int i;
    
    assert(pp_all->size + pp_new->size <= pp_all->maxsize);
    
    for (i = 0; i < pp_new->size; i++)
    {
	pp_all->ind_array[pp_all->size + i] = pp_new->ind_array[i];
    }
    
    pp_all->size += pp_new->size;
    
    free_pop(pp_new);
    pp_new = NULL;
}


void calcBounds()
{
    int i, j;
    int size;
    
    size = pp_all->size;

    for (i = 0; i < dim; i++)
    {
	bounds[i].min = pp_all->ind_array[0]->f[i];
	bounds[i].max = pp_all->ind_array[0]->f[i];
    }
    for (i = 1; i < size; i++)
    {
	for (j = 0; j < dim; j++)
	{
	    if (bounds[j].min > pp_all->ind_array[i]->f[j])
		bounds[j].min = pp_all->ind_array[i]->f[j];
	    if (bounds[j].max < pp_all->ind_array[i]->f[j])
		bounds[j].max = pp_all->ind_array[i]->f[j];
	}
    }
    
    return;
}


void calcFitnessComponents()
{
    double maxAbsIndicatorValue = 0;
    int i, j;
    int size = pp_all->size;

    /* determine indicator values and their maximum */
    for (i = 0; i < size; i++)
    {
	for (j = 0; j < size; j++)
	{
	    fitcomp[i][j] = calcIndicatorValue(pp_all->ind_array[i],
					       pp_all->ind_array[j]);
	    if (maxAbsIndicatorValue < fabs(fitcomp[i][j]))
		maxAbsIndicatorValue = fabs(fitcomp[i][j]);
	}
    }
    /* calculate for each pair of invidiuals the corresponding fitness
       component */
    for (i = 0; i < size; i++)    
    {
	for (j = 0; j < size; j++)
	    fitcomp[i][j] = exp((-fitcomp[i][j]/maxAbsIndicatorValue)/kappa);
    }

    return;
}


void calcFitnesses()
{
    int i, j;
    int size;
    double sum;
    
    size = pp_all->size;
    
    for (i = 0; i < size; i++)
    {
        sum = 0;
        for (j = 0; j < size; j++)
	    if (i != j)
		sum += fitcomp[j][i];
        pp_all->ind_array[i]->fitness = sum;
    }
    
    return;
}


void environmentalSelection()
{
    int i, j, worst;
    int new_size = 0;

    for (i = pp_all->size - alpha; i > 0; i--)
    {
        for (j = 0; j < pp_all->size && pp_all->ind_array[j] == NULL; j++);
	worst = j;
	assert(worst < pp_all->size && pp_all->ind_array[worst] != NULL);
        for (j = j + 1; j < pp_all->size; j++)
        {
            if (pp_all->ind_array[j] != NULL)
	    {
		if (pp_all->ind_array[j]->fitness >
		    pp_all->ind_array[worst]->fitness)
		    worst = j;
            }
        }

        for (j = 0; j < pp_all->size; j++)
	    if (pp_all->ind_array[j] != NULL)
		pp_all->ind_array[j]->fitness -= fitcomp[worst][j];
	
        free_ind(pp_all->ind_array[worst]);       
        pp_all->ind_array[worst] = NULL;
    }

    /* Move remaining individuals to top of array in 'pp_all' */
    for (i = 0; i < pp_all->size; i++)
    {
	ind* temp_ind = pp_all->ind_array[i];
	if (temp_ind != NULL)
	{
	    pp_all->ind_array[i] = NULL;
	    pp_all->ind_array[new_size] = temp_ind;
	    new_size++;    
	}
    }
    assert(new_size <= alpha);
    pp_all->size = new_size;
    
    return;
}


void matingSelection()
/* Fills mating pool 'pp_sel' */
{
    int i, j;

    for (i = 0; i < mu; i++)
    {
	int winner = irand(pp_all->size);
	
	for (j = 1; j < tournament; j++)
	{
	    int opponent = irand(pp_all->size);
	    if (pp_all->ind_array[opponent]->fitness
		< pp_all->ind_array[winner]->fitness || winner == opponent)
	    {
		winner = opponent;
	    }
	}  
	pp_sel->ind_array[i] = pp_all->ind_array[winner];
    }
    pp_sel->size = mu;
}


void select_initial()
/* Performs initial selection. */
{
    selection();
}


void select_normal()
/* Performs normal selection.*/
{
    selection();
}


int dominates(ind *p_ind_a, ind *p_ind_b)
/* Determines if one individual dominates another.
   Minimizing fitness values. */
{
    int i;
    int a_is_worse = 0;
    int equal = 1;
    
     for (i = 0; i < dim && !a_is_worse; i++)
     {
	 a_is_worse = p_ind_a->f[i] > p_ind_b->f[i];
          equal = (p_ind_a->f[i] == p_ind_b->f[i]) && equal;
     }
     
     return (!equal && !a_is_worse);
}


double calcHypervolumeIndicator(ind *p_ind_a, ind *p_ind_b, int d)
/* calculates the hypervolume of that portion of the objective space that
   is dominated by individual a but not by individual b */
{
    double a, b, r, max;
    double volume = 0;

    r = rho * (bounds[d - 1].max - bounds[d - 1].min);
    max = bounds[d - 1].min + r;	
    
    assert(p_ind_a != NULL);
    a = p_ind_a->f[d - 1];
    if (p_ind_b == NULL)
	b = max;
    else
	b = p_ind_b->f[d - 1];
    
    assert(d > 0);
    if (d == 1)
    {
	if (a < b)
	    volume = (b - a) / r;
	else
	    volume = 0;
    }
    else
    {
	if (a < b)
	{
	    volume = calcHypervolumeIndicator(p_ind_a, NULL, d - 1) *
		(b - a) / r;
	    volume += calcHypervolumeIndicator(p_ind_a, p_ind_b, d - 1) *
		(max - b) / r;
	}
	else
	{
	    volume = calcHypervolumeIndicator(p_ind_a, p_ind_b, d - 1) *
		(max - b) / r;
	}
    }
    
    return (volume);
}


double calcAddEpsIndicator(ind *p_ind_a, ind *p_ind_b)
/* calculates the maximum epsilon value by which individual a must be
   decreased in all objectives such that individual b is weakly dominated */
{
    int i;
    double r;
    double eps = 0;

    r = bounds[0].max - bounds[0].min;
    eps = (p_ind_a->f[0] - bounds[0].min) / r -
	(p_ind_b->f[0] - bounds[0].min) / r;
    for (i = 1; i < dim; i++)
    {
	double temp_eps;

	r = bounds[i].max - bounds[i].min;
	temp_eps = (p_ind_a->f[i] - bounds[i].min) / r -
	    (p_ind_b->f[i] - bounds[i].min) / r;
	if (temp_eps > eps)
	    eps = temp_eps;
    }

    return (eps);
}


double calcIndicatorValue(ind *p_ind_a, ind *p_ind_b)
{
    double indicatorValue;
    
    if (indicator == 0)
	indicatorValue = calcAddEpsIndicator(p_ind_a, p_ind_b);
    else
    {
	if (dominates(p_ind_a, p_ind_b))
	    indicatorValue = -calcHypervolumeIndicator(p_ind_a, p_ind_b, dim);
	else
	    indicatorValue = calcHypervolumeIndicator(p_ind_b, p_ind_a, dim);
    }
    
    return (indicatorValue);
}


int irand(int range)
/* Generate a random integer. */
{
    int j;
    j=(int) ((double)range * (double) rand() / (RAND_MAX+1.0));
    return (j);
}


/*--------------------| data exchange functions |------------------------*/

int read_ini()
{
    int i;
    pp_new = create_pop(alpha, dim);
    
    for (i = 0; i < alpha; i++)
	pp_new->ind_array[i] = create_ind(dim);
    pp_new->size = alpha;
    
    return (read_pop(inifile, pp_new, alpha, dim));                    
}


int read_var()
{
    int i;
    pp_new = create_pop(lambda, dim);
    
    for (i = 0; i < lambda; i++)
	pp_new->ind_array[i] = create_ind(dim);
    
    pp_new->size = lambda;
    return (read_pop(varfile, pp_new, lambda, dim));
}


void write_sel()
{
    write_pop(selfile, pp_sel, mu);
}


void write_arc()
{
     write_pop(arcfile, pp_all, pp_all->size);
}


int check_sel()
{
     return (check_file(selfile));
}


int check_arc()
{
     return (check_file(arcfile));
}
