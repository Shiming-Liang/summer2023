/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Functions the user must implement
  
  C file.
  
  file: monitor_user.h
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch

  last change: $date$
   
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "monitor.h"
#include "monitor_user.h"

/*--------------------| global variable definitions |-------------------*/
/* from monitor parameter file */
int seed;
int numberOfRuns;
int numberOfGenerations;
int outputType;
int outputSet;
int LOG;

int population_size;
double **population;
int paretoset_space;
int paretoset_size;
double **paretoset;

char front_offline[FILE_NAME_LENGTH];
char front_online[FILE_NAME_LENGTH];

void read_local_parameters()
{
     FILE *fp;
     int result;
	 char tmp[16];
     char str[CFG_NAME_LENGTH];

     /* reading parameter file with parameters for selection */
     fp = fopen(parnamebase_monitor, "r"); 
     assert(fp != NULL);

	 fscanf(fp, "%s", str);
     assert(strcmp(str, "seed") == 0);
     fscanf(fp, "%d", &seed);

	 fscanf(fp, "%s", str);
     assert(strcmp(str, "numberOfRuns") == 0);
     fscanf(fp, "%d", &numberOfRuns);

	 fscanf(fp, "%s", str);
     assert(strcmp(str, "numberOfGenerations") == 0);
     fscanf(fp, "%d", &numberOfGenerations);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "outputType") == 0);
     fscanf(fp, "%s", tmp);
	 if (strcmp(tmp, "all") == 0) {
		outputType = ALL;
	 } else if (strcmp(tmp, "online") == 0) {
		outputType = ONLINE;
	 } else if (strcmp(tmp, "offline") == 0) {
		outputType = OFFLINE;
	 } else {
		assert(0);
	 }

	 fscanf(fp, "%s", str);
     assert(strcmp(str, "outputSet") == 0);
     fscanf(fp, "%d", &outputSet);
     
	 result = fscanf(fp, "%s", str);
     assert(strcmp(str, "debug") == 0);
     fscanf(fp, "%d", &LOG);

     assert(result != EOF); /* no EOF, outfile correctly read */
     
     fclose(fp);
}

/*-------------------------| statemachine functions |-------------------*/

void state1_user(double **f)
{
    int i, j;
    
    population_size = alpha;
    population = (double**) malloc(population_size * sizeof(double*));
    /*  population[i][0] = Index of element (0 <= i < alpha)
	    population[i][j] = jth objective of element (0 <= i < alpha) 
	    This is the current population, i.e. a replicate of the 
	    selectors archive. */


    for (i = 0; i < alpha; i++)
    {
	population[i] = (double *) malloc((dimension + 1 ) * sizeof(double));
	for (j = 0; j < dimension + 1; j++)
	{
	    population[i][j] = f[i][j];
	}
    }

    paretoset_size = 0;
    paretoset_space = alpha;
    paretoset = (double**) malloc(paretoset_space * sizeof(double*));
    /*  paretoset[i][0] = Index of element (0 <= i < paretoset_size)
	    paretoset[i][j] = jth objective of element (0 <= i < paretoset_size) 
	    This is the accumulated pareto set. */

    for (i = 0; i < alpha; i++)
    {
	paretoset[i] = (double *) malloc((dimension + 1 ) * sizeof(double));
    }
    
    for (i = 0; i < alpha; i++)
    {
	paretoset_join(f[i]);
    }
}

void state2_user(int *id)
{
    int i, j, k;
    int old_population_size = population_size;
    double **old_population = population;

    population_size = id[0];
    population = (double**) malloc(population_size * sizeof(double*));

    k = 0;
    for (i = 0; i < old_population_size; i++)
    {
	for(j = 1; j <= population_size; j++)
	{
	    if (((int)old_population[i][0]) == id[j])
	    {
		population[k] = old_population[i];
		k++;
		break;
	    }
	}
	if (j > population_size)
	{
	    free(old_population[i]);    
	}
    }
    
    assert(k == population_size);

    free(old_population);
    
/*
    if (iteration % output == 0)
    {
	print_front_offline();
	print_front_online();
    }
*/

}

void state3_user(double **f)
{
    int i, j;
    int old_population_size = population_size;
    double **old_population = population;
    
    population_size = old_population_size + lambda;
    population = (double**) malloc(population_size * sizeof(double*));

    for (i = 0; i < old_population_size; i++)
    {
	population[i] = old_population[i];
    }
    
    for (i = old_population_size; i < population_size; i++)
    {
	population[i] = (double*) malloc((dimension + 1 ) * sizeof(double));
	for (j = 0; j <  dimension + 1; j++)
	{
	    population[i][j] = f[i - old_population_size][j];
	}
    }

    free(old_population);
    
    if (lambda > paretoset_space - paretoset_size)
    {
	int old_paretoset_space = paretoset_space;
	double **old_paretoset = paretoset;
	
	paretoset_space = old_paretoset_space + lambda;
	paretoset = (double**) malloc(paretoset_space * sizeof(double*));

	for (i = 0; i < old_paretoset_space; i++)
	{
	    paretoset[i] = old_paretoset[i];
	}
	
	for (i = old_paretoset_space; i < paretoset_space; i++)
	{
	    paretoset[i] = (double*) malloc((dimension + 1 ) * sizeof(double));
	}

	free(old_paretoset);
    }
	
    for (i = 0; i < lambda; i++)
    {
	paretoset_join(f[i]);
    }
}

void paretoset_join(double* f)
{
    int i, j;

    for (i = paretoset_size - 1; i >= 0; i--)
    {
	if (dominates(paretoset[i], f) == 1)
	{
	    return;
	}
	else if (dominates(f, paretoset[i]) == 1)
	{
	    /* remove element i from paretoset */
	    paretoset_size--;
	    for (j = 0; j <= dimension; j++)
	    {
		paretoset[i][j] = paretoset[paretoset_size][j];
	    }
	}
    }
    /* new element f is not dominated => insert it, increment size */
    for (j = 0; j <= dimension; j++)
    {
	paretoset[paretoset_size][j] = f[j];
    }
    paretoset_size++;
}


int dominates(double *f, double *g)
{
    int k;
    int dominates = 0;
    
    for (k = 1; k <= dimension; k++)
    {
	if (f[k] > g[k])
	{
	    return (0);
	}
	
	else if (f[k] < g[k])
	{
	    dominates = 1;
	}
    }
    
    return (dominates);    
}

void appendOutput() {

	FILE *fp = NULL;
	char output_monitor_file[FILE_NAME_LENGTH];

	if ((outputSet == 0 && currentGeneration == numberOfGenerations) || (outputSet !=0 && (currentGeneration % outputSet) == 0)) {
		sprintf(output_monitor_file, "%s.%d", filenamebase_monitor, currentGeneration);
		if (currentRun == 0) {
			fp = fopen(output_monitor_file, "w");
		} else {
			fp = fopen(output_monitor_file, "a");
		}
		assert(fp != NULL);
		
		if (outputType == ALL) {
			if(LOG) printf("    append ALL generation %d\n", currentGeneration);
			outputAll(fp);
		} else if (outputType == ONLINE) {
			if(LOG) printf("    append ONLINE generation %d\n", currentGeneration);
			outputOnline(fp);
		} else if (outputType == OFFLINE) {
			if(LOG) printf("    append OFFLINE generation %d\n", currentGeneration);
			outputOffline(fp);
		}
		fclose(fp);
    }
}

void outputAll(FILE *fp) {
	int i, j;

	for (i = 0; i < population_size; i++) {
		for (j = 1; j <= dimension; j++) {
			fprintf(fp,"%.9e ", population[i][j]);
		}
	    fprintf(fp,"\n");
    }
	fprintf(fp,"\n");
}

void outputOnline(FILE *fp) {
	int i, j;
    int* pareto = (int *) malloc(population_size * sizeof(int));

    for (i = 0; i < population_size; i++) {
		pareto[i] = 1;
		for (j = 0; j < population_size; j++) {
			if (dominates(population[j],population[i]) == 1) {
				pareto[i] = 0;
			}
		}
    }

    for (i = 0; i < population_size; i++) {
		if (pareto[i] == 1) {
			for (j = 1; j <= dimension; j++) {
				fprintf(fp,"%.9e ", population[i][j]);
			}
	    fprintf(fp,"\n");
		}
    }
	fprintf(fp,"\n");
    
    free(pareto);
}

void outputOffline(FILE *fp) {
    int i, j;

	for (i = 0; i < paretoset_size; i++) {
		for (j = 1; j <= dimension; j++) {
			fprintf(fp,"%.9e ", paretoset[i][j]);
		}
		fprintf(fp,"\n");
    }
	fprintf(fp,"\n");
}

/* Generate a random integer. */
int irand(int range)
{
     int j;
     j=(int) ((double) range * (double) rand() / (RAND_MAX + 1.0));
     return (j);
}

void updateVariatorSeed(){

	FILE *fp = NULL;
	char *(contents[LINE_LENGTH]);
	int linectr = 0;
	int line;
	char str[CFG_NAME_LENGTH]; 
	int oldSeed, newSeed;

	if(LOG) printf("    update variator seed\n");

	// read from parameter file into array of strings contents

	fp = fopen(parnamebase_variator, "r");
	assert(fp != NULL);

	contents[0] = (char *) malloc(sizeof(char[LINE_LENGTH]));
	assert(contents[0] != NULL);

	while (fgets(contents[linectr], LINE_LENGTH, fp) != NULL) {
		linectr++;
		contents[linectr] = (char *) malloc(sizeof(char[LINE_LENGTH]));
		assert(contents[linectr] != NULL);
	}
	free(contents[linectr]);
	fclose(fp);

	// write to parameter file from array of strings contents
	// replace seed

	fp = fopen(parnamebase_variator, "w");
	assert(fp != NULL);

	line = 0;
	while (line < linectr) {
		sscanf(contents[line], "%s %d", str, &oldSeed);
		if (strcmp(str, "seed") == 0) {
			newSeed = irand(INT_MAX);
			fprintf(fp, "seed %d\n", newSeed);
			if(LOG) printf("      seed %d replace by %d\n", oldSeed, newSeed);
		} else {
			fputs(contents[line], fp);
		}
		free(contents[line]);
		line++;
	}
	fclose(fp);
}

void printInformation() {

	FILE *fp = NULL;
	FILE *fp2 = NULL;
	char str[LINE_LENGTH];
	char log_monitor_file[FILE_NAME_LENGTH];

	if(LOG) printf("write monitor information file\n");

	sprintf(log_monitor_file, "%s.txt", filenamebase_monitor);
	fp = fopen(log_monitor_file, "w");
	assert(fp != NULL);

	// write command line of monitor
	fprintf(fp, "start commandLine\n");
	fprintf(fp, "%s %s %s %s %s %s %g\n",parnamebase_variator,filenamebase_variator,parnamebase_selector,filenamebase_selector,parnamebase_monitor,filenamebase_monitor,poll);
	fprintf(fp, "end commandLine\n\n");

	// write monitor parameters
	fp2 = fopen(parnamebase_monitor, "r");
	assert(fp2 != NULL);
	fprintf(fp, "start monParameter\n");
	while (fgets(str, LINE_LENGTH, fp2) != NULL) fputs(str, fp);
	fprintf(fp, "\nend monParameter\n\n");
	fclose(fp2);

	// write variator common parameters
	fp2 = fopen(cfg_file_variator, "r");
	assert(fp2 != NULL);
	fprintf(fp, "start varCommonParameter\n");
	while (fgets(str, LINE_LENGTH, fp2) != NULL) fputs(str, fp);
	fprintf(fp, "\nend varCommonParameter\n\n");
	fclose(fp2);

	// write variator parameters
	fp2 = fopen(parnamebase_variator, "r");
	assert(fp2 != NULL);
	fprintf(fp, "start varParameter\n");
	while (fgets(str, LINE_LENGTH, fp2) != NULL) fputs(str, fp);
	fprintf(fp, "\nend varParameter\n\n");
	fclose(fp2);

	// write selector common parameters
	fp2 = fopen(cfg_file_selector, "r");
	assert(fp2 != NULL);
	fprintf(fp, "start selCommonParameter\n");
	while (fgets(str, LINE_LENGTH, fp2) != NULL) fputs(str, fp);
	fprintf(fp, "\nend selCommonParameter\n\n");
	fclose(fp2);

	// write selector parameters
	fp2 = fopen(parnamebase_selector, "r");
	assert(fp2 != NULL);
	fprintf(fp, "start selParameter\n");
	while (fgets(str, LINE_LENGTH, fp2) != NULL) fputs(str, fp);
	fprintf(fp, "\nend selParameter");
	fclose(fp2);

	fclose(fp);
}

