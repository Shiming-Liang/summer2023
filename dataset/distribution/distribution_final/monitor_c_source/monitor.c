/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Pisa basic functions for use in monitor_user.c
  
  C file.
  
  file: monitor.c
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch

  last change: $date$
  
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "monitor.h"
#include "monitor_user.h"
#include "monitor_internal.h"


/*--------------------| global variable definitions |-------------------*/

int alpha; /* number of individuals in initial population */
int mu; /* number of individuals selected as parents */
int lambda; /* number of offspring individuals */
int dimension; /* number of objectives */
double poll; /* polling interval in seconds */

char filenamebase_variator[FILE_NAME_LENGTH_INTERNAL];
char filenamebase_selector[FILE_NAME_LENGTH_INTERNAL];
char parnamebase_selector[FILE_NAME_LENGTH_INTERNAL];
char parnamebase_variator[FILE_NAME_LENGTH_INTERNAL];
char filenamebase_monitor[FILE_NAME_LENGTH_INTERNAL];
char parnamebase_monitor[FILE_NAME_LENGTH_INTERNAL];
char cfg_file_variator[FILE_NAME_LENGTH_INTERNAL];  
char ini_file_variator[FILE_NAME_LENGTH_INTERNAL];  
char sel_file_variator[FILE_NAME_LENGTH_INTERNAL]; 
char arc_file_variator[FILE_NAME_LENGTH_INTERNAL]; 
char var_file_variator[FILE_NAME_LENGTH_INTERNAL]; 
char sta_file_variator[FILE_NAME_LENGTH_INTERNAL]; 
char cfg_file_selector[FILE_NAME_LENGTH_INTERNAL];  
char ini_file_selector[FILE_NAME_LENGTH_INTERNAL];  
char sel_file_selector[FILE_NAME_LENGTH_INTERNAL]; 
char arc_file_selector[FILE_NAME_LENGTH_INTERNAL]; 
char var_file_selector[FILE_NAME_LENGTH_INTERNAL]; 
char sta_file_selector[FILE_NAME_LENGTH_INTERNAL]; 

int currentRun;
int currentGeneration;

/*-------------------------| main() |-----------------------------------*/

int main(int argc, char *argv[])
{
     int current_state = 0;
     int monitor_state = 1; /* polling variator (0) or selector (1) */
     int variator_terminated = 0;
     int selector_terminated = 0;

     if (argc == 8)
     {
          sscanf(argv[1], "%s", parnamebase_variator);
          sscanf(argv[2], "%s", filenamebase_variator);
		  sscanf(argv[3], "%s", parnamebase_selector);
          sscanf(argv[4], "%s", filenamebase_selector);
		  sscanf(argv[5], "%s", parnamebase_monitor);
          sscanf(argv[6], "%s", filenamebase_monitor);
          sscanf(argv[7], "%lf", &poll);
          assert(poll >= 0);
     }
     else
     {
          printf("Monitor - wrong number of arguments:\n");
		  printf("monitor varPar varBase selPar selBase monPar monBase poll\n");
          return (1);
     }

     /* generate file names based on 'filenamebase'*/
     sprintf(var_file_variator, "%svar", filenamebase_variator);
     sprintf(sel_file_variator, "%ssel", filenamebase_variator);
     sprintf(cfg_file_variator, "%scfg", filenamebase_variator);
     sprintf(ini_file_variator, "%sini", filenamebase_variator);
     sprintf(arc_file_variator, "%sarc", filenamebase_variator);
     sprintf(sta_file_variator, "%ssta", filenamebase_variator);
     sprintf(var_file_selector, "%svar", filenamebase_selector);
     sprintf(sel_file_selector, "%ssel", filenamebase_selector);
     sprintf(cfg_file_selector, "%scfg", filenamebase_selector);
     sprintf(ini_file_selector, "%sini", filenamebase_selector);
     sprintf(arc_file_selector, "%sarc", filenamebase_selector);
     sprintf(sta_file_selector, "%ssta", filenamebase_selector);

     /* initialize common parameters */
     alpha = 0;
     mu = 0;
     lambda = 0;
     dimension = 0;

	 /* read and check common parameters (they should be equal)*/
     read_common_parameters(cfg_file_variator);
     read_common_parameters(cfg_file_selector);

	 /* read monitor parameters */
     read_local_parameters();

	 /* seed random generator */
	 srand(seed);
	 
	 /* print information file of monitor */
	 printInformation();

	 for (currentRun = 0 ; currentRun < numberOfRuns; currentRun++ ) {
		if (LOG) printf("currentRun = %d \n",currentRun);
		resetAll();
		for (currentGeneration = 1 ; currentGeneration <= numberOfGenerations; currentGeneration++ ) {
			if (LOG) printf("  currentGeneration = %d \n",currentGeneration);
			while(1) {
				if (read_state(sta_file_variator) == 3) {
					copyVariate();
					write_state(sta_file_selector,3);
					if (LOG) printf("    variation done; selector state 3\n");
					break;
				}
				wait(poll);
			}
			while(1) {
				if (read_state(sta_file_selector) == 2) {
					copyArchiveSelected();
					appendOutput();
					write_state(sta_file_variator, 2);
					if (LOG) printf("    selection done; variator state 2\n");
					break;
				};
				wait(poll);
			}
		}
	 }

	 if (LOG) printf("selector state 6 (kill)\n");
	 write_state(sta_file_selector, 6);
	 while(1) {
		 if (read_state(sta_file_selector) == 7) {
			 if (LOG) printf("selector killed\n");
			 break;
		 }
		 wait(poll);
	 }

	 while(1) {
		 if (read_state(sta_file_variator) == 3) {
			 if (LOG) printf("variator state 4 (kill)\n");
			 write_state(sta_file_variator, 4);
			 break;
		 }
		 wait(poll);
	 }
	 while(1) {
		 if (read_state(sta_file_variator) == 5) {
			 if (LOG) printf("variator killed\n");
			 break;
		 }
		 wait(poll);
	 }

	 if (LOG) printf("kill myself\n");
     return (0);
}

/*-------------------------| reset variator and selector |--------------*/

void resetAll() {
	
	if (LOG) printf("  starting reset\n");
	currentGeneration = 0;
    write_state(sta_file_selector, 10);
	write_state(sta_file_variator, 8);
	if (LOG) printf("    variator, selector states 8, 10\n");

	while(1){
		if (read_state(sta_file_variator) != 9) {
			write_state(sta_file_variator, 8);
		};
		if (read_state(sta_file_selector) != 11) {
			write_state(sta_file_selector, 10);
		};
		if (read_state(sta_file_variator) == 9 && read_state(sta_file_selector) == 11) {
			updateVariatorSeed();
			write_state(ini_file_variator,0); // make ini file ready for writing
			write_state(sta_file_variator, 0);
			if (LOG) printf("    variator state 0\n");
			if (LOG) printf("  currentGeneration = %d \n",currentGeneration);
			break;
		};
		wait(poll);
	}

	while(1) {
		if (read_state(sta_file_variator) == 1) {
			copyInitialPop();
			write_state(arc_file_selector,0); // make arc file ready for write
			write_state(sel_file_selector,0); // make sel file ready for write
			write_state(sta_file_selector, 1);
			if (LOG) printf("    ini_pop done; selector state 1\n");
			break;
		};
		wait(poll);
	}

	while(1) {
		if (read_state(sta_file_selector) == 2) {
			copyArchiveSelected();
			appendOutput();
			write_state(var_file_variator,0); // make var file ready for write
			write_state(sta_file_variator, 2);
			if (LOG) printf("    selection done; variator state 2\n");
			break;
		};
		wait(poll);
	}

	return;
}


/*-------------------------| copy files |-----------------------------------*/

void copyInitialPop()
{
    int i;
    double **f = NULL;

    f = move_ini();
    state1_user(f);

    for (i = 0; i < alpha; i++)
    {
	free(f[i]);
    }

    free(f);
}

void copyArchiveSelected()
{
    int *id = NULL;

    move_sel();
    id = move_arc();    
    state2_user(id);

    free(id);
}

void copyVariate()
{
    int i;
    double **f = NULL;

    f = move_var();
    state3_user(f);

    for (i = 0; i < lambda; i++)
    {
	free(f[i]);
    }

    free(f);
}

/*-------------------------| io |---------------------------------------*/

int* move_arc()
/* Moves arc file from selector to variator side.
   Returns dynamic array with ids. Size is stored in the first entry. */
{
     int size, result;
     FILE *fp, *fp2; 
     char tag[4];
     int i;
     int *keep = NULL;
     /* keep[0] = size of archive
	    keep[i] = ids of archive elements (1 <= i <= size) */


     fp = fopen(arc_file_selector, "r");
     fp2 = fopen(arc_file_variator, "w");
     assert(fp != NULL);
     assert(fp2 != NULL);
     
     /* read arc file and store indexes in keep array */
     fscanf(fp, "%d", &size);
     assert(size > 0); /* we need to keep at least one individual */
     fprintf(fp2, "%d\n", size);

     result = 0;
     keep = (int *) malloc(sizeof(int)*(size + 1));
     assert(keep != NULL);
     keep[0] = size; /* Store size of array in first entry */

     for (i = 1; i <= size; i++)
     {
          result = fscanf(fp, "%d", &keep[i]);
	  assert(result != EOF); /* fscanf() returns EOF if reading failed */
	  fprintf(fp2, "%d\n", keep[i]);
     }

     fscanf(fp, "%s", tag);
     assert(strcmp(tag, "END") == 0); /* "END" here ? */
     fprintf(fp2, "END");
     fclose(fp);
     fclose(fp2);
     
     /* deleting content */
     fp = fopen(arc_file_selector, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     return (keep);
}


void move_sel()
/* Moves sel file from selector to variator side.*/
{
     int size, result; 
     FILE *fp, *fp2; 
     char tag[4];
     int i;
     int id;
     
     fp = fopen(sel_file_selector, "r");
     fp2 = fopen(sel_file_variator, "w");
     assert(fp != NULL);
     assert(fp2 != NULL);
     
     /* read sel file and store indexes in 'id_array' */
     fscanf(fp, "%d", &size);
     fprintf(fp2, "%d\n", size);

     for(i = 0; i < size; i++)
     {
          result = fscanf(fp, "%d", &id);
          assert(result != EOF); /* fscanf() returns EOF if reading failed */
	  fprintf(fp2, "%d\n", id);
     }
     
     fscanf(fp, "%s", tag);
     assert(strcmp(tag, "END") == 0); /* "END" not found */
     fprintf(fp2, "END");
     fclose(fp);
     fclose(fp2);
     
     /* deleting content */
     fp = fopen(sel_file_selector, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);     
}


double** move_ini()
/*  */
{
     int i, j, size;
     char tag[4];
     FILE *fp, *fp2;
     int result; /* stores return value of called functions */
     int identity;
     double **f = NULL;
	 /* f[i][0] = Index of element (0 <= i < alpha)
	    f[i][j] = jth objective of element (0 <= i < alpha) */

     f = (double **) malloc(alpha * sizeof(double*));
     assert (f != NULL);
     
     fp = fopen(ini_file_variator, "r");
     fp2 = fopen(ini_file_selector, "w");
     assert(fp != NULL);
     assert(fp2 != NULL);

     fscanf(fp, "%d", &size);
     /* test if size has a valid value */
     assert (size == ((dimension + 1) * alpha));
     fprintf(fp2, "%d\n", (alpha * (dimension + 1)));
     
     for(i = 0; i < alpha; i++)
     {
	 f[i] = (double *) malloc((dimension + 1 ) * sizeof(double));

	 result = fscanf(fp, "%d", &identity); /* fscanf() returns EOF
                                                   if reading fails.*/
	 assert(result != EOF); /* file not completely written */
	 f[i][0] = identity;
	 fprintf(fp2, "%d ", identity); /* prints also a space */

	 for (j = 1; j < dimension + 1; j++)
	 {
	     result = fscanf(fp, "%le", &f[i][j]);
	     assert(result != EOF); /* file not completely written */
	     fprintf(fp2, "%E ", f[i][j]); /* prints also a space */
	 }
	 
	 fprintf(fp2, "\n");
     }
     
     fscanf(fp, "%s", tag);
     assert(strcmp(tag, "END") == 0);
     fprintf(fp2, "END");
     
     fclose(fp);
     fclose(fp2);
     
     /* deleting content */
     fp = fopen(ini_file_variator, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     return (f);  
}

double** move_var()
/*  */
{
     int i, j, size;
     char tag[4];
     FILE *fp, *fp2;
     int result; /* stores return value of called functions */
     int identity;
     double **f = NULL;
	 /* f[i][0] = Index of element (0 <= i < alpha)
	    f[i][j] = jth objective of element (0 <= i < alpha) */

     f = (double **) malloc(lambda * sizeof(double*));
     assert (f != NULL);
     
     fp = fopen(var_file_variator, "r");
     fp2 = fopen(var_file_selector, "w");
     assert(fp != NULL);
     assert(fp2 != NULL);

     fscanf(fp, "%d", &size);
     /* test if size has a valid value */
     assert (size == ((dimension + 1) * lambda));
     fprintf(fp2, "%d\n", (lambda * (dimension + 1)));
     
     for(i = 0; i < lambda; i++)
     {
	 f[i] = (double *) malloc((dimension + 1 ) * sizeof(double));

	 result = fscanf(fp, "%d", &identity); /* fscanf() returns EOF
                                                   if reading fails.*/
	 assert(result != EOF); /* file not completely written */
	 f[i][0] = identity;
	 fprintf(fp2, "%d ", identity); /* prints also a space */
	 
	 for (j = 1; j < dimension + 1; j++)
	 {
	     result = fscanf(fp, "%le", &f[i][j]);
	     assert(result != EOF); /* file not completely written */
	     fprintf(fp2, "%E ", f[i][j]); /* prints also a space */
	 }
	 
	 fprintf(fp2, "\n");
     }
     
     fscanf(fp, "%s", tag);
     assert(strcmp(tag, "END") == 0);
     fprintf(fp2, "END");
     
     fclose(fp);
     fclose(fp2);
     
     /* deleting content */
     fp = fopen(var_file_variator, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     return (f);  
}
