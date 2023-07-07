/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Internal functions, used in variator.c
  
  C file.
  
  file: variator_internal.c

  author: Fabian Landis, flandis@ee.ethz.ch
  
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
#include "variator_internal.h"

/* this is needed for the wait function */
#ifdef PISA_UNIX
#include <unistd.h>
#endif

#ifdef PISA_WIN
#include <windows.h>
#endif

/*--------------------| global variable definitions |-------------------*/

/* declared in variator_internal.h used in other files as well */

char cfg_file[FILE_NAME_LENGTH_INTERNAL];  
/* 'cfg' file (common parameters) */

char ini_file[FILE_NAME_LENGTH_INTERNAL];  
/* 'ini' file (initial population) */

char sel_file[FILE_NAME_LENGTH_INTERNAL]; 
/* 'sel' file (parents) */

char arc_file[FILE_NAME_LENGTH_INTERNAL]; 
/* 'arc' file (archive) */

char var_file[FILE_NAME_LENGTH_INTERNAL]; 
/* 'var' file (offspring) */

char sta_file[FILE_NAME_LENGTH_INTERNAL]; 
/* 'sta' file (current state) */

population global_population; /* pool of all existing individuals */


/*-------------------------| functions for handling state file |--------*/


/* Write the state flag */
int write_state(int state)
{
     FILE *fp;

     assert(0 <= state <= 11);
     
     fp = fopen(sta_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", state);
     fclose(fp);
     return(0);
}


/* Read state flag */
int read_state()
{
     int result;
     int state = -1;
     FILE *fp;
     fp = fopen(sta_file, "r");
     if (fp != NULL)
     {
          result = fscanf(fp, "%d", &state);
          fclose(fp);
          if (result == 1) /* exactly one element read */
          {
               if (state < 0 || state > 11)
                    log_to_file(log_file, __FILE__, __LINE__,
                                "invalid state read");   
          }
     }
     
     return (state);
}


int state_error(int error, int linenumber)
/* Outputs an error message and calls exit. */
{
     char error_message[50];
     printf("error in state %d \n", error);
     sprintf(error_message, "error in state %d \n", error);
     log_to_file(log_file, NULL, linenumber, error_message);
     exit(EXIT_FAILURE);
}


int wait(double sec)
/* Makes the calling process sleep for 'sec' seconds.
   
  pre: 'sec' >= 0.01
  post: Calling process is sleeping for 'sec' * 1e6 microseconds.
        The requested time is rounded up to the next integer multiple
        of the resolution the system can deliver.

  CAUTION: sleep and usleep() are not standard C, use Sleep(milliseconds)
           in <windows.h> for Windows version.
*/
{
#ifdef PISA_UNIX
     unsigned int int_sec;
     unsigned int usec;

     assert(sec > 0);
     
     int_sec = (unsigned int) floor(sec);
     usec = (unsigned int) floor((sec - floor(sec)) * 1e6);
     /* split it up, usleep can fail if argument greater than 1e6 */

     
     /* two asserts to make sure your file server doesn't break down */
     assert(!((int_sec == 0) && (usec == 0))); /* should never be 0 */
     assert((int_sec * 1e6) + usec >= 10000);  /* you might change this one
                                                  if you know what you are
                                                  doing */
    
     sleep(int_sec);
     usleep(usec);
#endif

#ifdef PISA_WIN
     unsigned int msec;
     assert(sec > 0);
     msec = (unsigned int) floor(sec * 1e3);
     assert(msec >= 10); /* making sure we are really sleeping for some time*/
     Sleep(msec);
#endif

     return (0);
}

/*-------------------------| stack functions  |-------------------------*/

int free_stack(stack *st)
{
     while(pop(st) != -1)
     {
          /* don't do anything, pop() does it already. */
     }
     return (0);
}

int push(stack *st,int id)
{
     stack_node *new_el = (stack_node *) malloc(sizeof(stack_node));
     if (new_el == NULL)
          return(1);
     new_el->next = st->top;
     new_el->identity = id;
     st->top = new_el;
     st->size++;
     return(0);
}

int pop(stack *st)
{
     int identity;
     stack_node *next;
     if(st->size == 0)
          return(-1);
     identity = st->top->identity;
     next = st->top->next;
     free(st->top);
     st->top = next;
     st->size--;
     return(identity);
}

/*-------------------| global population functions |---------------------*/

int remove_individual(int identity)
/* Removes the individual with ID 'identity' from the global population. */
{
     individual *temp;
     int result;
     if((identity > global_population.last_identity) || (identity < 0))
          return (1);
     temp = get_individual(identity);
     if(temp == NULL)
          return (1);

     global_population.individual_array[identity] = NULL;

     if(identity == global_population.last_identity)
     {
          global_population.last_identity--;
     }
     else
     {
          result = push(&global_population.free_ids_stack, identity);
          if (result == 1)
          {
               log_to_file(log_file, __FILE__, __LINE__, 
                           "Pushing a free identity to stack failed.");
               return (1);
          }
     }

     global_population.size--;

     free_individual(temp);
     
     return (0);
}


int clean_population()
/* Frees memory for all individuals in population and for the global
   population itself. */
{
     int current_id;

     if (NULL != global_population.individual_array)
     {
        current_id = get_first();
        while(current_id != -1)
        {
           remove_individual(current_id);
           current_id = get_next(current_id);
        }
        
        free_stack(&global_population.free_ids_stack);
        free(global_population.individual_array);

        global_population.individual_array = NULL;
        global_population.size = 0;
        global_population.last_identity = -1;
        
     }
     
        
     return (0);
}


/*-------------------------| other functions |-------------------------*/

int read_common_parameters()
/* Reads global parameters from 'cfg' file. */
{
     FILE *fp;
     
     int result;
     char str[CFG_ENTRY_LENGTH_INTERNAL];

     /* reading cfg file with common configurations for both parts */
     fp = fopen(cfg_file, "r");
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

     result = fscanf(fp, "%s", str);
     assert(strcmp(str, "dim") == 0);
     result = fscanf(fp, "%d", &dimension);
     assert(result != EOF); /* no EOF, dim correctly read */
     assert(dimension > 0);
     
     fclose(fp);     
     return (0);
}




int cmp_int(const void *p_i1, const void *p_i2)
/* Compares the two integers '*p_i1' and '*p_i2'.
   Returns 0 if *p_i1 == *p_i2.
   Returns 1 if *p_i1 > *p_i2.
   Returns -1 if *p_i1 < *p_i2. */
{
     int i1 = *((int *)p_i1);
     int i2 = *((int *)p_i2);
     
     if(i1 == i2)
          return (0);

     if(i1 > i2)
          return (1);
     else
          return (-1);
}


int check_ini()
/* Returns 0 if 'var_file' contains only '0'and returns 1 otherwise. */
{
     int control_element = 1;

     FILE *fp;

     fp = fopen(ini_file, "r");
     assert(fp != NULL);
     fscanf(fp, "%d", &control_element);
     fclose(fp);
     
     if (0 == control_element)
          return (0); /* file is ready for writing */
     else
          return (1); /* file is not ready for writing */
}

int check_var()
/* Returns 0 if 'var_file' contains only '0'and returns 1 otherwise. */
{
     int control_element = 1;

     FILE *fp;

     fp = fopen(var_file, "r");
     assert(fp != NULL);
     fscanf(fp, "%d", &control_element);
     fclose(fp);
     
     if (0 == control_element)
          return (0); /* file is ready for writing */
     else
          return (1); /* file is not ready for writing */
}
