/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Pisa basic functions for use in variator_user.c
  
  C file.
  
  file: variator.c
  author: Fabian Landis, flandis@ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
  
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "variator.h"
#include "variator_user.h"
#include "variator_internal.h"


/*--------------------| global variable definitions |-------------------*/

/* declared in variator.h used in other files as well */

int alpha; /* number of individuals in initial population */

int mu; /* number of individuals selected as parents */

int lambda; /* number of offspring individuals */

int dimension; /* number of objectives */


/* only used in this file */

int current_max_size; 
/* starting array size of the individuals array in global_population,
   defined in variator_internal.c */


/*-------------------------| main() |-----------------------------------*/

int main(int argc, char *argv[])
{
     FILE *fp;
     int returncode; /* storing the values that the state functions return */
     int current_state = 0;

     char filenamebase[FILE_NAME_LENGTH_INTERNAL]; /* filename base,
                                                      e.g. "dir/test." */

     double poll;                  /* polling interval in seconds*/
 
     
     if (argc == 4)
     {
          sscanf(argv[1], "%s", paramfile);
          sscanf(argv[2], "%s", filenamebase);
          sscanf(argv[3], "%lf", &poll);
          assert(poll >= 0);
     }
     else
     {
          printf("Variator - wrong number of arguments\n");
          return (1);
     }

     /* generate file names based on 'filenamebase'*/
     sprintf(var_file, "%svar", filenamebase);
     sprintf(sel_file, "%ssel", filenamebase);
     sprintf(cfg_file, "%scfg", filenamebase);
     sprintf(ini_file, "%sini", filenamebase);
     sprintf(arc_file, "%sarc", filenamebase);
     sprintf(sta_file, "%ssta", filenamebase);


     /* initializing global_population */
     global_population.individual_array = NULL;
     global_population.size = 0;
     global_population.last_identity = -1;
     
     
     /* creating files and writing 0 in there */
     fp = fopen(var_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     fp = fopen(sel_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);

     fp = fopen(ini_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     fp = fopen(arc_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", 0);
     fclose(fp);
     
     /* state machine: uses the stateX() functions to do the steps required
        in state X. Those state functions have to be implemented by 
        the user in selector_user.c */
     
     write_state(current_state);
     
     while (current_state != 4) /* Caution: if reading of the sta_file
                                   fails (e.g. no permission) this is an
                                   infinite loop */
     {
          current_state = read_state(); /* state == -1 if reading
                                                   fails */
          if (current_state == 0)
          {
               read_common_parameters();

               returncode = state0();
               
               if (returncode == 0)
               {
                  current_state = 1;
                  write_state(current_state);     
               }
               
               else if (returncode != 2)  /* error other then
                                             file reading */
               {
                  state_error(0, __LINE__);
               } /* else do nothing and read state again */
          }
          
          else if (current_state == 2)
          { 
               if (is_finished()) /* checking termination criterion */
               {
                    current_state = 4; /* terminate */
                    write_state(current_state);
               }
               else
               {
                    if (check_var() == 0)
                    {
                         returncode = state2();
                         if (returncode == 0)/* if everything went ok set
                                                state 3 */
                         {
                              del_sel(); /* all ok => delete content of */
                              del_arc(); /* files. */
                              current_state = 3; 
                              write_state(current_state);
                         }
                         else if (returncode != 2)  /* error other then
                                                       file reading */
                         {
                              state_error(2, __LINE__);
                         } /* else do nothing and read state again */
                    }
               }
          }
          
      
          else if (current_state == 7) /* selector has just terminated;
                                          do what you like ... */
          { /* ... e.g. go to state 4, terminate as well */
               returncode = state7();
               if (returncode == 0)
               {
                    current_state = 4;
                    write_state(current_state);
               }
               else if (returncode != 2)  /* error other then
                                             file reading */
               {
                    state_error(7, __LINE__);
               } /* else do nothing and read state again */
          }
      
          else if (current_state == 8)
          {
               returncode = state8();
               if (returncode == 0)
               {
                    current_state = 9;
                    clean_population();
                    write_state(current_state);
               }
               else if (returncode != 2)  /* error other then
                                             file reading */
               {
                    state_error(8, __LINE__);
               } /* else do nothing and read state again */
               
          }
          
          else if (current_state == 11) /* selector has resetted and is ready
                                           to start again in state 0;
                                           do what you like ... */
          {
               returncode = state11();
               if (returncode == 0)
               {
                    current_state = 0;
                    write_state(current_state);
               }
               else if (returncode != 2)  /* error other then
                                             file reading */
               {
                    state_error(11, __LINE__);
               } /* else do nothing and read state again */
          }
      
          else /* no state which concerns variation */
          {
               wait(poll);
          }
      
     } /* state == 4 (stop) */

     returncode = state4();
     if (returncode == 0)
     {
          clean_population();
          current_state = 5;
          write_state(current_state);
     }
     else
     {
          state_error(4, __LINE__);
     }
     return (0);
}


/*-------------------------| populations functions |--------------------*/

int add_individual(individual *ind)
/* Takes a pointer to an individual and adds the individual to the
   global population.
   Returns the identity assigned to the individual.
   Returns -1 if adding failed.*/
{
     int i;
     int identity = -1;
     individual **tmp; /* in case we need to double array size */

     if (ind == NULL) /* there is no individual to add */
          return (-1);
     
     /* if size == 0 we need to allocate memory for our population */
     if(global_population.size == 0)
     {
          current_max_size = STANDARD_SIZE;
          global_population.individual_array =
               (individual **) malloc(current_max_size * sizeof(int));
          if (global_population.individual_array == NULL)
          {
                log_to_file(log_file, __FILE__, __LINE__,
                            "variator out of memory");
                return (-1);
          }
          global_population.last_identity = -1;
     }
  
     /* search for free id */ 
     identity = pop(&global_population.free_ids_stack);
     if (identity == -1)
     {
          identity = global_population.last_identity + 1;
          global_population.last_identity++;
     } 

     global_population.size++;

     if (global_population.last_identity < current_max_size)
     {
          global_population.individual_array[identity] = ind;    
     }
     else /* enlargement of individual array (size doubling) */
     { 
          tmp = (individual **) malloc(sizeof(int) * current_max_size * 2);
          if (tmp == NULL)
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "variator out of memory");
               return (-1);
          }
          /* copy old array */
          for (i = 0; i < current_max_size; i++)
               tmp[i] = global_population.individual_array[i];
          current_max_size = current_max_size * 2;
          /* free memory */
          free(global_population.individual_array);
          global_population.individual_array = tmp;
     }

     return (identity);
}


individual *get_individual(int identity) 
/* Returns a pointer to the individual corresponding to 'identity'. */
{
     if((identity > global_population.last_identity) || (identity < 0))
          return (NULL);
     return (global_population.individual_array[identity]);
}


int get_first()
/* Returns the identity of first individual in the global population. */
{
     return (get_next(-1));
}


int get_next(int identity) 
/* Takes an identity an returns the identity of the next following
   individual in the global population. */
{ 
     int next_id;

     if((identity < -1) || (identity > global_population.last_identity))
          return (-1);
  
     next_id = identity + 1;
     while(next_id <= global_population.last_identity) {
          if(global_population.individual_array[next_id] != NULL)
               return (next_id);
          next_id++;
     }

     return(-1);
}


int get_size()
/* Returns the size of the global population, i.e. the current number
   of individuals in the population. */
{
     return (global_population.size);
}

/*-------------------------| io |---------------------------------------*/


int read_arc()
/* Reads 'arc' file, and automatically removes all individuals from
   the global population which are not in the arc file. */
{
     int size, result; 
     int *keep;
     FILE *fp; 
     char tag[4];
     int i, current;

     fp = fopen(arc_file, "r");
     assert(fp != NULL);
     
     /* read arc file and store indexes in keep array */
     fscanf(fp, "%d", &size);
     if(size <= 0) /* we need to keep at least one individual */
     {
          log_to_file(log_file, __FILE__, __LINE__, "size<=0 in arc file!");
          fclose(fp);
          return (1);
     }

     result = 0;
     keep = (int *) malloc(sizeof(int)*size);

     if (keep == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          fclose(fp);
          return (1);
     }  

     for(i=0; i < size; i++)
     {
          result = fscanf(fp, "%d", &keep[i]);
          if (result == EOF) /* fscanf() returns EOF if reading failed */
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "EOF before END tag");
               fclose(fp);
               return (1);
          } 
     }

     fscanf(fp, "%s", tag);

     if (strcmp(tag, "END") != 0) /* no "END" here */
     {
          log_to_file(log_file, __FILE__, __LINE__, "no END tag");
          fclose(fp);
          return (1);
     }
     else /* "END" ok - all reading done*/
     {
          fclose(fp);
          /* deleting content must be done after arc and sel file are read */
     }

     /* sort the array of indexes to keep,
        so we can go through the array and delete all indexes,
        that are in between */
     qsort(keep, (size_t) size, sizeof(int), cmp_int);      
    
     /* delete all indexes in global_population not found in keep array */
     current = get_first();
     for(i = 0; i < size; i++)
     {
          while(current < keep[i])
          {
               result = remove_individual(current);
               if (result != 0)
                    return(1);
               current = get_next(current);
          }
          if (current == keep[i])
          {
               current = get_next(current);
          } /* this one we keep */
          else  /* current must be bigger than keep[i],
                   something went wrong... */
          { 
              
               log_to_file(log_file, __FILE__, __LINE__, 
                           "identity in arc_file not in global population!");
               return (1);
          }
     }

     /* delete the last individuals at end of list */
     while(current != -1)
     {
          result = remove_individual(current);
          if (result != 0) 
               return (1);
          current = get_next(current);
     } 
  
     free(keep);
     return (0);
}


void del_arc()
/* Deletes the content of the arc file. */
{
   FILE *fp;
   fp = fopen(arc_file, "w");
   assert(fp != NULL);
   fprintf(fp, "%d", 0);
   fclose(fp);
}


int read_sel(int *id_array)
/* Reads a list of IDs from the sel_file. These IDs denote the
   individuals chosen as parents. The IDs of the individuals are
   stored in 'id_array'.  The number of IDs in this array is
   'mu'. 'id_array' must be big enough to store 'mu' 'int'
   variables.

   If reading is successful function returns 0, otherwise it returns
   1. */
{
     int size, result; 
     FILE *fp; 
     char tag[4];
     int i;

     assert(id_array != NULL);
     
     fp = fopen(sel_file, "r");
     assert(fp != NULL);
     /* read sel file and store indexes in 'id_array' */
     fscanf(fp, "%d", &size);
     result = 0;

     for(i = 0; i < size; i++)
     {
          result = fscanf(fp, "%d", &id_array[i]);
          if (result == EOF) /* fscanf() returns EOF if reading failed */
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "EOF reached before array filled");
               fclose(fp);
               return (1);
          } 
     }
     fscanf(fp, "%s", tag);
     if (strcmp(tag, "END") != 0) /* "END" not found */
     {
          log_to_file(log_file, __FILE__, __LINE__, "couldn't find END tag");
          fclose(fp);
          return (1);
     }
     fclose(fp);
     
     /* deleting content must be done after arc file properly read*/
     
     return (0);
}


void del_sel()
/* Delete content of the sel file. */
{
   FILE *fp;
   fp = fopen(sel_file, "w");
   assert(fp != NULL);
   fprintf(fp, "%d", 0);
   fclose(fp);
}


int write_ini(int *identity)
/* Takes an array of 'alpha' identities and writes the corresponding
   individuals to the the ini file.
   Returns 0 if successful and 1 otherwise.*/
{
     FILE *fp;
     int i,j;
     int min_valid, max_valid;

     if(identity == NULL)
          return(1);
     /* test if identities are valid */
     min_valid = 0;
     max_valid = global_population.last_identity;
     for(i = 0; i < alpha; i++)
     {
          if (identity[i] < min_valid || identity[i] > max_valid 
              || global_population.individual_array[identity[i]] == NULL) 
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "bad id, checked in write_ini");
               return (1);
          } 
     }

     fp = fopen(ini_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d\n", (alpha * (dimension + 1)));
     for (i = 0; i < alpha; i++)
     {
          fprintf(fp, "%d ", identity[i]); /* prints also a space */
          for(j = 0; j < dimension; j++) 
               fprintf(fp, "%E ", get_objective_value(identity[i], j));
          fprintf(fp, "\n");
     }
     fprintf(fp, "END");
     fclose(fp);
     return (0);
}


int write_var(int *identity)
/* Takes an array of 'lambda' identities and writes the corresponding
   individuals to the the var file.
   Returns 0 if successful and 1 otherwise.*/
{
     FILE *fp;
     int i,j;
     int min_valid, max_valid;

     if(identity == NULL)
          return (1);
     /* test if identities are valid */
     min_valid = 0;
     max_valid = global_population.last_identity;
     for(i = 0; i < lambda; i++)
     {
          if (identity[i] < min_valid || identity[i] > max_valid 
              || global_population.individual_array[identity[i]] == NULL) 
          {
               log_to_file(log_file, __FILE__, __LINE__, 
                           "bad ID, checked in write_var");
               return(1);
          } 
     }

     fp = fopen(var_file, "w");
     assert(fp != NULL);
     fprintf(fp, "%d\n", (lambda * (dimension+1)));  
     for (i = 0; i < lambda; i++)
     {
          fprintf(fp, "%d ", identity[i]); /* prints also a space */
          for(j = 0; j < dimension; j++) 
               fprintf(fp, "%E ", get_objective_value(identity[i],j));
          fprintf(fp, "\n");
     }
     fprintf(fp, "END");
     fclose(fp);
     return (0);
}


int log_to_file(char *file, char *infile, int linenumber, char *string)
/* Can be used for debugging purpose.  Writes the following to the
   logfile 'file':
   - current date and time
   - 'infile': the name of the source file which produced the error
   - 'linenumber'
   - 'string': the error message.
   If 'file == NULL' the same information is written to stderr.
   'infile' can be specified as 'NULL' and 'linenumber' can be set to
   -1 in order to leave them out.
*/
{
     FILE *fp;
     struct tm   *curr_date;
     char date_string[64];
     time_t now = time(NULL);
     curr_date = (struct tm*)localtime(&now);
     if (curr_date != NULL)
     {
          strftime(date_string, sizeof (date_string), "%d.%m.%Y %H:%M:%S",
                   curr_date);
     }
     else /* no data available */
     {
          strcpy(date_string, "no date");
     }
     
     if(file != NULL)
     {
          fp= fopen(file, "a");
          assert(fp != NULL);

          if(infile != NULL && linenumber != -1)
          { 
               fprintf(fp, "%s in file %s at line %d: %s\n", date_string,
                       infile, linenumber, string);
          }
          else if(infile != NULL && linenumber == -1)
          {
               fprintf(fp, "%s in file %s: %s\n", date_string, infile,
                       string);
          }
          else if(infile == NULL && linenumber != -1)
          {
               fprintf(fp, "%s at line %d: %s\n", date_string, linenumber,
                       string); 
          } 
          else
          {
               fprintf(fp, "%s: %s\n", date_string, string); 
          }

          fclose(fp);
     }
     else
     {
          fprintf(stderr, "%s at line %d: %s\n", 
                  date_string, linenumber, string);
     }
     return (0);
}
