/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Pisa basic functions for use in variator_user.c
  
  Header file.
  
  file: variator.h
  author: Fabian Landis, flandis@ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
  
  ========================================================================
*/

#ifndef VARIATOR_H
#define VARIATOR_H

/*-----------------------| common parameters |---------------------------*/

extern int alpha; /* number of individuals in initial population */

extern int mu; /* number of individuals selected as parents */

extern int lambda; /* number of offspring individuals */

extern int dimension; /* number of objectives */

/*-------------------------| individual |-------------------------------*/

typedef struct individual_t individual; 
/* 'individual_t' has to be defined in variator_user.h */

/*-------------------------| global population |------------------------*/

int add_individual(individual *ind);
/* Takes a pointer to an individual and adds the individual to the
   global population.
   Returns the identity assigned to the individual.
   Returns -1 if adding failed.*/


individual *get_individual(int identity);
/* Returns a pointer to the individual corresponding to 'identity'. */


int get_first();
/* Returns the identity of first individual in the global population. */


int get_next(int identity);
/* Takes an identity an returns the identity of the next following
   individual in the global population. */


int get_size();
/* Returns the size of the global population, i.e. the current number
   of individuals in the population. */


/*-------------------------| io |---------------------------------------*/


int read_arc();
/* Reads 'arc' file, and automatically removes all individuals from
   the global population which are not in the arc file. */

void del_arc();
/* Deletes the content of the arc file. */

int read_sel(int *id_array); 
/* Reads a list of IDs from the sel_file. These IDs denote the
   individuals chosen as parents. The IDs of the individuals are
   stored in 'id_array'.  The number of IDs in this array is
   'mu'. 'id_array' must be big enough to store 'mu' 'int'
   variables.

   If reading is successful function returns 0, otherwise it returns
   1. */

void del_sel();
/* Deletes the content of the sel file. */

int write_ini(int *identity);
/* Takes an array of 'alpha' identities and writes the corresponding
   individuals to the the ini file.
   Returns 0 if successful and 1 otherwise.*/


int write_var(int *identity);
/* Takes an array of 'lambda' identities and writes the corresponding
   individuals to the the var file.
   Returns 0 if successful and 1 otherwise.*/


int log_to_file(char *file, char *infile, int linenumber, char *string);
/* can be used for debugging purpose.  Writes the following to the
   logfile 'file':
   - current date and time
   - 'infile': the name of the source file which produced the error
   - 'linenumber'
   - 'string': the error message.
   If 'file == NULL' the same information is written to stderr.
   'infile' can be specified as 'NULL' and 'linenumber' can be set to
   -1 in order to leave them out. */


#endif /* VARIATOR.H */
