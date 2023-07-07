/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 

  Helper functions used by main and functions in variator.c
   
  Header file.
  
  file: variator_internal.h
  author: Fabian Landis, flandis@ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
  
  ========================================================================
*/

#ifndef VARIATOR_INTERNAL_H
#define VARIATOR_INTERNAL_H

/*-------------------------| constants |--------------------------------*/

#define FILE_NAME_LENGTH_INTERNAL 128
/* maximal length of filenames */


#define CFG_ENTRY_LENGTH_INTERNAL 128 
/* maximal length of entries in cfg file */


#define STANDARD_SIZE 32200  
/* Start with array of this size for global population */


/*---------------| declaration of global variables |-------------------*/

/* file names - defined in variator_internal.c */

extern char cfg_file[];  
/* 'cfg' file (common parameters) */

extern char ini_file[];  
/* 'ini' file (initial population) */

extern char sel_file[]; 
/* 'sel' file (parents) */

extern char arc_file[]; 
/* 'arc' file (archive) */

extern char var_file[]; 
/* 'var' file (offspring) */

extern char sta_file[]; 
/* 'sta' file (current state) */


/*-----------------| functions for handling states |--------------------*/

int write_state(int state);
/* Write the state flag */


int read_state(void);
/* Read state flag */

int state_error(int error, int linenumber);
/* Outputs an error message and calls exit. */

int wait(double sec);
/* Makes the calling process sleep for 'sec' seconds.
   
  pre: 'sec' >= 0.01
  post: Calling process is sleeping for 'sec' * 1e6 microseconds.
        The requested time is rounded up to the next integer multiple
        of the resolution the system can deliver.

  CAUTION: sleep and usleep() are not standard C, use Sleep(milliseconds)
           in <windows.h> for Windows version.
*/



/*-------------------------| stack |------------------------------------*/

/* stack structure used in global population for the free available ids
   between 0 and last_identity */
typedef struct stack_node_t
{
  int identity;
  struct stack_node_t *next;
} stack_node;

typedef struct stack_t
{
  stack_node *top;
  int size;
} stack;

int free_stack(stack *st);

int push(stack *st, int id);

int pop(stack *st);

/*----------------------| global population |---------------------------*/

/* pool of all existing individuals */

typedef struct population_t
{
  int size; /* size of the population */
  individual **individual_array; /* array of pointers to individuals */
  int last_identity; /* identity of last individual */  
  stack free_ids_stack; /* stack for keeping freed ids with remove */

} population;

/* the only population we need is */
extern population global_population; /* defined in variator_internal.c */

int clean_population(void);
/* Frees memory for all individuals in population and for the global
   population itself. */

int remove_individual(int identity);
/* Removes the individual with ID 'identity' from the global population. */

/*----------------------| other functions |----------------------------*/

int read_common_parameters(void);
/* Reads global parameters from 'cfg' file. */


int cmp_int(const void *p_i1, const void *p_i2);
/* Compares the two integers '*p_i1' and '*p_i2'.
   Returns 0 if *p_i1 == *p_i2.
   Returns 1 if *p_i1 > *p_i2.
   Returns -1 if *p_i1 < *p_i2. */

int check_var(void);
/* Returns 0 if 'var_file' contains only '0'and returns 1 otherwise. */

int check_ini(void);
/* Returns 0 if 'ini_file' contains only '0'and returns 1 otherwise. */


#endif /* VARIATOR_INTERNAL.H */
