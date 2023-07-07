 /*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Internal functions, used in monitor.c
  
  C file.
  
  file: monitor_internal.c
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch

  last change: $date$
  
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "monitor.h"
#include "monitor_user.h"
#include "monitor_internal.h"

/* this is needed for the wait function */
#ifdef PISA_UNIX
#include <unistd.h>
#endif

#ifdef PISA_WIN
#include <windows.h>
#endif



/*--------------------| global variable definitions |-------------------*/

/* declared in variator_internal.h used in other files as well */

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



/*-------------------------| functions for handling state file |--------*/

/* Write the state flag */
int write_state(char *filename, int state)
{
     FILE *fp;

     assert(0 <= state <= 11);
     
     fp = fopen(filename, "w");
     assert(fp != NULL);
     fprintf(fp, "%d", state);
     fclose(fp);
     return(0);
}


/* Read state flag */
int read_state(char *filename)
{
     int result;
     int state = -1;
     FILE *fp;
     fp = fopen(filename, "r");
     if (fp != NULL)
     {
          result = fscanf(fp, "%d", &state);
          fclose(fp);
          if (result == 1) /* exactly one element read */
          {
	      assert(state >= 0 && state <= 11);
	  }
     }
     
     return (state);
}


int state_error(int error, int linenumber)
/* Outputs an error message and calls exit. */
{
     printf("error in state %d \n", error);
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



/*-------------------------| other functions |-------------------------*/

int read_common_parameters(char *filename)
/* Reads global parameters from 'cfg' file. */
{
     FILE *fp;
     
     int result;
     int new_alpha;
     int new_mu;
     int new_lambda;
     int new_dimension;     
     char str[CFG_ENTRY_LENGTH_INTERNAL];

     /* reading cfg file with common configurations for both parts */
     fp = fopen(filename, "r");
     assert(fp != NULL);     
 
     fscanf(fp, "%s", str);
     assert(strcmp(str, "alpha") == 0);
     fscanf(fp, "%d", &new_alpha);
     assert(new_alpha > 0);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "mu") == 0);
     fscanf(fp, "%d", &new_mu);
     assert(new_mu > 0);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "lambda") == 0);
     fscanf(fp, "%d", &new_lambda);
     assert(new_lambda > 0);

     result = fscanf(fp, "%s", str);
     assert(strcmp(str, "dim") == 0);
     result = fscanf(fp, "%d", &new_dimension);
     assert(result != EOF); /* no EOF, dim correctly read */
     assert(new_dimension > 0);

     fclose(fp);     

     if (alpha == 0)
     {
	 alpha = new_alpha;
	 mu = new_mu;
	 lambda = new_lambda;
	 dimension = new_dimension;
	 
     }
     else
     {
	 assert(new_alpha == alpha);
	 assert(new_mu == mu);
	 assert(new_lambda == lambda);
	 assert(new_dimension == dimension);
     }
     return (0);
}


int check_file(char* filename)
/* Returns 0 if <filename> contains only '0' and returns 1 otherwise. */
{
     int control_element = 1;

     FILE *fp;

     fp = fopen(filename, "r");
     assert(fp != NULL);
     
     fscanf(fp, "%d", &control_element);
     fclose(fp);
     
     if (0 == control_element)
          return (0); /* file is ready for writing */
     else
          return (1); /* file is not ready for writing */
}
