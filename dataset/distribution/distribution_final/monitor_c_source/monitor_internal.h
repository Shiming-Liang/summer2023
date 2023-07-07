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
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch
  
  last change: $date$
  
  ========================================================================
*/

#ifndef MONITOR_INTERNAL_H
#define MONITOR_INTERNAL_H

/*-------------------------| constants |--------------------------------*/

#define FILE_NAME_LENGTH_INTERNAL 128
/* maximal length of filenames */


#define CFG_ENTRY_LENGTH_INTERNAL 128 
/* maximal length of entries in cfg file */


/*-----------------| functions for handling states |--------------------*/

int write_state(char *filename, int state);
/* Write the state flag */

int read_state(char *filename);
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



/*----------------------| other functions |----------------------------*/

int read_common_parameters(char *filename);
/* Reads global parameters from 'cfg' file. */

int check_file(char *filename);
/* Returns 0 if 'var_file' contains only '0'and returns 1 otherwise. */


#endif /* MONITOR_INTERNAL.H */
