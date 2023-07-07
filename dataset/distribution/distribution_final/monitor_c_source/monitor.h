/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Pisa basic functions for use in monitor_user.c
  
  Header file.
  
  file: monitor.h
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch

  last change: $date$
  
  ========================================================================
*/

#ifndef MONITOR_H
#define MONITOR_H

/*-----------------------| global common parameters |---------------------------*/

extern int alpha; /* number of individuals in initial population */
extern int mu; /* number of individuals selected as parents */
extern int lambda; /* number of offspring individuals */
extern int dimension; /* number of objectives */
extern double poll; /* poll interval of monitor */

/*---------| global parameter and output file names |---------------------------*/

extern char filenamebase_variator[];
extern char filenamebase_selector[];
extern char parnamebase_selector[];
extern char parnamebase_variator[];
extern char filenamebase_monitor[];
extern char parnamebase_monitor[];
extern char cfg_file_variator[];  
extern char ini_file_variator[];  
extern char sel_file_variator[]; 
extern char arc_file_variator[]; 
extern char var_file_variator[]; 
extern char sta_file_variator[]; 
extern char cfg_file_selector[];  
extern char ini_file_selector[];  
extern char sel_file_selector[]; 
extern char arc_file_selector[]; 
extern char var_file_selector[]; 
extern char sta_file_selector[]; 

/*-----------------------| global variables |---------------------------*/

extern int currentRun;
extern int currentGeneration;

/*-------------------------| functions |-----------------------------------*/

void resetAll(); /* resets variator and selector, creates first archive, 
				 sample and offspring; after executing, the state of the selector
				 is 2 (has written 2) and that of the variator is 3 */

void copyInitialPop(); /* copies the initial population from the variator
				 to the selector and records the transferred data; the function
				 uses move_ini() defined in monitor.c and state1_user() 
				 defined in monitor_user.c. */

void copyArchiveSelected(); /* copies the archive and selected elements
				 from the selector to the variator and records the 
				 archive; the function uses move_sel() and move_arc() defined
				 in monitor.c and state2_user() defined in monitor_user.c. */

void copyVariate(); /* copies the variated elements from the variator to
			     the selector and records them; the function uses move_var() 
                 defined in monitor.c and state3_user() defined in 
                 monitor_user.c. */

/*-------------------------| io |---------------------------------------*/

int* move_arc();
void move_sel(); 
double** move_ini();
double** move_var();

#endif /* MONITOR.H */
