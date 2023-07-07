/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 

  Pisa basic functionality that have to be implemented by the user  
   
  Header file.
  
  file: monitor_user.h
  author: Marco Laumanns, Lothar Thiele, {thiele,laumanns}@tik.ee.ethz.ch

  last change: $date$
  
  ========================================================================
*/

#ifndef MONITOR_USER_H
#define MONITOR_USER_H

#define PISA_UNIX /**** replace with PISA_WIN or PISA_UNIX  if compiling 
						for Windows or UNIX */

/* maximal length of filenames */
#define FILE_NAME_LENGTH 128  /**** change the value if you like */

/* maximal length of entries in local cfg file */
#define CFG_NAME_LENGTH 128   /**** change the value if you like */

/* maximal length of lines in parameter file */
#define LINE_LENGTH 256   /**** change the value if you like */


/*---------| global variables from parameterfile |---------------------*/

extern int seed;
extern int numberOfRuns;
extern int numberOfGenerations;
extern int outputType;
extern int outputSet;
extern int LOG;
#define ALL 1
#define ONLINE 2
#define OFFLINE 3

/*-------------------------| statemachine |-----------------------------*/

void state1_user(double**); 
void state2_user(int*); 
void state3_user(double**); 

/*-------------------------| construct pareto set |---------------------*/

void paretoset_join(double *);
int dominates(double*, double*);

/*-------------------------| read and write |---------------------------*/

void read_local_parameters();
void appendOutput();
void outputAll(FILE*);
void outputOnline(FILE*);
void outputOffline(FILE*);
void updateVariatorSeed();
void printInformation();

int irand(int); // determine integer random number


#endif /* MONITOR_USER.H */
