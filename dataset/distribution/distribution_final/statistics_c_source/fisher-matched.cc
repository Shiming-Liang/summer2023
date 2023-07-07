
/* fisher-matched.cc  (C) Joshua Knowles, 2005

Implements a nonparametric test for differences between two paired (or matched) samples,
as described in W.J.Conover (1999) "Practical Nonparametric Statisticsn (3rd Edition)", Wiley.

This version accepts multiple (more than two) samples. In that case, the test is
carried out for each pair and a warning, which advises the p-values are not
accurate because the samples are no longer independent random samples, is issued.



Compile:
   g++ fisher-matched.cc -o fisher-matched -lm

Run:
   ./fisher_matched <indicator_file> <paramfile> <outputfile>
   
   where:

   <indicator_file> is the name of a file containing a single column of
     indicator values. Blank lines in the file divide the sample
     populations. All the sample populations must be the same size, and ordered
     so that the entries are matched pairs;
   <param_file> is the name of a file with precisely two lines, as follows:

      nresamples 5000
      seed 6756519834

     where nresamples specifies an integer number of randomizations of the 
       data to carry out in each test, and
     seed is a long integer to seed the random number generator;
   <output_file> is a filename to write to.

Output:

   The output is the one-tailed p-value that the null hypothesis is true for each pair.
   If the normal approximation has been used then this is indicated in brackets.

   A warning message is output if there are multiple (more than two) samples.

   With VERBOSE set to true, some output to stdout is also given.

*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// using namespace std;

#define RN rand()/(RAND_MAX+1.0)
#define MAX_STR_LENGTH 100
#define MAX_LINE_LENGTH 100
#define MAX_DISTS 30
#define VERBOSE true

typedef struct data
{
  double value;
  int label;
}D;

typedef struct pairs
{
  double diff; // signed difference
  int dummysign;  // -1 or +1
}P;

P *p;
D *d;

long int seed;
int NR; // resampling parameter
int N; // the total number of values in the input
int *Nsamp; // the number of values in each sample population
int ndist; // the number of sample populations



FILE *fp;


double myabs(double v);
void  check_file(FILE  *fp, int  *no_runsp, int  *max_pointsp, int *Nsamp);
void  read_file(FILE  *fp, int  *no_pointsp, D *d);

int main(int argc, char **argv)
{
    int i;

    char strseed[10];

  if(argc!=4)
    {
      fprintf(stderr,"./fisher-matched <indicator_file> <param_file> <output_file>\n");
      exit(1);
    }

  if((fp=fopen(argv[2],"rb")))
    {
      if(fscanf(fp, "%*s %d\n", &NR)==EOF)  
    fprintf(stderr, "Error occurred in parameter file.\n"), exit(1);
      
      if(fscanf(fp, "%*s %10s\n", strseed)==EOF)
    fprintf(stderr, "Error occurred in parameter file.\n"), exit(1);
      fclose(fp);
    }
  else
    {
      fprintf(stderr, "Couldn't open param file %s for reading.\n", argv[2]);
      exit(1);
    }

  seed = atol(strseed);
  srand(seed);
  

  Nsamp = (int *)malloc(MAX_DISTS*sizeof(int));
  for(int j=0;j<MAX_DISTS;j++) Nsamp[j] = 0;

  if((fp=fopen(argv[1],"rb")))
    {
      check_file(fp, &ndist, &N, Nsamp);
      if(VERBOSE)
    fprintf(stdout, "Number of samples (populations) = %d. Total number of values in the input = %d\n", ndist, N);
      rewind(fp);
      if(VERBOSE)
    {
      fprintf(stdout,"Numbers of values in each sample = ");
      for(int j=0;j<ndist;j++)
        fprintf(stdout,"%d ", Nsamp[j]);
      fprintf(stdout,"\n");
    }
      d = (D *)malloc(N *sizeof(D));
      p = (P *)malloc(Nsamp[0] *sizeof(P));
      read_file(fp, &N, d);
      fclose(fp);
    }
  else
    {
      fprintf(stderr,"Couldn't open %s for reading\n", argv[1]);
      exit(1);
    }

  if((fp=fopen(argv[3],"w")))
    fclose(fp);
  else
    {
      fprintf(stderr,"Couldn't open %s for writing.\n", argv[2]);
      exit(1);
    }

  for(int a=0;a<ndist;a++)
    {
      for(int b=0;b<ndist;b++)
    {
      if(a==b)
        continue;
      // check for pairs that do not differ (ties) and remove them
      int ties=0;
      for(i=0;i<Nsamp[0]; i++)
        {
          p[i-ties].diff = d[Nsamp[0]*b+i].value-d[Nsamp[0]*a+i].value;
          if(VERBOSE)
        fprintf(stdout, "%g %g ", d[Nsamp[0]*a+i].value,d[Nsamp[0]*b+i].value);
          if(VERBOSE)
        fprintf(stdout, "%g\n", p[i].diff);
          if(p[i-ties].diff==0)
        {
          ties++;
        }
        }
      
      int n=Nsamp[0]-ties;


      if(VERBOSE)
        {
          fprintf(stdout, "__diff__\tabs_diff\n");
          for(int i=0;i<n;i++)
        {
          fprintf(stdout, "%8g\t%8g\n", p[i].diff, myabs(p[i].diff));
          
        }
        }


      double T;
      double T2=0.0;
  
      for(i=0;i<n;i++)
        {
          if(p[i].diff>0)
        {
          // fprintf(stdout, "+%g\n",p[i].diff);
          T2 += p[i].diff;  // Equation 2, page 412 of Conover (1999)
        }
        }
      int lowerc=0;
      int upperc=0;
      for(int nr=0;nr<NR;nr++)
        {
          T=0;
          for(int i=0;i<n;i++)
        {
          if(RN<0.5)
            T+= myabs(p[i].diff);
        }
          if(T > T2)
        upperc++;
          else if (T < T2)
        lowerc++;
          // printf("T=%g, T2 = %g\n", T, T2);
        }
      double upperp, lowerp;

      lowerp = (double)upperc/double(NR);
      upperp = (double)lowerc/double(NR);

      double pvalue;
      bool zero=false;

      if(upperp==0)
        zero=true;


      if(upperp<lowerp)
        pvalue = 2.0*upperp;
      else
        pvalue = 2.0*lowerp;

      
      // one-tailed pvalue:
      pvalue = upperp;

      if(zero)
        pvalue = 2.0*1.0/pow(2.0,n);


      if(VERBOSE)
        fprintf(stdout, "The one-tailed p-value for accepting the null hypothesis that the expected value of the difference is zero is p=%g\n", pvalue);
      if((fp=fopen(argv[3],"a")))
        {
          if(!zero)
        fprintf(fp, "%d better than %d with a p-value of %g\n", b+1, a+1, pvalue);
          else
        fprintf(fp, "%d better than %d with a p-value < %g\n", b+1, a+1, pvalue);
          fclose(fp);     
        }
      else
        {
          fprintf(stderr,"Couldn't open output file for writing.\n");
          exit(1);
        }
      
    }
    }
  
  if(ndist>2)
    fprintf(stderr, "Warning: the p-values for accepting the null hypothesis that the expected differences are zero, are not correct because multiple tests have been carried out using the same sample. Therefore, these values should only be used in preliminary (explorative) tests, and do not indicate true probabilities. Consider collecting new, independent random samples for each statistical test to be performed.\n");
  
  return(0);
  
}

inline void double2binary(double n, int *bin, int dim)
{
  int i;
  int dig;
 
  for(i=0;i<dim;i++)
    {
      dig=int(pow(2.0,dim-1-i));
      //      printf("dig=%d\n",dig);
      if(n>=dig)
        {
          n-=dig;
          bin[i]=1;
        }
      else
        bin[i]=0;
    }
}


double myabs(double v)
{
  if(v>=0)
    return v;
  else
    return -v;
}


void  check_file(FILE  *fp, int  *no_runsp, int  *totalp, int *Nsamp)
{
  char  line[MAX_STR_LENGTH];
  double  number;
  int new_run;
  
  *totalp = 0;
  *no_runsp = 0;
  new_run = 1;

  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
      if (sscanf(line, "%lf", &number) != 1) {
            new_run = 1;
      } else {
            if (new_run == 1) (*no_runsp)++;
            new_run = 0;
            (*totalp)++;
            if(*no_runsp<=MAX_DISTS) {
                (Nsamp[*no_runsp-1])++;
            } else {
                fprintf(stderr,"Please edit MAX_DISTS. Number of sample distributions exceeded the current setting.\n");
                exit(1);
            }
      }
    
  }
  for(int j=1;j<*no_runsp;j++)
  if(Nsamp[j]!=Nsamp[0])
    {
      fprintf(stderr,"Two samples of indicator values are not of the same size. This program computes statistics for paired samples only. Exiting.\n");
      exit(1);
    }
  
}

void  read_file(FILE  *fp, int  *no_pointsp, D *d)
{
  char  line[MAX_STR_LENGTH];
  double  number;
  int clabel=0;
  
  *no_pointsp = 0;
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) 
    {
      if (sscanf(line, "%lf", &number) != 1)
    {
      clabel++;
    }
      else 
    {
      d[*no_pointsp].value = number;
      d[*no_pointsp].label = clabel;
      (*no_pointsp)++;
    }
    } 
}
