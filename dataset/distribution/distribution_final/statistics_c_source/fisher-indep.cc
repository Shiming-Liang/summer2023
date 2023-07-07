/* fisher-indep.cc  (C) Joshua Knowles, 2005

Implements a nonparametric test for differences between precisely two independent samples,
as described in W.J.Conover (1999) "Practical Nonparametric Statistics (3rd Edition)", Wiley.

This version accepts multiple (more than two) samples. In that case, the test is
carried out for each pair and a warning, which advises that the p-values are not
accurate because the samples are no longer independent random samples, is issued.



Compile:
    g++ fisher-indep.cc -o fisher-indep -lm


Run:
   ./fisher-indep <indicator_file> <param_file> <output_file>
   
   where:

   <indicator_file> is the name of a file containing a single column of
     indicator values. Blank lines in the file divide the separate sample
     populations;
   <param_file> is the name of a file with precisely two lines, as follows

      nresamples 5000
      seed 6756519834

     where nresamples specifies an integer number of randomizations of the 
       data to carry out in each test, and
     seed is a long integer to seed the random number generator;
   <output_file> is a filename to write to.


Output:

   The output is the p-value of the one-tailed test for rejecting 
     the null hypothesis of no difference between each pair of sample populations.

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

D *d;
D *pair;
long int seed; // random seed
int NR; // resampling parameter
int N; // the total number of values in the input
int *Nsamp; // the number of values in each sample population
int ndist; // the number of sample populations
FILE *fp;

double choose (int n, int m);
inline void cwr(int *target, int k, int n);
double sum_of_values(D *d, int index, int N);
double myabs(double v);
int compare(const void *, const void *);
void  check_file(FILE  *fp, int  *no_runsp, int  *max_pointsp, int *Nsamp);
void  read_file(FILE  *fp, int  *no_pointsp, D *d);

int main(int argc, char **argv)
{
    int i, j, k, nr;
  int test=1;
  int start=0;
  char strseed[10];
    
  if(argc!=4)
    {
      fprintf(stderr,"./fisher_indep <indicator_file> <param_file> <output_file> \n");
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
  for( j=0;j<MAX_DISTS;j++) Nsamp[j] = 0;

  if((fp=fopen(argv[1],"rb")))
    {
      check_file(fp, &ndist, &N, Nsamp);
      if(VERBOSE)
    fprintf(stdout, "Number of samples (populations) = %d. Total number of values in the input = %d\n", ndist, N);
      rewind(fp);
      if(VERBOSE)
    {
      fprintf(stdout,"Numbers of values in each sample = ");
      for( j=0;j<ndist;j++)
        fprintf(stdout,"%d ", Nsamp[j]);
      fprintf(stdout,"\n");
    }      
      d = (D *)malloc(N *sizeof(D));
      pair = (D *)malloc(N *sizeof(D));
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
      fprintf(stderr,"Couldn't open %s for writing.\n", argv[3]);
      exit(1);
    }
  
  for(j=0;j<ndist;j++)
    {
      for( k=0;k<ndist;k++)
    {
      if(j==k)
        continue;
      if(VERBOSE)
        fprintf(stdout, "\n\n**** Test %d between %d and %d ****\n", test++, j+1, k+1);
      start=0;
      for( i=0;i<j;i++)
        start+=Nsamp[i];
      for(i=0;i<Nsamp[j];i++)
        {
          pair[i].value = d[i+start].value;
          pair[i].label = j;
        }
      start=0;
      for(i=0;i<k;i++)
        start+=Nsamp[i];
      for(i=0;i<Nsamp[k];i++)
        {
          pair[i+Nsamp[j]].value = d[i+start].value;
          pair[i+Nsamp[j]].label = k;
        }

      
      if(VERBOSE)
        {
          for(i=0;i<Nsamp[j]+Nsamp[k];i++)
        {
          fprintf(stdout, "%g %d\n", pair[i].value, pair[i].label);
        }
        }
      // exit(0);
      double sj=sum_of_values(pair, j, Nsamp[j]);
      double sk=sum_of_values(&(pair[Nsamp[j]]), k, Nsamp[k]);
      if(VERBOSE)
        {
          fprintf(stdout, "Number of samples = %d; sum = %g\n", Nsamp[j], sj);
          fprintf(stdout, "Number of samples = %d; sum = %g\n", Nsamp[k], sk);
        }

      double p_value;

//    int idx[Nsamp[j]+Nsamp[k]];
      int *idx;
      idx = (int *)malloc((Nsamp[j]+Nsamp[k])*sizeof(int));

      for(i=0;i<Nsamp[j]+Nsamp[k];i++)
        idx[i]=i;
      
      double sum;
      int nupper=0;
      int nlower=0;
      for( nr=0;nr<NR;nr++)
        {
          cwr(idx, Nsamp[j], Nsamp[j]+Nsamp[k]);
          
          sum=0;
          for(i=0;i<Nsamp[j];i++)
        sum+=pair[idx[i]].value;
          // printf("sum=%g %g\n", sum, sj);

          if(sum > sj)
        nupper++;
          if(sum < sj)
        nlower++;
          
        }
      if(VERBOSE)
        {
          fprintf(stdout, "Upper p-value = %g\n", double(nupper)/(double)NR);
          fprintf(stdout, "Lower p-value = %g\n", double(nlower)/(double)NR);   
        }

      if(nupper<nlower)
        p_value = 2.0*double(nupper)/(double)NR;
      else
        p_value = 2.0*double(nlower)/(double)NR;

      // one-tailed p-value:
      p_value = double(nupper)/(double)NR;

      bool zero=false;
      if(nupper==0)
        {
          zero=true;
          p_value = 1.0/choose(N,Nsamp[j]);
        }

      if(VERBOSE)
        {
          if(!zero)
        fprintf(stdout, "One-tailed p-value = %g\n", p_value);
          else
        fprintf(stdout, "One-tailed p-value < %g\n", p_value);
        }
      if((fp=fopen(argv[3],"a")))
        {    
          if(!zero)
        fprintf(fp, "%d better than %d with a p-value of %g\n", k+1, j+1, p_value);
          else
        fprintf(fp, "%d better than %d with a p-value < %g\n", k+1, j+1, p_value);
          fclose(fp);
        }
      else
        {
          fprintf(stderr, "Couldn't open output file for writing\n");
          exit(1);
        }

      free(idx);
    }
    }
  if(ndist>2)
    fprintf(stderr, "Warning: the p-values for accepting the null hypothesis that these are two samples from the same underlying distribution are not correct because multiple tests have been carried out using the same sample. Therefore, these values should only be used in preliminary (explorative) tests, and do not indicate true probabilities. Consider collecting new, independent random samples for each statistical test to be performed. Alternatively, use the Kruskal-Wallis test.\n");
 
  return(0);
}

double choose (int n, int m)
{
      double cnm = 1;
      int i, f;

      if (m*2 >n) m = n-m;
      for (i=1 ; i <= m; n--, i++)
      {
            if ((f=n) % i == 0)
                  f   /= i;
            else  cnm /= i;
            cnm *= f;
      }
      return cnm;
}

int compare(const void *i, const void *j)
{
  double x;
  x = (*(D *)i).value - (*(D *)j).value;
  
  if(x<0)
    return(-1);

  else if (x>0)
    return(1);
  
  else
    return(0);

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

double sum_of_values(D *d, int index, int N)
{
  double sum=0.0;
  int i;
  
  for( i=0;i<N;i++)
    {
      if(d[i].label == index)
    sum+=d[i].value;
    }

  return(sum);
}

inline void cwr(int *target, int k, int n)  // choose without replacement k items from n
{
  // this function takes target and returns it reordered such that the first k itemshave changed
  int i,j;
  int l_t; //(length of the list at time t)
   
  if(k>n)
    {
      fprintf(stderr,"trying to choose k items without replacement from n but k > n!!\n");
      exit(1);
    }
            
//  int from[n];

  int *from;
  from = (int *)malloc(n*sizeof(int));

  for(i=0;i<n;i++)
    from[i]=i;

  l_t = n;
  for(i=0;i<k;i++)
    {
      j=(int)(RN*l_t);
      target[i]=from[j];
      from[j]=from[l_t-1];
      l_t--;
    }

  free(from);
}
