/*===========================================================================*
 * dominance-rank.cc: implements the dominance ranking method as proposed in the
 *              performance assessment tutorial by J. Knowles, L. Thiele, and
 *              E. Zitzler, available as TIK-Report No 214 
 *
 * Compile:
 *   g++ -lm -o dominance-rank dominance-rank.cc
 *
 * Usage:
 *   dominance-rank [<param_file>] <data_file1> <data_file2> <output_file>
 *
 *   <param_file> specifies the name of the parameter file for dominance-rank; the
 *     file has the following format:
 *
 *       dim <integer>
 *       obj <+|-> <+|-> ...
 *
 *     The first line defines the number of objectives, and the second
 *     for each objective whether it is minimized (-) or maximized.
 *     If the parameter file is omitted, the number of objectives is
 *     determined from the data file and it is assumed that all objectives
 *     are to be minimized.
 *
 *   <data_file> specifies a file that contains the output of one or
 *     several runs of a selector/variator pair; the format corresponds to
 *     the one defined in the specification of the PISA monitor program.
 *
 *   <data_file2> specifies the same for the second set of approximation sets. 
 *
 *   <output_file> defines the name of the file to which the computed
 *     indicator values are written to.
 *
 * Authors:
 *   Eckart Zitzler, February 3, 2005
 *   Joshua Knowles, June 21, 2005
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define error(X,Y)  if (X) fprintf(stderr, Y "\n"), exit(1)

#define MAX_LINE_LENGTH  2048 /* maximal length of lines in the files */
#define MAX_STR_LENGTH  256 /* maximal length of strings in the files */

int  dim;  /* number of objectives */
int  *obj;  /* obj[i] = 0 means objective i is to be minimized */
int  method;  /* 0 = additive, 1 = multiplicative */


double  calc_ind_value(double  *a, int  size_a, double  *b, int  size_b)
{
    int  i, j, k;
    double  eps, eps_j, eps_k, eps_temp;

    if (method == 0)
	eps = DBL_MIN;
    else
	eps= 0;
    
    for (i = 0; i < size_a; i++) {
	for (j = 0; j < size_b; j++) {
	    for (k = 0; k < dim; k++) {
		switch (method) {
		case 0:
		    if (obj[k] == 0)
			eps_temp = b[j * dim + k] - a[i * dim + k];
		    else
			eps_temp = a[i * dim + k] - b[j * dim + k];
		    break;
		default:
		    error((a[i * dim + k] < 0 && b[j * dim + k] > 0) ||
			  (a[i * dim + k] > 0 && b[j * dim + k] < 0) ||
			  a[i * dim + k] == 0 || b[j * dim + k] == 0,
			  "error in data file");
		    if (obj[k] == 0)
			eps_temp = b[j * dim + k] / a[i * dim + k];
		    else
			eps_temp = a[i * dim + k] / b[j * dim + k];
		    break;
		}
		if (k == 0)
		    eps_k = eps_temp;
		else if (eps_k < eps_temp)
		    eps_k = eps_temp;
	    }
	    if (j == 0)
		eps_j = eps_k;
	    else if (eps_j > eps_k)
		eps_j = eps_k;
	}
	if (i == 0)
	    eps = eps_j;
	else if (eps < eps_j)
	    eps = eps_j;
    }
    
    return eps;
}

void  read_params(FILE  *fp)
{
    char str[MAX_STR_LENGTH];
    int  i;
    
    fscanf(fp, "%s", str);
    error(strcmp(str, "dim") != 0, "error in parameter file");
    fscanf(fp, "%d", &dim);
    error(dim <= 0, "error in parameter file");
    obj = (int *)malloc(dim * sizeof(int));
    error(obj == NULL, "memory overflow");

    fscanf(fp, "%s", str);
    error(strcmp(str, "obj") != 0, "error in parameter file");
    for (i = 0; i < dim; i++) {
	fscanf(fp, "%s", str);
	error(str[0] != '-' && str[0] != '+', "error in parameter file");
	if (str[0] == '-')
	    obj[i] = 0;
	else
	    obj[i] = 1;
    }

/*    fscanf(fp, "%s", str);

    error(strcmp(str, "method") != 0, "error in parameter file");
    fscanf(fp, "%d", &method);
    error(method != 0 && method != 1, "error in parameter file");*/
    method = 0;
}

void  set_params(void)
{
    int  i;
    
    obj = (int *)malloc(dim * sizeof(int));
    error(obj == NULL, "memory overflow");
    for (i = 0; i < dim; i++)
	obj[i] = 0;
    method = 0;
}

void  check_file(FILE  *fp, int  *no_runsp, int  *max_pointsp)
    /* determines the maximum number of points and the number of runs
       for the data resp. the reference set file; if the array v is
       specified, the data read in will be stored in v
    */
{
    char  line[MAX_STR_LENGTH];
    int  i, j;
    int  new_run;
    int  no_points;
    double  number;

    no_points = 0;
    *max_pointsp = 0;
    *no_runsp = 0;
    new_run = 1;
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
	if (sscanf(line, "%lf", &number) != 1)
	    new_run = 1;
	else {
	    if (new_run == 1)
	    {
		(*no_runsp)++;
		if (*max_pointsp < no_points)
		    *max_pointsp = no_points;
		no_points = 0;
	    }
	    new_run = 0;
	    i = 0;
	    for (j = 1; j < dim; j++) {
		while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		    i++;
		error(sscanf(&(line[i]), "%lf", &number) <= 0,
		      "error in data or reference set file");
		while (line[i] == ' ' && line[i] != '\0')
		    i++;
	    }
	    no_points++;
	}
    }
    if (*max_pointsp < no_points)
	*max_pointsp = no_points;
}

int  determine_dim(FILE  *fp)
{
    char  line[MAX_STR_LENGTH];
    int  i, no_obj;
    int  line_found, number_found;
    double  number;
    
    no_obj = 0;
    line_found = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL && !line_found)
        line_found = sscanf(line, "%lf", &number);
    if (line_found) {
	i = 0;
	do {
	    no_obj++;
	    while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		i++;
	    number_found = sscanf(&(line[i]), "%lf", &number);
	    while (line[i] == ' ' && line[i] != '\0')
		i++;
	} while (number_found == 1);
    }
    
    return no_obj;
}

void  read_file(FILE  *fp, int  *no_pointsp, double  *points)
{
    char  line[MAX_STR_LENGTH];
    int  i, j, k;
    int  reading;
    double  number;

    k = 0;
    reading = 0;
    *no_pointsp = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
        if (sscanf(line, "%lf", &number) != 1) {
	    if (reading)
	        break;
	}
	else {
	    reading = 1;
	    points[k++] = number;
	    i = 0;
	    for (j = 1; j < dim; j++) {
		while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		    i++;
		error(sscanf(&(line[i]), "%lf", &number) <= 0,
		      "error in data or reference set file");
		points[k++] = number;
		while (line[i] == ' ' && line[i] != '\0')
		    i++;
	    }
	    (*no_pointsp)++;
	}
    } 
}

int  main(int  argc, char  *argv[])
{
    int  i; int j; int k;
    int  no_runs;  /* number of runs */
    int  max_points;  /* maximum number of points per run */
    int  curr_run_size;  /* number of points associated with the current run */
    double  *curr_run; /* objective vectors fur current run */

    int  no_runs2;  /* number of runs */
    int  max_points2;  /* maximum number of points per run */
    int  curr_run_size2;  /* number of points associated with the current run */
    double  *curr_run2; /* objective vectors fur current run */

    double  ind_value;

    int *A;
    int *B;


    FILE  *fp, *out_fp, *fp2;
    
    error(argc != 4 && argc != 5,
	  "dominance rank - wrong number of arguments:\ndominance-rank [parFile] datFile refSet outFile");

    /* set parameters */
    if (argc == 5) {
	fp = fopen(argv[1], "r");
	error(fp == NULL, "parameter file not found");
	read_params(fp);
	fclose(fp);
    }
    else {
	fp = fopen(argv[1], "r");
	error(fp == NULL, "data file not found");
	dim = determine_dim(fp);
	error(dim < 1, "error in data file");
	fclose(fp);
	obj = (int *)malloc(dim * sizeof(int));
	error(obj == NULL, "memory overflow");
	for (i = 0; i < dim; i++)
	    obj[i] = 0;
	method = 0;	
    }

    /* read reference set 
    if (argc == 5)
	fp = fopen(argv[3], "r");
    else
	fp = fopen(argv[2], "r");
    error(fp == NULL, "reference set file not found");
    check_file(fp, &no_runs, &max_points);
    error(no_runs != 1 || max_points < 1, "error in reference set file");
    ref_set = malloc(dim * max_points * sizeof(double));
    error(ref_set == NULL, "memory overflow");
    rewind(fp);
    read_file(fp, &ref_set_size, ref_set);
    fclose(fp);
    no_runs = 0;
    max_points = 0; */
    

    if (argc == 5)
	fp = fopen(argv[2], "r");
    else
	fp = fopen(argv[1], "r");
    error(fp == NULL, "data file not found");
    check_file(fp, &no_runs, &max_points);
    error(no_runs < 1 || max_points < 1, "error in data file");
    //    curr_run = (double *)malloc(dim * max_points * sizeof(double));
    rewind(fp);



    /* check second data file */
    if (argc == 5)
	fp2 = fopen(argv[3], "r");
    else
	fp2 = fopen(argv[2], "r");
    error(fp2 == NULL, "data file not found");
    check_file(fp2, &no_runs2, &max_points2);
    error(no_runs2 < 1 || max_points2 < 1, "error in data file");

    if(max_points2<max_points)
      max_points2=max_points;
    curr_run = (double *)malloc(dim * max_points2 * sizeof(double));
    curr_run2 = (double *)malloc(dim * max_points2 * sizeof(double));
    rewind(fp2);

    A = (int *)malloc(no_runs*sizeof(int));
    B = (int *)malloc(no_runs2*sizeof(int));
    for(i=0;i<no_runs;i++)
      A[i]=0;
    for(i=0;i<no_runs2;i++)
      B[i]=0;
  
  
    /* process data */
    if (argc == 5)
	out_fp = fopen(argv[4], "w");
    else
	out_fp = fopen(argv[3], "w");
    error(out_fp == NULL, "output file could not be generated");
    for (i=0;i<no_runs; i++) {
      for(k=0;k<i;k++)
	read_file(fp, &curr_run_size, curr_run); // read and discard
      read_file(fp, &curr_run_size, curr_run); // read and store
      rewind(fp); 
      for(j=0;j<no_runs;j++){
	read_file(fp, &curr_run_size2, curr_run2);
	ind_value = calc_ind_value(curr_run, curr_run_size,
				   curr_run2, curr_run_size2);
	// fprintf(out_fp, "%d %d %.9e\n", i, j, ind_value);
	if(ind_value<0)
	  A[i]++;
	ind_value = calc_ind_value(curr_run2, curr_run_size2,
				   curr_run, curr_run_size);
	fprintf(out_fp, "%d %d %.9e\n", i, j, ind_value);
      }
      for(j=0;j<no_runs2;j++){
	read_file(fp2, &curr_run_size2, curr_run2);
	ind_value = calc_ind_value(curr_run, curr_run_size,
				   curr_run2, curr_run_size2);
	if(ind_value<0)
	  A[i]++;
	//	fprintf(out_fp, "%d %d %.9e\n", i, j, ind_value);
	ind_value = calc_ind_value(curr_run2, curr_run_size2,
				   curr_run, curr_run_size);
	//	fprintf(out_fp, "%d %d %.9e\n", i, j, ind_value);
      }
      rewind(fp);
      rewind(fp2);
    }
    
    for (i=0;i<no_runs2; i++) {
      for(k=0;k<i;k++)
	read_file(fp2, &curr_run_size, curr_run); // read and discard
      read_file(fp2, &curr_run_size, curr_run); // read and store
      rewind(fp2); 
      for(j=0;j<no_runs;j++){
	read_file(fp, &curr_run_size2, curr_run2);
	ind_value = calc_ind_value(curr_run, curr_run_size,
				   curr_run2, curr_run_size2);
	//	fprintf(out_fp, "%.9e\n", ind_value);
	if(ind_value<0)
	  B[i]++;
	ind_value = calc_ind_value(curr_run2, curr_run_size2,
				   curr_run, curr_run_size);
	//	fprintf(out_fp, "%.9e\n", ind_value);
      }
      for(j=0;j<no_runs2;j++){
	read_file(fp2, &curr_run_size2, curr_run2);
	ind_value = calc_ind_value(curr_run, curr_run_size,
				   curr_run2, curr_run_size2);
	if(ind_value<0)
	  B[i]++;
	//	fprintf(out_fp, "%.9e\n", ind_value);
	ind_value = calc_ind_value(curr_run2, curr_run_size2,
				   curr_run, curr_run_size);
	//	fprintf(out_fp, "%.9e\n", ind_value);
      }
      rewind(fp);
      rewind(fp2);
    }

    fclose(out_fp);
    fclose(fp);
    fclose(fp2);



    for(i=0;i<no_runs;i++)
      printf("%d\n", A[i]);
    
    printf("\n");
    for(i=0;i<no_runs2;i++)
      printf("%d\n", B[i]);
   

    return 0;
}
