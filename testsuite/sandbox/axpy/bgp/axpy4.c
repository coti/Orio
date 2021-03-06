
void axpy4(int N, double *y, double a1, double *x1, double a2, double *x2, 
	   double a3, double *x3, double a4, double *x4) {

/*@ begin PerfTuning (
 def build {
   arg build_command = 'mpixlc_r -O3 -qstrict -qarch=450d -qtune=450 -qhot -qsmp=omp:noauto';
   arg batch_command = 'qsub -n 128 -t 20 -q short --env "OMP_NUM_THREADS=4"';
   arg status_command = 'qstat';
   arg num_procs = 128;
 }
 def performance_counter {
   arg method = 'bgp counter';
   arg repetitions = 1000;
 }
 def performance_params {
   param UF[] = range(1,33);
   param P[] = [True,False];
 }
 def input_params {
   param N[] = [10,100,1000,10000,50000,100000,500000,1000000,5000000,10000000];
 }
 def input_vars {
   decl dynamic double y[N] = 0;
   decl dynamic double x1[N] = random;
   decl dynamic double x2[N] = random;
   decl dynamic double x3[N] = random;
   decl dynamic double x4[N] = random;
   decl double a1 = random;
   decl double a2 = random;
   decl double a3 = random;
   decl double a4 = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

  int n=N;
  register int i;

/*@ begin Align (x1[],x2[],x3[],x4[],y[]) @*/
/*@ begin Loop (
  transform Unroll(ufactor=UF, parallelize=P) 
    for (i=0; i<=n-1; i++)
      y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];
) @*/

 for (i=0; i<=n-1; i++)
   y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];

/*@ end @*/
/*@ end @*/
/*@ end @*/

}
