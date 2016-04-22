#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <xmmintrin.h>

static const int NB = 63;
static const int MU = 4;
static const int NU = 2;

static const int MAX_MATRIX_NUMBER = 30;

double *generateRandomMatrixOfSize(int size);
double *matrixOfZerosWithSize(int size);
int matrixMultiply(const double *const sourceA, 
    const double *const sourceB, 
    double *const destination, 
    const int size);
void printMatrix(const double *const m, const int size, const char *name);
double *copyA(const double *const m, const int size, const int start);
double *copyB(const double *const m, const int size, const int start);
double *copyC(const double *const m, const int size, const int start_i, const int start_j);
double rtclock();

int main(int argc, const char* argv[])
{
  if (argc >= 2) {
    int m_size = atoi(argv[1]);
    int debug = 0, retVal = 0;
    double startTime, endTime, time;

    //Get needed info for the multiplication
    if (argc == 3 && strncmp(argv[2], "-d", 2) == 0) debug = 1;
    double * __attribute__((aligned(16))) a = generateRandomMatrixOfSize(m_size);
    double * __attribute__((aligned(16))) b = generateRandomMatrixOfSize(m_size);
    double * __attribute__((aligned(16))) c = matrixOfZerosWithSize(m_size);

    //Multiply
    startTime = rtclock();
    retVal = matrixMultiply(a, b, c, m_size);
    endTime = rtclock();

    //Print Flops
    time = endTime - startTime;
    printf("Time: %0.02f\n", time);
    printf("GFLOPS: %.02f\n", ((2.0*pow(m_size, 3))/time)/1000000000.0);

    //Debug info
    if (debug) {
      printf("\nDebug-------\n");
      printf("NB: %d\n", NB);
      printf("NU: %d\n", NU);
      printf("MU: %d\n", MU);
      printf("\n");
      printMatrix(a, m_size, "A");
      printMatrix(b, m_size, "B");
      printMatrix(c, m_size, "C");
      printf("-------End Debug\n");
    }

    //Free all memory used
    free(a);
    free(b);
    free(c);
    return retVal;
  }
  printf("Patricio Beltran \n"
      "PMB784 \n"
      "Fast Matrix Multiply \n"
      "------------ \n"
      "Usage: ./mmm <matrix size> [-d]\n\n"
      "d:\t Use this flag to print debugging info.\n");
  return 1;
}

int matrixMultiply(const double *const __restrict__ sourceA, 
    const double *const __restrict__ sourceB, 
    double *const __restrict__ destination, 
    const int size)
{
  //__assume_aligned(sourceA, 16); 
  //__assume_aligned(sourceB, 16); 
  //__assume_aligned(destination, 16);

  const int cleanup_i = size % MU != 0;
  const int cleanup_j = size % NU != 0;
  const int m_size = fmin(size, NB);
  int outer_j, outer_i, outer_k, j, i, k;
  for(outer_j = 0; outer_j < size; outer_j+=NB)
  {
    for(outer_i = 0; outer_i < size; outer_i+=NB)
    {
      for(outer_k = 0; outer_k < size; outer_k+=NB)
      {
        const int max_j = fmin(outer_j+NB, size-NU+1);
        for (j = outer_j; j < max_j; j+=NU)
        {
          const double *const copy_B __attribute__((aligned(16))) = copyB(sourceB, m_size, j);
          const int max_i = fmin(outer_i+NB, size-MU+1);
          for (i = outer_i; i < max_i; i+=MU)
          {
            const double *const copy_A __attribute__((aligned(16))) = copyA(sourceA, m_size, i);
            double *const copy_C __attribute__((aligned(16))) = copyC(destination, m_size, i, j);

            register __m128d C1 = _mm_load_pd(&copy_C[0]);
            register __m128d C2 = _mm_load_pd(&copy_C[2]);
            register __m128d C3 = _mm_load_pd(&copy_C[4]);
            register __m128d C4 = _mm_load_pd(&copy_C[6]);

            const int max_k = fmin(outer_k+NB, size);
            for (k = outer_k; k < max_k; ++k)
            {
              register __m128d A1 = _mm_load1_pd(&copy_A[((k-outer_k)*MU)]);
              register __m128d A2 = _mm_load1_pd(&copy_A[((k-outer_k)*MU)+1]);
              register __m128d A3 = _mm_load1_pd(&copy_A[((k-outer_k)*MU)+2]);
              register __m128d A4 = _mm_load1_pd(&copy_A[((k-outer_k)*MU)+3]);

              register __m128d B = _mm_load_pd(&copy_B[2*(k-outer_k)]);

              register __m128d rC1 = _mm_mul_pd(A1,B);
              register __m128d rC2 = _mm_mul_pd(A2, B);
              register __m128d rC3 = _mm_mul_pd(A3, B);
              register __m128d rC4 = _mm_mul_pd(A4, B);
              
              C1 = _mm_add_pd(C1,rC1);
              C2 = _mm_add_pd(C2,rC2);
              C3 = _mm_add_pd(C3,rC3);
              C4 = _mm_add_pd(C4,rC4);
            }
            _mm_store_pd(&copy_C[0], C1);
            _mm_store_pd(&copy_C[2], C2);
            _mm_store_pd(&copy_C[4], C3);
            _mm_store_pd(&copy_C[6], C4);
            destination[((i)*size)+(j)] = copy_C[0];
            destination[((i)*size)+(j+1)] = copy_C[1];
            destination[((i+1)*size)+(j)] = copy_C[2];
            destination[((i+1)*size)+(j+1)] = copy_C[3];
            destination[((i+2)*size)+(j)] = copy_C[4];
            destination[((i+2)*size)+(j+1)] = copy_C[5];
            destination[((i+3)*size)+(j)] = copy_C[6];
            destination[((i+3)*size)+(j+1)] = copy_C[7];
            
            free((void *)copy_A);
            free(copy_C);
          }
          if (cleanup_i)
          {
            int ci, cj;
            for(cj = j; cj < j+NU; ++cj)
            {
              for (ci = i; ci < size; ++ci)
              {
                register double C = destination[((ci)*size)+cj];
                const int max_k = fmin(outer_k+NB, size);
                for (k = outer_k; k < max_k; ++k)
                {
                  register double const A = sourceA[((ci)*size)+k];
                  register double const B = sourceB[(k*size)+cj];
                  C += A*B;
                }
                destination[((ci)*size)+cj] = C;
              }
            }
          }
          free((void *)copy_B);
        }
        if (cleanup_j)
        {
          int ci, cj;
          for(cj = j; cj < size; ++cj)
          {
            for (ci = outer_i; ci < size; ++ci)
            {
              register double C = destination[((ci)*size)+cj];
              const int max_k = fmin(outer_k+NB, size);
              for (k = outer_k; k < max_k; ++k)
              {
                register double const A = sourceA[((ci)*size)+k];
                register double const B = sourceB[(k*size)+cj];
                C += A*B;
              }
              destination[((ci)*size)+cj] = C;
            }
          }
        }
      }
    }
  }
  return 0;
}

double *copyA(const double *const m, const int size, const int start)
{
  int i, j;
  double *res __attribute__((aligned(16))) = (double*)malloc(MU*size*sizeof(double));
  for(j = 0; j < MU; ++j)
    for(i = 0; i < size; ++i)
      res[(i * MU) + j] = m[((start + j) * size) + i];
  
  return res;
}

double *copyB(const double *const m, const int size, const int start)
{
  int i;
  double *res __attribute__((aligned(16))) = (double*)malloc(NU*size*sizeof(double));
  for(j = 0; j < NU; ++j)
    for(i = 0; i < size; ++i)
      res[(NU * i) + j] = m[(i * size) + start + j];
  
  return res;
}

double *copyC(const double *const m, const int size, const int start_i, const int start_j)
{
  int i, j;
  double *res __attribute__((aligned(16))) = (double*)malloc(MU*NU*sizeof(double));
  for(i = 0; i < MU; ++i)
    for(j = 0; j < NU; ++j)
      res[(i*NU) + j] = m[((start_i + i) * size)+(start_j + j)];
  
  return res;
}

double *generateRandomMatrixOfSize(int size)
{
  int i, j;
  double *m __attribute__((aligned(16))) = (double*) malloc(size*size*sizeof(double));
  srand((unsigned int)time(NULL));
  for (i = 0; i < size; ++i)
    for (j = 0; j < size; ++j)
      m[(i*size)+j] = rand() % (MAX_MATRIX_NUMBER + 1);
  
  return m;
}

double *matrixOfZerosWithSize(int size)
{
  int i, j;
  double *m __attribute__((aligned(16))) = (double*) malloc(size*size*sizeof(double));
  for (i = 0; i < size; ++i)
    for (j = 0; j < size; ++j)
      m[(i*size)+j] = 0.0;
  
  return m;
}

void printMatrix(const double *const m, const int size, const char *name)
{
  int i, j;
  printf("%s\n", name);
  for (i = 0; i < size; ++i) 
  {
    printf("| ");
    for (j = 0; j < size; ++j) printf("%.0f ", m[(i*size)+j]);
    printf("|\n");
  }
  printf("\n");
}

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}
