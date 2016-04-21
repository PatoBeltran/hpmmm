#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <xmmintrin.h>

static const int NB = 63;
static const int MU = 4;
static const int NU = 1;

static const int MAX_MATRIX_NUMBER = 30;

double *generateRandomMatrixOfSize(int size);
double *matrixOfZerosWithSize(int size);
int matrixMultiply(const double *const sourceA, 
                   const double *const sourceB, 
                   double *const destination, 
                   const int size);
void printMatrix(const double *const m, const int size, const char *name);
double *copyMatrixToContiguosArray(const double *const m, const int initial, const int tileSize, const int size);
double rtclock();

int main(int argc, const char* argv[])
{
  if (argc >= 2) {
    int m_size = atoi(argv[1]);
    int debug = 0, retVal = 0;
    double startTime, endTime, time;

    //Get needed info for the multiplication
    if (argc == 3 && strncmp(argv[2], "-d", 2) == 0) debug = 1;
    double * a = generateRandomMatrixOfSize(m_size);
    double * b = generateRandomMatrixOfSize(m_size);
    double * c = matrixOfZerosWithSize(m_size);

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
  const int cleanup = size % MU != 0;
  int outer_j, outer_i, outer_k, j, i, k;
  for(outer_j = 0; outer_j < size; outer_j+=NB) {
    const double *const copy_B = copyMatrixToContiguosArray(sourceB, outer_j, NB, size);
    for(outer_i = 0; outer_i < size; outer_i+=NB) {
      const double *const copy_C = copyMatrixToContiguosArray(destination, outer_i, NB, size);
      for(outer_k = 0; outer_k < size; outer_k+=NB) {
        const int max_j = fmin(outer_j+NB, size);
        for (j = outer_j; j < max_j; j+=NU) {
          const int max_i = fmin(outer_i+NB, size-MU+1);
          for (i = outer_i; i < max_i; i+=MU) {
            register double C1 = copy_C[((j-outer_j)*size)+i];
            register double C2 = copy_C[((j-outer_j)*size)+i+1];
            register double C3 = copy_C[((j-outer_j)*size)+i+2];
            register double C4 = copy_C[((j-outer_j)*size)+i+3];

            const int max_k = fmin(outer_k+NB, size);
            for (k = outer_k; k < max_k; ++k) {
              register double const A1 = sourceA[((i)*size)+k];
              register double const A2 = sourceA[((i+1)*size)+k];
              register double const A3 = sourceA[((i+2)*size)+k];
              register double const A4 = sourceA[((i+3)*size)+k];

              register double const B = copy_B[((j-outer_j)*size)+k];

              C1 += A1*B; C2 += A2*B; C3 += A3*B; C4 += A4*B;
            }
            destination[((i)*size)+j] = C1;
            destination[((i+1)*size)+j] = C2;
            destination[((i+2)*size)+j] = C3;
            destination[((i+3)*size)+j] = C4;
          }
          if (cleanup) {
            for (;i < size; ++i) {
              register double C = destination[((i)*size)+j];
              const int max_k = fmin(outer_k+NB, size);
              for (k = outer_k; k < max_k; ++k) {
                register double const A = sourceA[((i)*size)+k];
                register double const B = copy_B[((j-outer_j)*size)+k];
                C += A*B;
              }
              destination[((i)*size)+j] = C;
            }
          }
        }
      }
    }
  }
  return 0;
}

double *copyMatrixToContiguosArray(const double *const m, const int initial, const int tileSize, const int size) {
  int i, j;
  double *res __attribute__((aligned(16))) = (double*) malloc(size*tileSize*sizeof(double));
  for(i = 0; i < tileSize; i++){
    for(j = initial; j < fmin(initial+tileSize, size); j++) {
      res[(j-initial)+(i*size)] = m[(i)+(j*size)];
    }
  }
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
  for (i = 0; i < size; ++i) {
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
