#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <emmintrin.h>

static const int NB = 47; // 3*NB^2 <= 20480
static const int MU = 6;
static const int NU = 5;
static const int MAX_MATRIX_NUMBER = 30;

double *generateRandomMatrixOfSize(int size);
double *matrixOfZerosWithSize(int size);
int matrixMultiply(const double *const sourceA, 
                   const double *const sourceB, 
                   double *const destination, 
                   const int size);
void printMatrix(const double *const m, const int size, const char *name);
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

int matrixMultiply(const double *const sourceA, 
                   const double *const sourceB, 
                   double *const destination, 
                   const int size)
{
  int outer_j, outer_i, outer_k, j, i, cj, ci, k;
  for(outer_j = 0; outer_j < size; outer_j+=NB) {
    for(outer_i = 0; outer_i < size; outer_i+=NB) {
      for(outer_k = 0; outer_k < size; outer_k+=NB) {
        //mini-kernel
        const int max_j = fmin(outer_j+NB, size);
        for (j = outer_j; j < max_j; j+=NU) {
          const int max_i = fmin(outer_i+NB, size);
          for (i = outer_i; i < max_i; i+=MU) {
            const int max_cj = fmin(j+NU, max_j);
            for (cj = j; cj < max_cj; ++cj) {
              const int max_ci = fmin(i+MU, max_i);
              for (ci = i; ci < max_ci; ++ci) {
                //load C[ci][cj] into register
                register double C = destination[(ci*size)+cj];
                //printf("load C(i:%d)(j:%d): %.02f\n", ci, cj, C);
                const int max_k = fmin(outer_k+NB, size);
                for (k = outer_k; k < max_k; ++k) {
                  //micro-kernel
                  //load A[ci][k] into register
                  //__m128d A = sourceA[(ci*size)+k];
                  //__m128d B = sourceB[(k*size)+cj];
                  register double const A = sourceA[(ci*size)+k];
                  //load B[k][cj] into register
                  register double const B = sourceB[(k*size)+cj];
                  //multiply A and B and add to C
                  //printf("Mutipy C(i:%d)(j:%d): %.02f x %.02f + %.02f\n", ci, cj, A, B, C);
                  //__mulsd
                  C += A*B;
                }
                //printf("Store C(i:%d)(j:%d): %.02f\n", ci, cj, C);
                //_mm_store_ps(destination[(ci*size)+cj], C); 
                destination[(ci*size)+cj] = C;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}

double *generateRandomMatrixOfSize(int size)
{
  int i, j;
  double *m = (double*) malloc(size*size*sizeof(double));
  srand((unsigned int)time(NULL));
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      //m[(i*size)+j] = rand() % (MAX_MATRIX_NUMBER + 1);
      m[(i*size)+j] = 2;
    }
  }
  return m;
}

double *matrixOfZerosWithSize(int size)
{
  int i, j;
  double *m = (double*) malloc(size*size*sizeof(double));
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      m[(i*size)+j] = 0.0;
    }
  }
  return m;
}

void printMatrix(const double *const m, const int size, const char *name)
{
  int i, j;
  printf("%s\n", name);
  for (i = 0; i < size; ++i) {
    printf("| ");
    for (j = 0; j < size; ++j) {
      printf("%.0f ", m[(i*size)+j]);
    }
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
