#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>

static int NB = 0;
static const int MU = 6;
static const int NU = 5;
static const int MAX_MATRIX_NUMBER = 30;

double *generateRandomMatrixOfSize(int size);
double *matrixOfZerosWithSize(int size);
int matrixMultiply(const double *const sourceA, 
                   const double *const sourceB, 
                   double *const destination, 
                   const int size);
void printMatrix(double *m, int size, char *name);
double rtclock();

int main(int argc, const char* argv[])
{
  if (argc >= 2) {
    int m_size = atoi(argv[1]);
    int debug = 0, retVal = 0;
    double *a, *b, *c, startTime, endTime, time;

    //Get needed info for the multiplication
    if (argc == 3 && strncmp(argv[2], "-d", 2) == 0) debug = 1;
    a = generateRandomMatrixOfSize(m_size);
    b = generateRandomMatrixOfSize(m_size);
    c = matrixOfZerosWithSize(m_size);

    //Register blocking NB = N
    NB = m_size;

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
  //mini-kernel
  for (int j = 0; j < NB; j+= NU) {
    for (int i = 0; i < NB; i+= MU) {
      for (int ci = i; ci < fmin(i+MU, NB); ++ci) {
        for (int cj = j; cj < fmin(j+NU, NB); ++cj) {
          //load C[ci][cj] into register
          register double C = destination[(ci*size)+cj];
          //printf("load C(i:%d)(j:%d): %.02f\n", ci, cj, cr);
          for (int k = 0; k < NB; k++) {
            //micro-kernel
            //load A[ci][k] into register
            //__m128d A = sourceA[(ci*size)+k];
            //__m128d B = sourceB[(k*size)+cj];
            register double const A = sourceA[(ci*size)+k];
            //load B[k][cj] into register
            register double const B = sourceB[(k*size)+cj];
            //multiply A and B and add to C
            //printf("Mutipy C(i:%d)(j:%d): %.02f x %.02f + %.02f\n", ci, cj, ar, br, cr);
            //__mulsd
            C += A*B;
          }
          //printf("Store C(i:%d)(j:%d): %.02f\n", ci, cj, cr);
          //_mm_store_ps(destination[(ci*size)+cj], C); 
          destination[(ci*size)+cj] = C;
        }
      }
    }
  }
  //Cleanup code for when NB is not multiple of NU and MU
  return 0;
}

double *generateRandomMatrixOfSize(int size)
{
  double *m __attribute__ ((aligned (16))) = (double*) malloc(size*size*sizeof(double));
  srand((unsigned int)rtclock(NULL));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      m[(i*size)+j] = rand() % (MAX_MATRIX_NUMBER + 1);
    }
  }
  return m;
}

double *matrixOfZerosWithSize(int size)
{
  double *m __attribute__ ((aligned (16))) = (double*) malloc(size*size*sizeof(double));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      m[(i*size)+j] = 0.0;
    }
  }
  return m;
}

void printMatrix(double *m, int size, char *name)
{
  printf("%s\n", name);
  for (int i = 0; i < size; ++i) {
    printf("| ");
    for (int j = 0; j < size; ++j) {
      printf("%.2f ", m[(i*size)+j]);
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
