#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

static const int NB = 64;
static const int MU = 4;
static const int NU = 2;

static const int MAX_MATRIX_NUMBER = 30;

double *generateRandomMatrixOfSize(int size);
double *matrixOfZerosWithSize(int size);
int matrixMultiply(const double *const __restrict__ sourceA, 
    const double *const __restrict__ sourceB, 
    double *const __restrict__ destination, 
    const int size);
void simpleMatrixMultiply(const double *const sourceA, 
    const double *const sourceB, 
    double *const destination, 
    const int size);
void printMatrix(const double *const m, const int size, const char *name);
void saveMatrixToFile(const double *const m, const int size, FILE *f);
double *copyA(const double *const m, const int size);
double *copyB(const double *const m, const int size, const int start_j);
double *copyC(const double *const m, const int size, const int start_i, const int start_j);
double rtclock();

int main(int argc, const char* argv[])
{
  srand((unsigned int)time(NULL));
  
  if (argc >= 2) {
    int m_size = atoi(argv[1]);
    int debug = 0, retVal = 0, test = 0;
    double startTime = 0, endTime = 0, time;

    //Get needed info for the multiplication
    if (argc == 3) {
      if (strncmp(argv[2], "-d", 2) == 0) debug = 1;
      else if (strncmp(argv[2], "-t", 2) == 0) test = 1;
    } 
    else if (argc == 4) {
      if (strncmp(argv[2], "-d", 2) == 0 || strncmp(argv[3], "-d", 2) == 0) debug = 1;
      if (strncmp(argv[2], "-t", 2) == 0 || strncmp(argv[3], "-t", 2) == 0) test = 1;
    }

    double * __attribute__((aligned(16))) a = generateRandomMatrixOfSize(m_size);
    double * __attribute__((aligned(16))) b = generateRandomMatrixOfSize(m_size);
    double * __attribute__((aligned(16))) c = matrixOfZerosWithSize(m_size);

    //Multiply
    if (test) {
      double * __attribute__((aligned(16))) test_c = matrixOfZerosWithSize(m_size);      
      simpleMatrixMultiply(a, b, test_c, m_size);
      matrixMultiply(a, b, c, m_size);
      
      FILE *correct = fopen("correct.txt", "w");
      saveMatrixToFile(test_c, m_size, correct); 
      fclose(correct);
      FILE *mine = fopen("mine.txt", "w");
      saveMatrixToFile(c, m_size, mine); 
      fclose(mine);
    } else {
      startTime = rtclock();
      retVal = matrixMultiply(a, b, c, m_size);
      endTime = rtclock();
    }

    //Print Flops
    time = endTime - startTime;
    if (!test) {
      printf("Size: %d\n", m_size);
      printf("Time: %0.02f\n", time);
      printf("GFLOPS: %.02f\n", ((2.0*pow(m_size, 3))/time)/1000000000.0);
    }

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
      "Usage: ./mmm <matrix size> [flags]\n\n"
      "Flags\n"
      "-d:\t Print debugging info.\n"
      "-t:\t Run Fast Matrix Multiply (save resulting matrix in mine.txt) and the simple ijk algorithm (save resulting matrix in correct.txt). Useful for proving correctness of Fast Matrix Multiply.\n");
  return 1;
}

int matrixMultiply(const double *const __restrict__ sourceA, 
    const double *const __restrict__ sourceB, 
    double *const __restrict__ destination, 
    const int size)
{
  #ifdef __INTEL_COMPILER
  __assume_aligned(sourceA, 16); 
  __assume_aligned(sourceB, 16); 
  __assume_aligned(destination, 16);
  #endif

  int outer_j, outer_i, outer_k, j, i, k, cleanup_i, cleanup_j;
  const int n_a = ceil((double)size/(double)NB);
  const int inner_n_a = floor((double)NB/(double)MU); 
  const double *const copy_A __attribute__((aligned(16))) = copyA(sourceA, size);
  for(outer_j = 0; outer_j < size; outer_j+=NB)
  {
    cleanup_j = fmod(fmin(size - outer_j, NB), NU) != 0;
    const double *const copy_B __attribute__((aligned(16))) = copyB(sourceB, size, outer_j);
    for(outer_i = 0; outer_i < size; outer_i+=NB)
    {
      cleanup_i = fmod(fmin(size-outer_i, NB), MU) != 0;
      double *const copy_C __attribute__((aligned(16))) = copyC(destination, size, outer_i, outer_j);
      for(outer_k = 0; outer_k < size; outer_k+=NB)
      {
        const int max_j = fmin(outer_j+NB-NU+1, size-NU+1);
        for (j = outer_j; j < max_j; j+=NU)
        {
          const int max_i = fmin(outer_i+NB-MU+1, size-MU+1);
          for (i = outer_i; i < max_i; i+=MU)
          {
            register double C1 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))];
            register double C2 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+1];
            register double C3 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+2];
            register double C4 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+3];
            register double C5 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+4];
            register double C6 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+5];
            register double C7 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+6];
            register double C8 = copy_C[((i-outer_i)*NU + ((j-outer_j)*NB))+7];
           
            const int max_k = fmin(outer_k+NB, size);
            for (k = outer_k; k < max_k; ++k)
            {
              register const double A1 = copy_A[((k-outer_k)*MU) + (MU*outer_k*(NB/MU)) + ((i-outer_i)*NB) + ((outer_i/NB)*NB*MU*inner_n_a*n_a) + 0];
              register const double A2 = copy_A[((k-outer_k)*MU) + (MU*outer_k*(NB/MU)) + ((i-outer_i)*NB) + ((outer_i/NB)*NB*MU*inner_n_a*n_a) + 1];
              register const double A3 = copy_A[((k-outer_k)*MU) + (MU*outer_k*(NB/MU)) + ((i-outer_i)*NB) + ((outer_i/NB)*NB*MU*inner_n_a*n_a) + 2];
              register const double A4 = copy_A[((k-outer_k)*MU) + (MU*outer_k*(NB/MU)) + ((i-outer_i)*NB) + ((outer_i/NB)*NB*MU*inner_n_a*n_a) + 3];

              register const double B1 = copy_B[(NU*(k-outer_k))+(NU*outer_k*(NB/NU))+((j-outer_j)*NB)];
              register const double B2 = copy_B[(NU*(k-outer_k))+(NU*outer_k*(NB/NU))+((j-outer_j)*NB)+1];

              C1 += A1*B1;
              C2 += A2*B1;

              C3 += A3*B1;
              C4 += A4*B1;

              C5 += A1*B2;
              C6 += A2*B2;
              
              C7 += A3*B2;
              C8 += A4*B2;
            }
            destination[((i)*size)+(j)] = C1;
            destination[((i)*size)+(j+1)] = C2;
            destination[((i+1)*size)+(j)] = C3;
            destination[((i+1)*size)+(j+1)] = C4;
            destination[((i+2)*size)+(j)] = C5;
            destination[((i+2)*size)+(j+1)] = C6;
            destination[((i+3)*size)+(j)] = C7;
            destination[((i+3)*size)+(j+1)] = C8;
          }
          if (cleanup_i)
          {
            const int iterations = fmin(size-i, (outer_i+NB)-i);
            int ci, cj;
            for(cj = j; cj < j+NU; ++cj)
            {
              for (ci = i; ci < i+iterations; ++ci)
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
      free(copy_C);
    }
    if (cleanup_j)
    {
      int ci, cj = j;
      for (ci = 0; ci < size; ++ci)
      {
        register double C = destination[((ci)*size)+cj];
        for (k = 0; k < size; ++k)
        {
          register double const A = sourceA[((ci)*size)+k];
          register double const B = sourceB[(k*size)+cj];
          C += A*B;
        }
        destination[((ci)*size)+cj] = C;
      }
    }
    free((void *)copy_B);
  }
  free((void *)copy_A);
  return 0;
}

double *copyA(const double *const m, const int size)
{
  #ifdef __INTEL_COMPILER
  __assume_aligned(m, 16);
  #endif
  int i, j, k, l, h;
  const int n = ceil((double)size/(double)NB);
  const int inner_n = floor((double)NB/(double)MU);
  double *res __attribute__((aligned(16))) = (double*)malloc(inner_n*n*n*MU*NB*sizeof(double));
  for(h=0; h < n; ++h)
    for(k=0; k < n; ++k)
      for(l = 0; l < inner_n; ++l)
        for(i = 0; i < NB; ++i)
          for(j = 0; j < MU; ++j)
            res[((MU * i) + j) + (l * NB * MU) + (k * MU * NB * inner_n) + (h * n * inner_n * NB * MU)] = m[(((h * NB) + j + (l*MU))*size) + i + (k * (MU * inner_n))];

  return res;
}

double *copyB(const double *const m, const int size, const int start_j)
{
  #ifdef __INTEL_COMPILER
  __assume_aligned(m, 16);
  #endif 
  int i, j, k, l;
  const int n = ceil((double)size/(double)NB);
  const int inner_n = floor((double)NB/(double)NU);
  double *res __attribute__((aligned(16))) = (double*)malloc(inner_n*n*NU*NB*sizeof(double));
  for(k=0; k < n; ++k)
    for(l = 0; l < inner_n; ++l)
      for(i = 0; i < NB; ++i)
        for(j = 0; j < NU; ++j)
          res[((NU * i) + j) + (l * NB * NU) + (k * NU * NB * inner_n)] = m[(((i) * size) + start_j + j+(l*NU)) + (k * NB * size)];

  return res;
}

double *copyC(const double *const m, const int size, const int start_i, const int start_j)
{
  #ifdef __INTEL_COMPILER
  __assume_aligned(m, 16);
  #endif
  int i, j, k, l;
  const int n_i = floor((double)NB/(double)MU);
  const int n_j = floor((double)NB/(double)NU);
  double *res __attribute__((aligned(16))) = (double*)malloc(n_i*n_j*MU*NU*sizeof(double));
  for(k=0; k < n_j; ++k)
    for(l=0; l < n_i; ++l)
      for(i = 0; i < MU; ++i)
        for(j = 0; j < NU; ++j)
          res[(i*NU)+j + l*(MU*NU) + k*(NU*NB)] = m[((start_i + i) * size)+(start_j + j) + ((l*MU)*size) + (k*NU)];
  
  return res;
}

/**
 * Helper Methods
 */

double *generateRandomMatrixOfSize(int size)
{
  int i, j;
  double *m __attribute__((aligned(16))) = (double*) malloc(size*size*sizeof(double));
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
    for (j = 0; j < size; ++j) printf("%.0f ", m[(i*size)+j]);
    printf("\n");
  }
  printf("\n");
}

void saveMatrixToFile(const double *const m, const int size, FILE *f)
{
  int i, j;
  for (i = 0; i < size; ++i) 
  {
    for (j = 0; j < size; ++j) fprintf(f, "%.0f ", m[(i*size)+j]);
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
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

void simpleMatrixMultiply(const double *const sourceA, 
    const double *const sourceB, 
    double *const destination, 
    const int size)
{
  int i, j, k;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        destination[(i*size)+j] += sourceA[(i*size)+k] * sourceB[(k*size)+j];
      }
    }
  }
}
