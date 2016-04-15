#include <stdio.h>
#include <stdlib.h>

double **generateRandomMatrixOfSize(int size);
int matrixMultiply(double **sourceA, double **sourceB, double **destination);

int main(int argc, const char* argv[])
{
  if (argc == 2) {
    int matrix_size = atoi( argv[1] );
    double **a = generateRandomMatrixOfSize(matrix_size);
    double **b = generateRandomMatrixOfSize(matrix_size);
    double **c = generateRandomMatrixOfSize(matrix_size);
    return matrixMultiply(a, b, c); 
  }
  printf("Patricio Beltran \n"
      "PMB784 \n"
      "Fast Matrix Multiply \n"
      "------------ \n"
      "Usage: ./mmm <matrix size>\n");
  return 1;
}

double **generateRandomMatrixOfSize(int size) {
  return NULL;
}

int matrixMultiply(double **sourceA, double **sourceB, double **destination) {
  return 0;
}
