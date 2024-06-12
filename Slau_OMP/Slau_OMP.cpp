#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <omp.h>


const long int n = 1200;
long double e = 10e-180;
double A[n][n];
double ipsilon[n];
double b[n];
const double PI = 3.1415926;
double r0[n],x0[n],z0[n],alpha,beta,Az[n];
double temp[n];

double scalMul(double u[n], double v[n]) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += u[i] * v[i];
    }
    return result;
}

double ModuleOfVector(double* vector, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += vector[i] * vector[i];
    }
    return sqrt(res);
}


void MatrixMulVector(double result[], double matrix[n][n], double vector[]) {
    double tmp;
    for (int str = 0; str < n; str++) {
        tmp = 0;
        for (int col = 0; col < n; col++)
            tmp += matrix[str][col]*vector[col];
        result[str] = tmp;
    }
}


void printMatrix(double matrix[n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", matrix[i][j]);
        }
    }
    printf("\n");
}
void printVector(double vector[]) {
    for (int j = 0; j < n; j++) {
        printf("%f  ", vector[j]);

    }
    printf("\n");
}

void clsVec(double vec[n]) {
       for (int i = 0; i < n; i++)vec[i] = 0;
}

void cpyVec(double source[n], double dest[n]) {
     for (int i = 0; i < n; i++) dest[i] = source[i];
}

int main(int argc, char** argv)
{
    long double sum1 , sum2=0, scalR ;
    int iter=0;
   
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j)
                A[i][j] = 2.;
            else A[i][j] = 1;
        }
   
    for (int i = 0; i < n; i++)
        ipsilon[i] = sin(2 * PI * i / n);

   
    MatrixMulVector(b,A,ipsilon);
    
    clsVec(x0);
   
    clsVec(temp);
    MatrixMulVector(temp, A, x0);
    for (int i = 0; i < n; i++) {
        r0[i] = b[i] - temp[i];
    }
    cpyVec(r0, z0);
    auto beginTime = std::chrono::steady_clock::now();
    
    
    sum2 = ModuleOfVector(&b[0],n);
   

    do{
        //Az = A*zn
        clsVec(Az);
        #pragma omp parallel sections
        {
          #pragma omp section
          MatrixMulVector(Az, A, z0);
          //(rn,rn)
          #pragma omp section
            scalR = scalMul(r0, r0);
        }
        //a = (rn,rn)/(Az,zn)
        alpha = scalR / scalMul(Az, z0);

#pragma omp parallel sections
        {
            //xn+1 = xn+a*zn
            #pragma omp section
            for (int i = 0; i < n; i++)
                x0[i] = x0[i] + alpha * z0[i];

            //rn+1=rn-alpha*Az
            #pragma omp section
            for (int i = 0; i < n; i++)
                r0[i] = r0[i] - alpha * Az[i];
        }
#pragma omp parallel sections
        {
            //beta=(rn+1,rn+1)/(rn,rn)
        #pragma omp section
            {
                beta = scalMul(r0, r0) / scalR;
                //zn+1=rn+1+beta*z
                for (int i = 0; i < n; i++)
                    z0[i] = r0[i] + beta * z0[i];
            }
        #pragma omp section
            {
                sum1 = 0;
                //критерий завершения
                sum1 = ModuleOfVector(&r0[0], n);
            }
        }
        if(++iter > 1000) break;
    }while((sum1 / sum2) >= e);

        auto endTime = std::chrono::steady_clock::now();
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime);
        printf("Iteration: %d\t from %lf sec\n", iter, elapsed_ns.count() / 1000000000.);
    
}


