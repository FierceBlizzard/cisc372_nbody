#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "vector.h"
#include "compute.h"
#include "config.h"

#define BLOCK_SIZE 256
#define G GRAV_CONSTANT

__global__ void computeForces(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j;
        vector3 f = {0.0, 0.0, 0.0};
        for (j = 0; j < n; j++) {
            if (i != j) {
                double dx = pos[j].x - pos[i].x;
                double dy = pos[j].y - pos[i].y;
                double dz = pos[j].z - pos[i].z;
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                double mag = G * mass[i] * mass[j] / (dist * dist * dist);
                f.x += mag * dx;
                f.y += mag * dy;
                f.z += mag * dz;
            }
        }
        force[i] = f;
    }
}

__global__ void computeAcceleration(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force, vector3 *acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i].x = force[i].x / mass[i];
        acc[i].y = force[i].y / mass[i];
        acc[i].z = force[i].z / mass[i];
    }
}

void compute(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *acc) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector3 *d_pos, *d_vel, *d_force, *d_acc;
    double *d_mass;
    cudaMalloc(&d_pos, n * sizeof(vector3));
    cudaMalloc(&d_vel, n * sizeof(vector3));
    cudaMalloc(&d_mass, n * sizeof(double));
    cudaMalloc(&d_force, n * sizeof(vector3));
    cudaMalloc(&d_acc, n * sizeof(vector3));
    cudaMemcpy(d_pos, pos, n * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, n * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, n * sizeof(double), cudaMemcpyHostToDevice);
    computeForces<<<numBlocks, BLOCK_SIZE>>>(n, d_pos, d_vel, d_mass, d_force);
    computeAcceleration<<<numBlocks, BLOCK_SIZE>>>(n, d_pos, d_vel, d_mass, d_force, d_acc);
    cudaMemcpy(acc, d_acc, n * sizeof(vector3), cudaMemcpyDeviceToHost);
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
    cudaFree(d_force);
    cudaFree(d_acc);
}
