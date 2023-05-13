#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "vector.h"
#include "compute.h"
#include "config.h"

#define BLOCK_SIZE 256

__global__ void computeForces(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j;
        vector3 f = {0.0, 0.0, 0.0};
        for (j = 0; j < n; j++) {
            if (i != j) {
                vector3 r = {pos[j].x - pos[i].x, pos[j].y - pos[i].y, pos[j].z - pos[i].z};
                vector3 r_mag = {r.x / (sqrt(r.x * r.x + r.y * r.y + r.z * r.z)),
                                 r.y / (sqrt(r.x * r.x + r.y * r.y + r.z * r.z)),
                                 r.z / (sqrt(r.x * r.x + r.y * r.y + r.z * r.z))};
                vector3 dist = {sqrt(r.x * r.x + r.y * r.y + r.z * r.z),
                                sqrt(r.x * r.x + r.y * r.y + r.z * r.z),
                                sqrt(r.x * r.x + r.y * r.y + r.z * r.z)};
                vector3 mag = {GRAV_CONSTANT * mass[i] * mass[j] / (dist.x * dist.x * dist.x),
                               GRAV_CONSTANT * mass[i] * mass[j] / (dist.y * dist.y * dist.y),
                               GRAV_CONSTANT * mass[i] * mass[j] / (dist.z * dist.z * dist.z)};
                f.x += mag.x * r_mag.x;
                f.y += mag.y * r_mag.y;
                f.z += mag.z * r_mag.z;
            }
        }
        force[i] = f;
    }
}

__global__ void computeAcceleration(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force, vector3 *acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vector3 a = {force[i].x / mass[i], force[i].y / mass[i], force[i].z / mass[i]};
        acc[i] = a;
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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in computeForces: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    computeAcceleration<<<numBlocks, BLOCK_SIZE>>>(n, d_pos, d_vel, d_mass, d_force, d_acc);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in computeAcceleration: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(acc, d_acc, n * sizeof(vector3), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
    cudaFree(d_force);
    cudaFree(d_acc);
}
