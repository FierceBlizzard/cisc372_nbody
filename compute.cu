#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "cuda.h"
#include "cuda_runtime.h"

// Compute the gravitational force between two bodies
#define G 6.674e-11

__global__ void computeForces(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vector3 f = {0.0, 0.0, 0.0};
        for (int j = 0; j < n; j++) {
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
        force[i].x = f.x;
        force[i].y = f.y;
        force[i].z = f.z;
    }
}

// Update the position and velocity of a body based on the forces acting on it
__global__ void updateBody(int n, double dt, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vector3 a = {force[i].x / mass[i], force[i].y / mass[i], force[i].z / mass[i]};
        vel[i].x += a.x * dt;
        vel[i].y += a.y * dt;
        vel[i].z += a.z * dt;
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
}

// Compute the gravitational forces between all pairs of bodies
void compute(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeForces<<<numBlocks, blockSize>>>(n, pos, vel, mass, force, G);
    cudaDeviceSynchronize();
    updateBody<<<numBlocks, blockSize>>>(n, DT, pos, vel, mass, force);
    cudaDeviceSynchronize();
}