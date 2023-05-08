#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
// nbodyForce kernel function
// Compute the gravitational force between two bodies
__global__
void computeForces(int n, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
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
        force[i] = f;
    }
}

// Update the position and velocity of a body based on the forces acting on it
__global__ 
void updateBody(int n, double dt, vector3 *pos, vector3 *vel, double *mass, vector3 *force) {
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

__global__ 
void computeAcceleration(int n, vector3 *pos, double *mass, double *accel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        if (i != j) {
            double dx = pos[j].x - pos[i].x;
            double dy = pos[j].y - pos[i].y;
            double dz = pos[j].z - pos[i].z;
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            double mag = G * mass[j] / (dist * dist * dist);
            accel[i * n + j] = mag * dx;
            accel[i * n + j + n * n] = mag * dy;
            accel[i * n + j + 2 * n * n] = mag * dz;
        }
    }
}

// Compute the gravitational forces between all bodies in the system and update their positions and velocities
void compute() {
    cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(mass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    // Compute gravitational forces between all bodies
    computeForces<<<numBlocks, threadsPerBlock>>>(NUMENTITIES, d_hPos, d_hVel, mass, force);

    // Compute acceleration of all bodies
    computeAcceleration<<<numBlocks, threadsPerBlock>>>(NUMENTITIES, d_hPos, mass, acceleration);

    // Update positions and velocities of all bodies
    updateBody<<<numBlocks, threadsPerBlock>>>(NUMENTITIES, DT, d_hPos, d_hVel, mass, acceleration);

    cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
}
