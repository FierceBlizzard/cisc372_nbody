#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include "vector.h"
#include "config.h"

#define BLOCK_SIZE 256

vector3* nums;
vector3** accels;

__global__ void PCompute(vector3* nums, vector3** accels, vector3* d_vel, vector3* d_pos, double* d_mass) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t / NUMENTITIES;
    int j = t % NUMENTITIES;

    accels[t] = &nums[t*NUMENTITIES];
    if (t < NUMENTITIES * NUMENTITIES) {
        if(i == j){
            FILL_VECTOR(accels[i][j], 0, 0, 0);
        }else{
            vector3 dist;

            //finding the distance of all 3 demimesnioal spaces
            dist[0] = d_pos[i][0] - d_pos[j][0];
            dist[1] = d_pos[i][1] - d_pos[j][1];
            dist[2] = d_pos[i][2] - d_pos[j][2];

            // calculating magnitude and acceleration 
            double mag_sq = dist[0] * dist[0] + dist[1] *dist[1] + dist[2] * dist[2];
            double mag = sqrt(mag_sq);
            double accelmag = -1 * GRAV_CONSTANT * d_mass[j]/mag_sq;
            FILL_VECTOR(accels[i][j], accelmag*dist[0]/mag, accelmag*dist[1]/mag, accelmag*dist[2]/mag);
        }
        
        vector3 accel_sum = {(double) *(accels[t])[0], (double) *(accels[t])[1], (double) *(accels[t])[2]};
        
        d_vel[i][0]+=accel_sum[0]*INTERVAL;
		d_pos[i][0]=d_vel[i][0]*INTERVAL;

		d_vel[i][1]+=accel_sum[1]*INTERVAL;
		d_pos[i][1]=d_vel[i][1]*INTERVAL;

		d_vel[i][2]+=accel_sum[2]*INTERVAL;
		d_pos[i][2]=d_vel[i][2]*INTERVAL;
    }
}

void compute() {
    vector3 *d_vel, *d_pos;
    double *d_mass;

    vector3 *d_vel, *d_pos;
    double *d_mass;

    cudaMallocManaged((void**) &d_vel, (sizeof(vector3) * NUMENTITIES));
    cudaMallocManaged((void**) &d_pos, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged((void**) &d_mass, (sizeof(double) * NUMENTITIES));

    cudaMemcpy(d_vel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    cudaMallocManaged((void**) &vals, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
    cudaMallocManaged((void**) &accels, sizeof(vector3*)*NUMENTITIES);

    int blockSize = 256; 
    int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

    parallelCompute<<<numBlocks, blockSize>>>(vals, accels, d_vel, d_pos, d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
    cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
    cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDefault);

    cudaFree(accels);
    cudaFree(vals);
}
