
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>
#include "struct.h"

/*--------------------------------inra-group legality check------------------------------------*/
//Function:Euler rotations
//Arguments:
//  [out]1.FRAGMENT** pdst_frag:fragments data after euler rotations
//  [in ]2.FRAGMENT* psrc_frag：original fragments data
//  [in ]3.int frag_cnt:the number of fragments 
//  [in ]4.int euler_step_cnt:the steps count of euler rotation through all orientations candidates
//  [in ]5.float step_rad:the rad of every step
//Return value:
__global__ void GPUKernelEuler_XYZ(FRAGMENT** pdst_frag, FRAGMENT* psrc_frag,int frag_cnt, int euler_step_cnt, float step_rad)
{
    unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < euler_step_cnt * euler_step_cnt * euler_step_cnt)
    {
        int x_angle_idx = idx % euler_step_cnt;
        int y_angle_idx = idx / euler_step_cnt % euler_step_cnt;
        int z_angle_idx = idx / euler_step_cnt / euler_step_cnt;
        int baseZ = z_angle_idx * euler_step_cnt*euler_step_cnt;
        int baseY = z_angle_idx * euler_step_cnt;

        //对整个配体的三维欧拉旋转
        float tmpx[FRAG_CNT][9], tmpy[FRAG_CNT][9], tmpz[FRAG_CNT][9];
                //绕z轴旋转
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
                tmpx[i][j] = cos(step_rad * z_angle_idx) * psrc_frag[i].atom[j].X - sin(step_rad * z_angle_idx) * psrc_frag[i].atom[j].Y;
                tmpy[i][j] = sin(step_rad * z_angle_idx) * psrc_frag[i].atom[j].X + cos(step_rad * z_angle_idx) * psrc_frag[i].atom[j].Y;
                tmpz[i][j] = psrc_frag[i].atom[j].Z;
            }
        }
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
                pdst_frag[i][idx].atom[j].X = tmpx[i][j];
                pdst_frag[i][idx].atom[j].Y = tmpy[i][j];
                pdst_frag[i][idx].atom[j].Z = tmpz[i][j];
            }
        }
        //绕y轴旋转  
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
                tmpx[i][j] =  cos(step_rad * y_angle_idx) * pdst_frag[i][idx].atom[j].X + sin(step_rad * y_angle_idx) * pdst_frag[i][idx].atom[j].Z;
                tmpz[i][j] = -sin(step_rad * y_angle_idx) * pdst_frag[i][idx].atom[j].X + cos(step_rad * y_angle_idx) * pdst_frag[i][idx].atom[j].Z;
                tmpy[i][j] = pdst_frag[i][idx].atom[j].Y;
            }
        }
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
                pdst_frag[i][idx].atom[j].X = tmpx[i][j];
                pdst_frag[i][idx].atom[j].Y = tmpy[i][j];
                pdst_frag[i][idx].atom[j].Z = tmpz[i][j];

            }
        }
                //绕x轴旋euler  
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
               tmpy[i][j] = cos(step_rad * x_angle_idx) * pdst_frag[i][idx].atom[j].Y - sin(step_rad * x_angle_idx) * pdst_frag[i][idx].atom[j].Z;
               tmpz[i][j] = sin(step_rad * x_angle_idx) * pdst_frag[i][idx].atom[j].Y + cos(step_rad * x_angle_idx) * pdst_frag[i][idx].atom[j].Z;
               tmpx[i][j] = pdst_frag[i][baseZ + baseY].atom[j].X;
               pdst_frag[i][idx].atom[j].van_table_idx = psrc_frag[i].atom[j].van_table_idx;
            }
            pdst_frag[i][idx].index = idx;
            pdst_frag[i][idx].atom_cnt = psrc_frag[i].atom_cnt;
            pdst_frag[i][idx].pose_cnt = euler_step_cnt*euler_step_cnt*euler_step_cnt;
        }
        for (int i = 0; i < frag_cnt; i++)
        {
            for (int j = 0; j < psrc_frag[i].atom_cnt; j++)
            {
                pdst_frag[i][idx].atom[j].X = tmpx[i][j];
                pdst_frag[i][idx].atom[j].Y = tmpy[i][j];
                pdst_frag[i][idx].atom[j].Z = tmpz[i][j];
            }
        }
    }
}

//Function:Get conformational rotation matrix
//Arguments:
//  [in ]1.FRAGMENT** pfrag: rigid fragments data
//  [out]2.float** R:Rotational matrix
//  [in ]3.unsigned int*frag_poses_cnt:the maximum poses count for every fragment
//  [in ]4.int ref_frag_idx: the index of reference fragment to the rotational fragments
//  [in ]5.int rot_frag_idx: the index of rotational fragment
//  [in ]6.int ref_atom_idx: the reference-side atom on the bond linking reference and rotational fragment
//  [in ]7.int rot_atom_idx: the rotation-side atom on the bond linking reference and rotational fragment
//  [in ]8.int euler_step_cnt:the count of euler rotational candidates 
//  [in ]9.int conf_step_cnt:the count of conformational rotation candidates 
//  [in ]10.float step_rad:the rad of every step 
//Return value:
__global__ void GPUKernelBondAxisRMatrix(FRAGMENT** pfrag, float** R, unsigned int*frag_poses_cnt, int ref_frag_idx, int rot_frag_idx, int ref_atom_idx, int rot_atom_idx, int euler_step_cnt, int conf_step_cnt, float step_rad)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int combinations = frag_poses_cnt[rot_frag_idx];
    if (idx < combinations)
    {
        int ridx = idx % conf_step_cnt;
        int pose_idx = idx / conf_step_cnt;

        //键轴坐标系一维旋转
        float vx = pfrag[rot_frag_idx][pose_idx].atom[rot_atom_idx].X - pfrag[ref_frag_idx][pose_idx].atom[ref_atom_idx].X;
        float vy = pfrag[rot_frag_idx][pose_idx].atom[rot_atom_idx].Y - pfrag[ref_frag_idx][pose_idx].atom[ref_atom_idx].Y;
        float vz = pfrag[rot_frag_idx][pose_idx].atom[rot_atom_idx].Z - pfrag[ref_frag_idx][pose_idx].atom[ref_atom_idx].Z;
        float len = sqrt(vx * vx + vy * vy + vz * vz);

        float c = cosf(ridx * step_rad);
        float s = sinf(ridx * step_rad);
        float t = 1 - c;

        if (len < 1e-6f) // 键轴与Z轴几乎平行
        {
            // 此时旋转矩阵就是绕Z轴旋转step_rad
            R[0][idx] = c;    R[1][idx] = 0.0f; R[2][idx] = 0.0f;
            R[3][idx] = 0.0f; R[4][idx] = c;    R[5][idx] = 0.0f;
            R[6][idx] = 0.0f; R[7][idx] = 0.0f; R[8][idx] = c;
        }
        else  // 正常情况，继续归一化
        {
            vx = vx / len;
            vy = vy / len;
            vz = vz / len;

            R[0][idx] = t * vx * vx + c;
            R[1][idx] = t * vx * vy - vz * s;
            R[2][idx] = t * vx * vz + vy * s;
            R[3][idx] = t * vx * vy + vz * s;
            R[4][idx] = t * vy * vy + c;
            R[5][idx] = t * vy * vz - vx * s;
            R[6][idx] = t * vx * vz - vy * s;
            R[7][idx] = t * vy * vz + vx * s;
            R[8][idx] = t * vz * vz + c;
        }
    }
}

//Function:Conformational rotations
//Arguments:
//  [out]1.FRAGMENT** pfrag:fragments data after conformational rotations
//  [in ]2.float** R：conformation-rotation matrix
//  [in ]3.unsigned int*frag_poses_cnt:the maximum poses count for every fragment
//  [in ]4.int* rot_frag_idx:the rotational fragments indexes
//  [in ]5.int start_rot_frag_idx:the index of the most upstream rotational fragment
//  [in ]6.int frag_cnt:the number of fragments 
//  [in ]7.int euler_step_cnt:the count of euler rotational candidates 
//  [in ]8.int conf_step_cnt:the count of conformational rotation candidates 
//  [in ]9.float step_rad:the rad of every step 
//Return value:
__global__ void GPUKernelConfRotate(FRAGMENT** pfrag, float** R, unsigned int* frag_poses_cnt, int* rot_frag_idx, int start_rot_frag_idx, int fragment_cnt, int euler_step_cnt, int conf_step_cnt, float step_rad)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int combinations = frag_poses_cnt[start_rot_frag_idx];
    float x[11][9], y[11][9], z[11][9];
    if (idx < combinations)
    {
        unsigned int ref_idx = idx / conf_step_cnt;
        //键轴坐标系一维旋转
        for (int k = 0; k < fragment_cnt; k++)
        {
            for (int i = 0; i < pfrag[rot_frag_idx[k]][0].atom_cnt; i++)
            {
                x[k][i] = R[0][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].X + R[3][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Y + R[6][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Z;
                y[k][i] = R[1][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].X + R[4][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Y + R[7][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Z;
                z[k][i] = R[2][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].X + R[5][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Y + R[8][idx] * pfrag[rot_frag_idx[k]][ref_idx].atom[i].Z;
            }
        }
        for (int k = 0; k < fragment_cnt; k++)
        {
            for (int i = 0; i < pfrag[rot_frag_idx[k]][0].atom_cnt; i++)
            {
                pfrag[rot_frag_idx[k]][idx].atom[i].X = x[k][i];
                pfrag[rot_frag_idx[k]][idx].atom[i].Y = y[k][i];
                pfrag[rot_frag_idx[k]][idx].atom[i].Z = z[k][i];
            }
        }
    }
}

//Function:Initialize All fragments for every pose after rotations
//Arguments:
//  [out]1.FRAGMENT** pdstfrag:data of the destination fragments
//  [in ]2.FRAGMENT* psrcfrag:the original data of fragments
//  [in ]3.unsigned int* frag_poses_cnt:the maximum poses count for every fragment
//Return value:
__global__ void GPUKernelInitFragsAfterRotate(FRAGMENT** pdstfrag,FRAGMENT* psrcfrag, unsigned int* frag_poses_cnt, int frag_idx)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    if (idx < frag_poses_cnt[frag_idx])
    {
        for (i = 0; i < psrcfrag[frag_idx].atom_cnt; i++)
        {
            pdstfrag[frag_idx][idx].atom[i].van_table_idx = psrcfrag[frag_idx].atom[i].van_table_idx;
        }

        pdstfrag[frag_idx][idx].index = idx;
        pdstfrag[frag_idx][idx].atom_cnt = i;
        pdstfrag[frag_idx][idx].pose_cnt = frag_poses_cnt[frag_idx];
    }
}

//Function:Rotates all the rigid fragments
//Arguments:
//  [in ]1.FRAGMENT* psrc_frag:the data of original fragments
//  [out]2.FRAGMENT** pdst_frag:the data of the fragments after rotations
//  [in ]3.unsigned int* frag_poses_cnt:the maximum poses count for every fragment
//  [in ]4.float pocket_cx:the X coordinates of pocket center
//  [in ]5.float pocket_cy:the Y coordinates of pocket center
//  [in ]6.float pocket_cz:the Z coordinates of pocket center
//Return value:
void Rotate(FRAGMENT* psrc_frag, FRAGMENT** pdst_frag, unsigned int* frag_poses_cnt, float pocket_cx, float pocket_cy, float pocket_cz)
{
    float* R_host[9];
    float** R_dev;
    cudaMalloc(&R_dev,  sizeof(float*)*9);

    GPUKernelEuler_XYZ<<<(EULER_STEP_CNT* EULER_STEP_CNT* EULER_STEP_CNT+511)>>9, 512 >>>(pdst_frag, psrc_frag, FRAG_CNT, EULER_STEP_CNT, EULER_STEP_RAD);
    cudaDeviceSynchronize();

    for(int j=0; j<9; j++)
        cudaMalloc(&R_host[j], sizeof(float)*frag_poses_cnt[FRAG_CNT-1]);
    cudaMemcpy(R_dev, R_host, sizeof(float*) * 9, cudaMemcpyHostToDevice);
    
    unsigned int* frag_poses_cnt_dev;
    int * rot_frag_idx_dev;


    //从片段1---下游所有片段枚举全部构向 1-9
    int rot_frag_idx[FRAG_CNT-1] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    cudaMalloc(&frag_poses_cnt_dev, sizeof(unsigned int)* FRAG_CNT);
    cudaMemcpy(frag_poses_cnt_dev, frag_poses_cnt, sizeof(unsigned int)* FRAG_CNT, cudaMemcpyHostToDevice);
   
    cudaMalloc(&rot_frag_idx_dev, sizeof(int)* (FRAG_CNT-1));
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int)* (FRAG_CNT - 1), cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[1] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 0, 1, 3, 4, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[1] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 1, 9, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    
    //从片段2---下游所有片段枚举全部构向 2-9
    rot_frag_idx[0] = 2;rot_frag_idx[1] = 3; rot_frag_idx[2] = 4;rot_frag_idx[3] = 5;rot_frag_idx[4] = 6;rot_frag_idx[5] = 7;rot_frag_idx[6] = 8;rot_frag_idx[7] = 9;
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int) * 8, cudaMemcpyHostToDevice);
   
    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[2] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 1, 2, 11, 12, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[2] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 2, 8, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
   
    //从片段3--下游所有片段枚举全部构向 3-9
    rot_frag_idx[0] = 3; rot_frag_idx[1] = 4; rot_frag_idx[2] = 5; rot_frag_idx[3] = 6; rot_frag_idx[4] = 7; rot_frag_idx[5] = 8;rot_frag_idx[6] = 9;
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int) * 7, cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[3] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 2, 3, 17, 18, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[3] + 511) >> 9, 512 >> >(pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 3, 7,  EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();

    //从片段4---下游所有片段枚举全部构向 4-9
    rot_frag_idx[0] = 4; rot_frag_idx[1] = 5; rot_frag_idx[2] = 6; rot_frag_idx[3] = 7; rot_frag_idx[4] = 8; rot_frag_idx[5] = 9;
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int) * 6, cudaMemcpyHostToDevice);
    
    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[4] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 3, 4, 20, 21, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[4] + 511) >> 9, 512 >> >(pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 4, 6,  EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
      
    //从片段5---下游所有片段枚举全部构向 5-9
    rot_frag_idx[0] = 5; rot_frag_idx[1] = 6; rot_frag_idx[2] = 7; rot_frag_idx[3] = 8; rot_frag_idx[4] = 9; 
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int) * 5, cudaMemcpyHostToDevice);
    
    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[5] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 4, 5, 25, 26, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[5] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 5, 5,  EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();

    //片段6---下游所有片段枚举全部构向 6-9
    rot_frag_idx[0] = 6; rot_frag_idx[1] = 7; rot_frag_idx[2] = 8; rot_frag_idx[3] = 9; 
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int)*4 , cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[6] + 511) >> 9, 512 >> > (pdst_frag, R_dev,frag_poses_cnt_dev, 5, 6, 32, 33, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[6] + 511) >> 9, 512 >> >(pdst_frag,  R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 6, 4,  EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
   
    //片段7---下游所有片段枚举全部构向 7-9
    rot_frag_idx[0] = 7; rot_frag_idx[1] = 8; rot_frag_idx[2] = 9;
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int)*3 , cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[7] + 511) >> 9, 512 >> > (pdst_frag, R_dev,frag_poses_cnt_dev, 6, 7, 41, 42, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[7] + 511) >> 9, 512 >> >(pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev, 7, 3, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    
    //片段8---下游所有片段枚举全部构向 8-9
    rot_frag_idx[0] = 8; rot_frag_idx[1] = 9;
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int)*2, cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[8] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, 7, 8, 50, 51, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[8] + 511) >> 9, 512 >> > (pdst_frag, R_dev,frag_poses_cnt_dev, rot_frag_idx_dev, 8, 2, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    
    //片段9---下游所有片段枚举全部构向 9
    rot_frag_idx[0] = 9; 
    cudaMemcpy(rot_frag_idx_dev, rot_frag_idx, sizeof(int), cudaMemcpyHostToDevice);

    GPUKernelBondAxisRMatrix << < (frag_poses_cnt[9] + 511) >> 9, 512 >> > (pdst_frag, R_dev,frag_poses_cnt_dev, 8, 9, 56, 57, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();
    GPUKernelConfRotate << < (frag_poses_cnt[9] + 511) >> 9, 512 >> > (pdst_frag, R_dev, frag_poses_cnt_dev, rot_frag_idx_dev,9, 1, EULER_STEP_CNT, CONF_STEP_CNT, CONF_STEP_RAD);
    cudaDeviceSynchronize();

   
    for (int i = 0; i < FRAG_CNT; i++)
    {
        GPUKernelInitFragsAfterRotate << < (frag_poses_cnt[i] + 511) >> 9, 512 >> > (pdst_frag, psrc_frag, frag_poses_cnt_dev, i);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 9; i++)
        cudaFree(R_host[i]);
    cudaFree(R_dev);

}
/*---------------------------------------------------------------------------------------------*/
