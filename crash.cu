#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>
#include "struct.h"
#include "crash.h"

/*----------------------------------------------------------The data structure------------------------------------------------------------------------*/
#if __CONFORMATION_CANDIDATES_NUM__  == 6
           unsigned int ConfStepCnt[11] = { 1, 6, 36, 216, 1296, 7776, 46656, 279936, 1679616, 10077696, 60466176 };  //CONF_STEP_CNT = 6
__device__ unsigned int ConfStepCntDev[11] = { 1, 6, 36, 216, 1296, 7776, 46656, 279936, 1679616, 10077696, 60466176 };  //CONF_STEP_CNT = 6
#elif __CONFORMATION_CANDIDATES_NUM__  == 5
           unsigned int ConfStepCnt   [11] = { 1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625 };  //CONF_STEP_CNT = 5
__device__ unsigned int ConfStepCntDev[11] = { 1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625 };  //CONF_STEP_CNT = 5
#elif __CONFORMATION_CANDIDATES_NUM__ == 4
           unsigned int ConfStepCnt[11]    = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 };  //CONF_STEP_CNT = 4
__device__ unsigned int ConfStepCntDev[11] = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 };  //CONF_STEP_CNT = 4
#elif __CONFORMATION_CANDIDATES_NUM__ == 8
		   unsigned int ConfStepCnt[11] = { 1, 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728, 1073741824 };  //CONF_STEP_CNT = 8
__device__ unsigned int ConfStepCnt[11] = { 1, 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728, 1073741824 };  //CONF_STEP_CNT = 8
#elif __CONFORMATION_CANDIDATES_NUM__ == 3
           unsigned int ConfStepCnt[11] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049 };  //CONF_STEP_CNT = 3
__device__ unsigned int ConfStepCntDev[11] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049 };  //CONF_STEP_CNT = 3
#elif __CONFORMATION_CANDIDATES_NUM__ == 2
           unsigned int ConfStepCnt[11] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };  //CONF_STEP_CNT = 2
__device__ unsigned int ConfStepCntDev[11] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };  //CONF_STEP_CNT = 2
#endif

//0:H; 1:C; 2:N; 3:O; 4:F; 5:P; 6:S; 7:Cl;
float VanderWaalsHost[8][8] =
{
	// H      C     N    O      F     P     S     Cl
	/*H*/ {2.4f , 2.90f, 2.75f, 2.72f, 2.67f, 3.00f, 3.00f, 2.95f},
	/*C*/ {2.90f, 3.40f, 3.25f, 3.22f, 3.17f, 3.07f, 3.58f, 3.54f},
	/*N*/ {2.75f, 3.25f, 3.10f, 3.07f, 3.02f, 3.35f, 3.35f, 3.30f},
	/*O*/ {2.72f, 3.22f, 3.07f, 3.04f, 2.99f, 3.32f, 3.32f, 3.27f},
	/*F*/ {2.67f, 3.17f, 3.02f, 2.99f, 2.94f, 3.27f, 3.27f, 3.22f},
	/*P*/ {3.00f, 3.50f, 3.35f, 3.32f, 3.27f, 3.60f, 3.60f, 3.55f},
	/*S*/ {3.00f, 3.50f, 3.35f, 3.32f, 3.27f, 3.60f, 3.60f, 3.55f},
	/*Cl*/{2.95f, 3.54f, 3.30f, 3.27f, 3.22f, 3.55f, 3.55f, 3.50f}
};
__device__ float VanderWaalsDev[8][8] =
{
	// H      C     N    O      F     P     S     Cl
	/*H*/ {2.4f , 2.90, 2.75, 2.72, 2.67, 3.00, 3.00, 2.95},
	/*C*/ {2.90f, 3.40, 3.25, 3.22, 3.17, 3.07, 3.58, 3.54},
	/*N*/ {2.75f, 3.25, 3.10, 3.07, 3.02, 3.35, 3.35, 3.30},
	/*O*/ {2.72f, 3.22, 3.07, 3.04, 2.99, 3.32, 3.32, 3.27},
	/*F*/ {2.67f, 3.17, 3.02, 2.99, 2.94, 3.27, 3.27, 3.22},
	/*P*/ {3.00f, 3.50, 3.35, 3.32, 3.27, 3.60, 3.60, 3.55},
	/*S*/ {3.00f, 3.50, 3.35, 3.32, 3.27, 3.60, 3.60, 3.55},
	/*Cl*/{2.95f, 3.54, 3.30, 3.27, 3.22, 3.55, 3.55, 3.50}
};
__device__ int CrashTIdxMapDev[3][3] =
{ {-1, 0, 1},
  { 0,-1, 2},
  { 1, 2,-1},
};
int CrashTIdxMapHost[CRASH_T_CNT - 1][CRASH_T_CNT - 1] =
{ {-1, 0, 1, 2},
  { 0,-1, 3, 4},
  { 1, 3,-1, 5},
  { 2, 4, 5,-1}
};
/*--------------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------Identify a pair of atoms is on the bond-------------------------*/
//Function:Check if a pair of atoms is on the same bond or not
//Argument:
//  [in ]1.const BOND* bond:bond data
//  [in ]2.int bond_cnt: the number of global bonds
//  [in ]3.int frag1_idx:the index of fragment1 which atom1 is on
//  [in ]4.int frag2_idx:the index of fragment2 which atom2 is on
//  [in ]3.int atom1_idx:the index of atom1
//  [in ]4.int atom2_idx:the index of atom2
//Return value: 1--the pair of atoms is on the bond; 0--the pair of atoms isn't on the bond
__device__ inline bool IsOnBondDev(const BOND* bond, int bond_cnt, int frag1_idx, int frag2_idx, int atom1_idx, int atom2_idx)
{
	for (int i = 0; i < bond_cnt; i++)
	{
		if ((bond[i].frag_idx[0] == frag1_idx && bond[i].frag_idx[1] == frag2_idx && bond[i].atom_idx[0] == atom1_idx && bond[i].atom_idx[1] == atom2_idx)
		 || (bond[i].frag_idx[0] == frag2_idx && bond[i].frag_idx[1] == frag1_idx && bond[i].atom_idx[0] == atom2_idx && bond[i].atom_idx[1] == atom1_idx))

		{
			return true;
		}
	}			
	return false;
}
/*------------------------------------------------------------------------------------------------*/

/*----------------------------------extract legal groups ----------------------------------*/
//Function: Extract legal groups from group search space
//Argument: 
//  [out]1.unsigned int* pcombidx: table recording indexes of groups  
//  [in ]2.unsigned char* pconflictT: lookup table recording conflict flags of an intra-group
//  [in ]3.unsigned long long conflictT_byte_cnt: the size in bytes of the lookup table
//Reurn value: return the number of legal groups
unsigned int GetUnconflictedGroups(unsigned int* pcombidx, unsigned char* pconflictT, unsigned long long conflictT_byte_cnt)
{
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < conflictT_byte_cnt; i++)
	{
		if (0 == pconflictT[i])
		{
			pcombidx[cnt++] = i;
		}
	}
	return cnt;
}
/*-------------------------------------------------------------------------------------------*/

/*--------------------------------inra-group legality check------------------------------------*/
//Function: Check if some intra-combinations of some group conflict or not and fill the look up table
//Argument: 
//  [in ]1.FRAGMENT** pfragsr: the source fragments data 
//  [in ]2.BOND* bond: the bonds data
//  [in ]3.int frag_idx0:the possible fragment index of the group 
//  [in ]4.int frag_idx1:the possible fragment index of the group 
//  [in ]5.int frag_idx2:the possible fragment index of the group 
//  [in ]6.int frag_idx3:the possible fragment index of the group 
//  [in ]7.int frag_idx4:the possible fragment index of the group 
//  [in ]8.unsigned long long frag_poses_cnt: the number of combinations of different fragments poses in the group
//  [in ]9.int group_idx:current group index
//  [out]10.unsigned  char* pconflictTDev: look up table: 0--unconflicted; 1--conflicted
//  [in ]11.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void GPUCheckGroup(FRAGMENT** pfragsrc, BOND* bond, int frag_idx0, int frag_idx1,int frag_idx2, int frag_idx3,int frag_idx4, unsigned long long frag_poses_cnt, int group_idx, unsigned  char* pconflictTDev,float threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < frag_poses_cnt)
	{		
		unsigned int pose[3];
		if      (group_idx == 0)//3个片段(group0)
		{
			pose[0] = idx / ConfStepCntDev[2];//片段0姿态
			pose[1] = idx / CONF_STEP_CNT;//片段1姿态
			pose[2] = idx;//片段2姿态
			
			for (int m = 0; m < pfragsrc[frag_idx0][pose[0]].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx1][pose[1]].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx1, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[0]].atom[m].X - pfragsrc[frag_idx1][pose[1]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[0]].atom[m].Y - pfragsrc[frag_idx1][pose[1]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[0]].atom[m].Z - pfragsrc[frag_idx1][pose[1]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx1][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
			for (int m = 0; m < pfragsrc[frag_idx0][pose[0]].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx2][pose[2]].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx2, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[0]].atom[m].X - pfragsrc[frag_idx2][pose[2]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[0]].atom[m].Y - pfragsrc[frag_idx2][pose[2]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[0]].atom[m].Z - pfragsrc[frag_idx2][pose[2]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx2][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
			for (int m = 0; m < pfragsrc[frag_idx1][pose[1]].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx2][pose[2]].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx1, frag_idx2, m, n)) continue;
					float dx = pfragsrc[frag_idx1][pose[1]].atom[m].X - pfragsrc[frag_idx2][pose[2]].atom[n].X;
					float dy = pfragsrc[frag_idx1][pose[1]].atom[m].Y - pfragsrc[frag_idx2][pose[2]].atom[n].Y;
					float dz = pfragsrc[frag_idx1][pose[1]].atom[m].Z - pfragsrc[frag_idx2][pose[2]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx1][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx2][0].atom[n].van_table_idx;

					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
		}
		else if (group_idx == 1)//3个片段(group1)
		{
			pose[0] = idx / ConfStepCntDev[2];//片段3姿态
			pose[1] = idx / CONF_STEP_CNT;//片段4姿态
			pose[2] = idx;//片段5姿态
			for (int m = 0; m < pfragsrc[frag_idx0][0].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx1][0].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx1, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[0]].atom[m].X - pfragsrc[frag_idx1][pose[1]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[0]].atom[m].Y - pfragsrc[frag_idx1][pose[1]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[0]].atom[m].Z - pfragsrc[frag_idx1][pose[1]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx1][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
			for (int m = 0; m < pfragsrc[frag_idx0][0].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx2][0].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx2, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[0]].atom[m].X - pfragsrc[frag_idx2][pose[2]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[0]].atom[m].Y - pfragsrc[frag_idx2][pose[2]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[0]].atom[m].Z - pfragsrc[frag_idx2][pose[2]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx2][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
			for (int m = 0; m < pfragsrc[frag_idx1][0].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx2][0].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx1, frag_idx2, m, n)) continue;
					float dx = pfragsrc[frag_idx1][pose[1]].atom[m].X - pfragsrc[frag_idx2][pose[2]].atom[n].X;
					float dy = pfragsrc[frag_idx1][pose[1]].atom[m].Y - pfragsrc[frag_idx2][pose[2]].atom[n].Y;
					float dz = pfragsrc[frag_idx1][pose[1]].atom[m].Z - pfragsrc[frag_idx2][pose[2]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx1][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx2][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
		}
		else if (group_idx == 2)//2个片段(group2)
		{
			pose[0] = idx / CONF_STEP_CNT;//片段6姿态
			pose[1] = idx;//片段7姿态
			for (int m = 0; m < pfragsrc[frag_idx0][pose[frag_idx0 - 6]].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx1][pose[frag_idx1 - 6]].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx1, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[frag_idx0-6]].atom[m].X - pfragsrc[frag_idx1][pose[frag_idx1-6]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[frag_idx0-6]].atom[m].Y - pfragsrc[frag_idx1][pose[frag_idx1-6]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[frag_idx0-6]].atom[m].Z - pfragsrc[frag_idx1][pose[frag_idx1-6]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx1][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
		}
		else if (group_idx == 3)//2个片段(group3)
		{
			pose[0] = idx / CONF_STEP_CNT;//片段8姿态
			pose[1] = idx;//片段9姿态
			for (int m = 0; m < pfragsrc[frag_idx0][pose[frag_idx0 - 8]].atom_cnt; m++)
			{
				for (int n = 0; n < pfragsrc[frag_idx1][pose[frag_idx1 - 8]].atom_cnt; n++)
				{
					if (IsOnBondDev(bond, 9, frag_idx0, frag_idx1, m, n)) continue;
					float dx = pfragsrc[frag_idx0][pose[frag_idx0-8]].atom[m].X - pfragsrc[frag_idx1][pose[frag_idx1-8]].atom[n].X;
					float dy = pfragsrc[frag_idx0][pose[frag_idx0-8]].atom[m].Y - pfragsrc[frag_idx1][pose[frag_idx1-8]].atom[n].Y;
					float dz = pfragsrc[frag_idx0][pose[frag_idx0-8]].atom[m].Z - pfragsrc[frag_idx1][pose[frag_idx1-8]].atom[n].Z;

					int vanidx1 = pfragsrc[frag_idx0][0].atom[m].van_table_idx;
					int vanidx2 = pfragsrc[frag_idx1][0].atom[n].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						pconflictTDev[idx] = 1;
						return;
					}
				}
			}
		}
	}
}
/*---------------------------------------------------------------------------------------------*/

/*---------------------------------extract legal groups -------------------------------------*/
//Function: Check if some combinations of group0 conflict or not and fill the look up table
//Argument: 
//  [out]1.GROUP& group: the legal groups 
//  [in ]2.FRAGMENT** pfragsrc:the source fragments data 
//  [in ]3.unsigned int* frag_poses_cnt: the number of combinations of different fragments poses in group0
//  [out]4.unsigned int* pcombidx: table recording indexes of legal groups 
//  [in ]5.int frag_cnt:the number of fragments
//  [in ]6.int group_idx:current group index
//  [in ]7.BOND* bond_dev: the bonds data
//  [in ]8.double threshold_factor:conflict threshold coefficent
//Reurn value: the number of unconflicted groups
unsigned int  LoadGroup(GROUP& group, FRAGMENT** pfragsrc, unsigned int* frag_poses_cnt, unsigned int* pcombidx, int frag_cnt, int group_idx ,BOND* bond_dev, float threshold_factor)
{
	unsigned int discrashed_comb_cnt = 0;
	unsigned char* pconflictTDev;
	unsigned char* pconflictTHost;
	switch (group_idx)
	{
		//group0: frag 0,1,2
		case 0:	pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[2]);
				cudaMalloc(&pconflictTDev, frag_poses_cnt[2]);
				cudaMemset(pconflictTDev, 0, frag_poses_cnt[2]);
				GPUCheckGroup << <(frag_poses_cnt[2]>>9)+1, 512 >> > (pfragsrc, bond_dev, 0,1,2,-1, -1, frag_poses_cnt[2], 0, pconflictTDev, threshold_factor);
				cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[2], cudaMemcpyDeviceToHost);
				discrashed_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[2]);
				group.frag_idx[0] = 0;
				group.frag_idx[1] = 1;
				group.frag_idx[2] = 2;
				break;
		//group1: frag 3,4,5
		case 1: pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[5]);
				cudaMalloc(&pconflictTDev, frag_poses_cnt[5]);
				cudaMemset(pconflictTDev, 0, frag_poses_cnt[5]);
				GPUCheckGroup << <(frag_poses_cnt[5]>>9)+1, 512 >> > (pfragsrc, bond_dev, 3,4,5,-1, -1, frag_poses_cnt[5], 1, pconflictTDev, threshold_factor);
				cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[5], cudaMemcpyDeviceToHost);
				discrashed_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[5]);
				group.frag_idx[0] = 3;
				group.frag_idx[1] = 4;
				group.frag_idx[2] = 5;
				break;
		//group2: frag 6,7
		case 2: pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[7]);
				cudaMalloc(&pconflictTDev, frag_poses_cnt[7]);
				cudaMemset(pconflictTDev, 0, frag_poses_cnt[7]);
				GPUCheckGroup << < (frag_poses_cnt[7] >> 9) + 1, 512 >> > (pfragsrc, bond_dev, 6, 7, -1, -1, -1, frag_poses_cnt[7],2, pconflictTDev, threshold_factor);
				cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[7], cudaMemcpyDeviceToHost);
				discrashed_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[7]);
				group.frag_idx[0] = 6;
				group.frag_idx[1] = 7;
				break;
		//group2: frag 8,9
		case 3: pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[9]);
				cudaMalloc(&pconflictTDev, frag_poses_cnt[9]);
				cudaMemset(pconflictTDev, 0, frag_poses_cnt[9]);
				GPUCheckGroup << < (frag_poses_cnt[9] >> 9)+1, 512 >> > (pfragsrc, bond_dev, 8, 9, -1, -1, -1, frag_poses_cnt[9], 3, pconflictTDev, threshold_factor);
				cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[9], cudaMemcpyDeviceToHost);
				discrashed_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[9]);
				group.frag_idx[0] = 8;
				group.frag_idx[1] = 9;
				break;
	}
	cudaFree(pconflictTDev);
	free(pconflictTHost);

	group.frag_cnt = frag_cnt;
	group.group_idx = group_idx;
	group.pcombidx = pcombidx;
	group.comb_cnt = discrashed_comb_cnt;

	return discrashed_comb_cnt;
}
/*-------------------------------------------------------------------------------------------*/

/*----------------------------group & pocket legality check---------------------------------*/
//Function: Check if some combinations of groups & the pocket atoms conflict or not and fill the look up table
//Argument: 
//  [out]1.unsigned char* pconflictTDev: lookup table recording conflict flags of an group & pocket atoms
//  [in ]2.ATOM* pocket_atom:data of pocket atoms
//  [in ]3.FRAGMENT** psrc_frag:the source fragments data 
//  [in ]4.GROUP& group: the legal groups 
//  [in ]5.unsigned int* pcombidx: table recording indexes of legal groups 
//  [in ]5.int pocket_atom_cnt: number of pocket atoms 
//  [in ]7.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void CheckGroup0_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < group->comb_cnt )
	{
		long  long pose_idx[3];
		pose_idx[0] = pcombidx[idx] / ConfStepCntDev[2];
		pose_idx[1] = pcombidx[idx] /  CONF_STEP_CNT;
		pose_idx[2] = pcombidx[idx];
		for (int k = 0; k < psrc_frag[0][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[0][pose_idx[0]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[0][pose_idx[0]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[0][pose_idx[0]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[0][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[1][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[1][pose_idx[1]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[1][pose_idx[1]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[1][pose_idx[1]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[1][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[2][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[2][pose_idx[2]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[2][pose_idx[2]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[2][pose_idx[2]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[2][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
	}
}
__global__ void CheckGroup1_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < group->comb_cnt )
	{
		unsigned long  long pose_idx[3];
		pose_idx[0] = pcombidx[idx] / ConfStepCntDev[2];
		pose_idx[1] = pcombidx[idx] / (CONF_STEP_CNT);
		pose_idx[2] = pcombidx[idx];

		for (int k = 0; k < psrc_frag[3][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[3][pose_idx[0]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[3][pose_idx[0]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[3][pose_idx[0]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[3][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[4][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[4][pose_idx[1]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[4][pose_idx[1]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[4][pose_idx[1]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[4][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[5][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[5][pose_idx[2]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[5][pose_idx[2]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[5][pose_idx[2]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[5][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
	}
}
__global__ void CheckGroup2_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)group->comb_cnt)
	{
		unsigned long long  pose[2];
		pose[0] = pcombidx[idx] / (CONF_STEP_CNT);
		pose[1] = pcombidx[idx];

		for (int k = 0; k < psrc_frag[6][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx     = psrc_frag[6][pose[0]].atom[k].X - pocket_atom[m].X;
				float dy     = psrc_frag[6][pose[0]].atom[k].Y - pocket_atom[m].Y;
				float dz     = psrc_frag[6][pose[0]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1  = psrc_frag[6][0].atom[k].van_table_idx;
				int vanidx2  = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{	
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[7][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx = psrc_frag[7][pose[1]].atom[k].X - pocket_atom[m].X;
				float dy = psrc_frag[7][pose[1]].atom[k].Y - pocket_atom[m].Y;
				float dz = psrc_frag[7][pose[1]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1 = psrc_frag[7][0].atom[k].van_table_idx;
				int vanidx2 = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
	}
}
__global__ void CheckGroup3_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)group->comb_cnt)
	{
		unsigned long long  pose[2];
		pose[0] = pcombidx[idx] / (CONF_STEP_CNT);
		pose[1] = pcombidx[idx];

		for (int k = 0; k < psrc_frag[8][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx     = psrc_frag[8][pose[0]].atom[k].X - pocket_atom[m].X;
				float dy     = psrc_frag[8][pose[0]].atom[k].Y - pocket_atom[m].Y;
				float dz     = psrc_frag[8][pose[0]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1  = psrc_frag[8][0].atom[k].van_table_idx;
				int vanidx2  = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{	
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
		for (int k = 0; k < psrc_frag[9][0].atom_cnt; k++)
		{
			for (int m = 0; m < pocket_atom_cnt; m++)
			{
				float dx     = psrc_frag[9][pose[1]].atom[k].X - pocket_atom[m].X;
				float dy     = psrc_frag[9][pose[1]].atom[k].Y - pocket_atom[m].Y;
				float dz     = psrc_frag[9][pose[1]].atom[k].Z - pocket_atom[m].Z;
				int vanidx1  = psrc_frag[9][0].atom[k].van_table_idx;
				int vanidx2  = pocket_atom[m].van_table_idx;
				if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
				{	
					pconflictTDev[idx] = 1;
					return;
				}
			}
		}
	}
}
/*-------------------------------------------------------------------------------------------*/

/*------------------------------inter-group legality check-----------------------------------*/
//Function: Check if some combinations of group & groups conflict or not and fill the look up table
//Argument: 
//   [out]1.unsigned char* pconflictTDev: lookup table recording conflict flags of an group & group
//   [in ]2.FRAGMENT** psrc_frag:the source fragments data 
//   [in ]3.BOND* bond_dev: the bonds data
//   [in ]4.unsigned int conflictTlen: number of bytes of the conflict table
//   [in ]5.unsigned int* pcomidx_dev: table recording indexes of legal groups 
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void CheckGroups_01(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pose[6];
	if (idx < conflictTlen)
	{
		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[5];
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[4];
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[3];

		pose[3] = pcomidx_dev[idx] / ConfStepCntDev[2];
		pose[4] = pcomidx_dev[idx] / ConfStepCntDev[1];
		pose[5] = pcomidx_dev[idx];

		for (int m = 0; m < 3; m++)
		{
			for (int n = 3; n < 6; n++)
			{
				for (int i = 0; i < psrc_frag[m][0].atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n][0].atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose[m]].atom[i].X - psrc_frag[n][pose[n]].atom[j].X;
						float dy = psrc_frag[m][pose[m]].atom[i].Y - psrc_frag[n][pose[n]].atom[j].Y;
						float dz = psrc_frag[m][pose[m]].atom[i].Z - psrc_frag[n][pose[n]].atom[j].Z;
						int vanidx1 = psrc_frag[m][pose[m]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n]].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
__global__ void CheckGroups_02(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)conflictTlen)
	{
		unsigned long long pose[5];

		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[7];//0
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[6];//1
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[5];//2
		pose[3] = pcomidx_dev[idx] / (CONF_STEP_CNT); // 6
		pose[4] = pcomidx_dev[idx];//7

		for (int m = 0; m < 3; m++)
		{
			for (int n = 6; n < 8; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;

						float dx = psrc_frag[m][pose[m]].atom[i].X - psrc_frag[n][pose[n-3]].atom[j].X;
						float dy = psrc_frag[m][pose[m]].atom[i].Y - psrc_frag[n][pose[n-3]].atom[j].Y;
						float dz = psrc_frag[m][pose[m]].atom[i].Z - psrc_frag[n][pose[n-3]].atom[j].Z;

						int vanidx1 = psrc_frag[m][pose[m]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n-3]].atom[j].van_table_idx;

						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
__global__ void CheckGroups_12(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pose[5];
	if (idx < conflictTlen)
	{
		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[4]; //3
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[3]; //4
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[2]; //5

		pose[3] = pcomidx_dev[idx] / CONF_STEP_CNT; //6
		pose[4] = pcomidx_dev[idx]; //7

		for (int m = 3; m < 6; m++)
		{
			for (int n = 6; n < 8; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose[m-3]].atom[i].X - psrc_frag[n][pose[n-3]].atom[j].X;
						float dy = psrc_frag[m][pose[m-3]].atom[i].Y - psrc_frag[n][pose[n-3]].atom[j].Y;
						float dz = psrc_frag[m][pose[m-3]].atom[i].Z - psrc_frag[n][pose[n-3]].atom[j].Z;
						int vanidx1 = psrc_frag[m][0].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][0].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
__global__ void CheckGroups_03(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)conflictTlen)
	{
		unsigned long long pose[5];

		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[9];//0
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[8];//1
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[7];//2
		pose[3] = pcomidx_dev[idx] / (CONF_STEP_CNT); // 8
		pose[4] = pcomidx_dev[idx];//9

		for (int m = 0; m < 3; m++)
		{
			for (int n = 8; n < 10; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;

						float dx = psrc_frag[m][pose[m]].atom[i].X - psrc_frag[n][pose[n-5]].atom[j].X;
						float dy = psrc_frag[m][pose[m]].atom[i].Y - psrc_frag[n][pose[n-5]].atom[j].Y;
						float dz = psrc_frag[m][pose[m]].atom[i].Z - psrc_frag[n][pose[n-5]].atom[j].Z;

						int vanidx1 = psrc_frag[m][pose[m]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n-5]].atom[j].van_table_idx;

						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
__global__ void CheckGroups_13(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pose[5];
	if (idx < conflictTlen)
	{
		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[6]; //3
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[5]; //4
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[4]; //5

		pose[3] = pcomidx_dev[idx] / CONF_STEP_CNT; //8
		pose[4] = pcomidx_dev[idx]; //9

		for (int m = 3; m < 6; m++)
		{
			for (int n = 8; n < 10; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose[m-3]].atom[i].X - psrc_frag[n][pose[n-5]].atom[j].X;
						float dy = psrc_frag[m][pose[m-3]].atom[i].Y - psrc_frag[n][pose[n-5]].atom[j].Y;
						float dz = psrc_frag[m][pose[m-3]].atom[i].Z - psrc_frag[n][pose[n-5]].atom[j].Z;
						int vanidx1 = psrc_frag[m][pose[m-3]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n-5]].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
__global__ void CheckGroups_23(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond_dev, unsigned int* pcomidx_dev, unsigned long long conflictTlen, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pose[4];
	if (idx < conflictTlen)
	{
		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[3]; //6
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[2]; //7

		pose[2] = pcomidx_dev[idx] / CONF_STEP_CNT; //8
		pose[3] = pcomidx_dev[idx]; //9

		for (int m = 6; m < 8; m++)
		{
			for (int n = 8; n < 10; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond_dev, 9, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose[m-6]].atom[i].X - psrc_frag[n][pose[n-6]].atom[j].X;
						float dy = psrc_frag[m][pose[m-6]].atom[i].Y - psrc_frag[n][pose[n-6]].atom[j].Y;
						float dz = psrc_frag[m][pose[m-6]].atom[i].Z - psrc_frag[n][pose[n-6]].atom[j].Z;
						int vanidx1 = psrc_frag[m][pose[m-6]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n-6]].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pconflictTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
/*-------------------------------------------------------------------------------------------*/


/*---------------------compress the conflict table to a smaller size-------------------------*/
//Function: compress the conflict table
//Argument:
//  [in ]1.unsigned char* pconflictTDev:source conflict table
//  [out]2.unsigned char* pcrashCompressedTDev:compressed conflict table
//  [in ]3.unsigned long long bytes:the number of bytes of the source conflict table
//Return value:
__global__ void CompressCrashT(unsigned char* pconflictTDev, unsigned char* pcrashCompressedTDev, unsigned long long bytes)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < bytes && ((idx & 7) == 0))
	{
		pcrashCompressedTDev[idx >> 3] |= (pconflictTDev[idx] | (pconflictTDev[idx + 1] << 1) | (pconflictTDev[idx + 2] << 2) | (pconflictTDev[idx + 3] << 3) | (pconflictTDev[idx + 4] << 4) | (pconflictTDev[idx + 5] << 5) | (pconflictTDev[idx + 6] << 6) | (pconflictTDev[idx + 7] << 7));
	}
}
/*-------------------------------------------------------------------------------------------*/

/*--------------------build inter-group conflict table available for DFS---------------------*/
//Function: build inter-group conflict table available for DFS
//Argument:
//   [out]1.unsigned char** pGroupsCrashTHost:conflict look up table
//   [in ]2.FRAGMENT** pfrag:the source fragments data 
//   [in ]3.BOND* bond_dev:the bonds data
//   [in ]4.GROUP* groups:groups data
//   [in ]5.unsigned long long len:maximum number of bytes of lookup table occupying
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Return value:
void BuildGroups01ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int *pcomidx_dev;
	
	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[1].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0,len);
	CheckGroups_01 <<<(len >> 9)+1,512>>>(ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len, threshold_factor);
	
	pGroupsCrashTHost[0] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[0], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[0][i]));
	}

	std::cout <<"Group(0) vs Group(1): " << ((float)cnt) / len << std::endl;
}
void BuildGroups02ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0, len);
	CheckGroups_02<< < (len >> 9)+1, 512 >> > (ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len, threshold_factor);

	pGroupsCrashTHost[1] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[1], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[1][i]));
	}

	std::cout << "Group(0) vs Group(2): " << ((float)cnt) / len << std::endl;
}
void BuildGroups03ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[3].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0, len);
	CheckGroups_03<< < (len >> 9)+1, 512 >> > (ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len,  threshold_factor);

	pGroupsCrashTHost[2] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[2], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[2][i]));
	}

	std::cout << "Group(0) vs Group(3): " << ((float)cnt) / len << std::endl;
}
void BuildGroups12ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0, len);
	CheckGroups_12 << < (len >> 9)+1, 512 >> > (ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len, threshold_factor);

	pGroupsCrashTHost[3] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[3], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[3][i]));
	}

	std::cout << "Group(1) vs Group(2): " << ((float)cnt) / len << std::endl;
}
void BuildGroups13ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[3].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0, len);
	CheckGroups_13<< < (len >> 9)+1, 512 >> > (ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len,  threshold_factor);

	pGroupsCrashTHost[4] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[4], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[4][i]));
	}

	std::cout << "Group(1) vs Group(3): " << ((float)cnt) / len << std::endl;
}
void BuildGroups23ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, float threshold_factor)
{
	unsigned char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, len);
	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[3].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, 0, len);
	CheckGroups_23<< < (len >> 9)+1, 512 >> > (ptmp_crashTDev, pfrag, bond_dev, pcomidx_dev, len,  threshold_factor);

	pGroupsCrashTHost[5] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[5], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[5][i]));
	}

	std::cout << "Group(2) vs Group(3): " << ((float)cnt) / len << std::endl;
}
/*-------------------------------------------------------------------------------------------*/

/*-------------------build group & pocket conflict table available for DFS-------------------*/
//Function: build group & pocket conflict table available for DFS
//Argument:
//   [out]1.unsigned char** pGroupsCrashTHost:conflict look up table
//   [in ]2.ATOM* pocket_atom_dev:pocket atom data
//   [in ]3.FRAGMENT** psrc_frag:the source fragments data 
//   [in ]4.GROUP* groups:groups data
//   [in ]5.int pocket_atom_cnt:number of pocket atoms
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Return value:
void BuildGroup0_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[0] = new unsigned char[groups[0].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[0].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[0], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[0].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[0].pcombidx, sizeof(unsigned int)* groups[0].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组0(片段0-1)和口袋冲5
	cudaMemset(pPocketConflictTDev, 0, groups[0].comb_cnt);
	CheckGroup0_Pocket<<<(groups[0].comb_cnt >> 9)+1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt, threshold_factor);
	cudaMemcpy(pPocketConflictTHost[0], pPocketConflictTDev, groups[0].comb_cnt, cudaMemcpyDeviceToHost);
	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[0].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[0][i]);
	}
	std::cout << "Groups(0) vs Pocket :" << ((float)cnt) / groups[0].comb_cnt << std::endl;
	
	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
}
void BuildGroup1_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[1] = new unsigned char[groups[1].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[1].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[1], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[1].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[1].pcombidx, sizeof(unsigned int)* groups[1].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组1(片段2-3)和口袋冲5
	cudaMemset(pPocketConflictTDev, 0, groups[1].comb_cnt);
	CheckGroup1_Pocket<<<(groups[1].comb_cnt >> 9)+1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt,threshold_factor);
	cudaMemcpy(pPocketConflictTHost[1], pPocketConflictTDev, groups[1].comb_cnt, cudaMemcpyDeviceToHost);

	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[1].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[1][i]);
	}
	std::cout << "Groups(1) vs Pocket :" << ((float)cnt) / groups[1].comb_cnt << std::endl;

	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
}
void BuildGroup2_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[2] = new unsigned char[groups[2].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[2].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[2], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[2].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int)* groups[2].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组1(片段4-5)和口袋冲5
	cudaMemset(pPocketConflictTDev, 0, groups[2].comb_cnt);
	CheckGroup2_Pocket << <(groups[2].comb_cnt >> 9) + 1, 512 >> > (pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev, pcomidx_dev, pocket_atom_cnt, threshold_factor);
	cudaMemcpy(pPocketConflictTHost[2], pPocketConflictTDev, groups[2].comb_cnt, cudaMemcpyDeviceToHost);

	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[2].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[2][i]);
	}
	std::cout << "Groups(2) vs Pocket :" << ((float)cnt) / groups[2].comb_cnt << std::endl;
	cudaError a = cudaGetLastError();

	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
}
void BuildGroup3_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[3] = new unsigned char[groups[3].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[3].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[3], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[3].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[3].pcombidx, sizeof(unsigned int)* groups[3].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组1(片段4-5)和口袋冲5
	cudaMemset(pPocketConflictTDev, (unsigned char)0, groups[3].comb_cnt);
	CheckGroup3_Pocket<<<(groups[3].comb_cnt >> 9)+1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt,threshold_factor );
	cudaMemcpy(pPocketConflictTHost[3], pPocketConflictTDev, groups[3].comb_cnt, cudaMemcpyDeviceToHost);

	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[3].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[3][i]);
	}
	std::cout << "Groups(3) vs Pocket :" << ((float)cnt) / groups[3].comb_cnt << std::endl;

	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
	//cudaFree(pocket_atom_dev);
}
/*-------------------------------------------------------------------------------------------*/
