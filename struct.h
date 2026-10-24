#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.141592654

#define EULER_STEP_RAD 6.28318530712
#define CONF_STEP_RAD 1.0471975512
#define EULER_STEP_CNT 1
#define CONF_STEP_CNT 6
#define FRAG_CNT 10 
#define M 10
#define __CONFORMATION_CANDIDATES_NUM__  CONF_STEP_CNT

#define CRASH_T_CNT 5
#define H_VAN_ATOM_IDX  0
#define C_VAN_ATOM_IDX  1
#define N_VAN_ATOM_IDX  2
#define O_VAN_ATOM_IDX  3
#define F_VAN_ATOM_IDX  4
#define P_VAN_ATOM_IDX  5
#define S_VAN_ATOM_IDX  6
#define CL_VAN_ATOM_IDX 7
#define VANDER_WAALS_N 8

typedef struct RESIDUE
{
    int residueNum;
    char residueName[4] = { 0 };

    int atomNum[62];
    char atomName[62][4] = { 0 };
    int van_idx[62] = { 0 };

    int atoms = 0;
    float atomX[62];
    float atomY[62];
    float atomZ[62];

    char link[2];
    float occupyRatio[62];
    float temperature[62];
}RESIDUE;
typedef struct ATOM
{
    int num;
    char name[4] = { 0 };
    int van_table_idx;
    char residue_name[4] = {0};
    float X;
    float Y;
    float Z;
}ATOM;

typedef struct FRAGMENT
{
    ATOM atom[9];
    int atom_cnt;
    int index;
    long long pose_cnt;
}FRAGMENT;
 
typedef struct GROUP
{
    int group_idx;
    int frag_cnt;
    int frag_idx[4];
    unsigned int* pcombidx;
    unsigned int comb_cnt;
    //int bond_atom_idx[2][2];
}GROUP;

typedef struct BOND
{
    int frag_idx[2];
    int atom_idx[2];
}BOND;