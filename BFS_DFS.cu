#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>
#include "struct.h"

using namespace std;
extern float VanderWaalsHost[VANDER_WAALS_N][VANDER_WAALS_N];
extern int CrashTIdxMapHost[CRASH_T_CNT-1][CRASH_T_CNT-1];
extern unsigned int ConfStepCnt[FRAG_CNT-1];  
extern __device__ unsigned int ConfStepCntDev[FRAG_CNT-1];

//init
extern int CPUReadPDBFile(FILE* pfile, unsigned char* pbuf);
extern int CPUGetAtomsInfo(const unsigned char* pbuf, ATOM* patom, unsigned int max_buf_len);
extern int CPUGetResidueInfo(const unsigned char* pbuf, unsigned char* residueName, RESIDUE& residue, unsigned int max_buf_len);
extern void CPULoadFragment(FRAGMENT* fragment, const RESIDUE& residue);
extern void CPUSetAtomVanTableIndex(FRAGMENT* pfrag, int frag_cnt);
extern bool CPUGetPocketGeoInfo(FRAGMENT* pfrag, int frag_cnt, float& cx, float& cy, float& cz, float& scope_minX, float& scope_maxX, float& scope_minY, float& scope_maxY, float& scope_minZ, float& scope_maxZ);
extern int CPUGetAtomsFromPocket(ATOM* psrc_data, ATOM* pdst_patom, RESIDUE* exclude_resdue, unsigned int src_len, unsigned int max_atom_cnt,
    float scope_minX, float scope_maxX, float scope_minY, float scope_maxY, float scope_minZ, float scope_maxZ, int exclude_resdue_cnt);
extern void Init(FILE** pfile, char* pbuf, ATOM* patom, ATOM* pocket_atom, RESIDUE& residue, FRAGMENT* fragment, BOND* bond, float& pocket_cx, float& pocket_cy, float& pocket_cz,
    float& scope_minX, float& scope_maxX, float& scope_minY, float& scope_maxY, float& scope_minZ, float& scope_maxZ, int& atom_cnt, unsigned int* frag_pose_cnt);
extern void PDBCor2PocketCor(FRAGMENT* fragment, ATOM* pocket_atom, int frag_cnt, int atom_cnt, float cx, float cy, float cz);
extern void LoadBond(BOND* bond);

//rotate
extern __global__ void GPUKernelBondAxisRMatrix(FRAGMENT** pfrag, float** R, unsigned int* frag_poses_cnt, int ref_frag_idx, int rot_frag_idx, int ref_atom_idx, int rot_atom_idx, int euler_step_cnt, int conf_step_cnt, float step_rad);
extern __global__ void GPUKernelEuler(FRAGMENT** pdst_frag, FRAGMENT* psrc_frag, float cx, float cy, float cz, int frag_cnt, int euler_step_cnt, int conf_step_cnt, float step_rad);
extern __global__ void GPUKernelConfRotate(FRAGMENT** pfrag, float** R, unsigned int* frag_poses_cnt, int* rot_frag_idx, int start_rot_frag_idx, int fragment_cnt, int euler_step_cnt, int conf_step_cnt, float step_rad);
extern __global__ void GPUKernelInitFragsAfterRotate(FRAGMENT** pdstfrag, FRAGMENT* psrcfrag, unsigned int* frag_poses_cnt, int frag_idx);
extern void Rotate(FRAGMENT* psrc_frag, FRAGMENT** pdst_frag, unsigned int* frag_poses_cnt, float pocket_cx, float pocket_cy, float pocket_cz);

__device__ bool IsOnBondDev(const BOND* bond, int bond_cnt, int frag1_idx, int frag2_idx, int atom1_idx, int atom2_idx);
extern unsigned int GetUnconflictedGroups(unsigned int* pcombidx, unsigned char* pcrashT, unsigned long long pcrashT_byte_cnt);
extern unsigned int LoadGroup(GROUP& group, FRAGMENT** pfragsrc, unsigned int* frag_poses_cnt, unsigned int* pcombidx, int frag_cnt, int group_idx, BOND* bond_dev, float threshold_fact);

extern void BuildGroups01ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroups02ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroups03ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroups12ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroups13ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroups23ConflictT(unsigned char** pGroupsConflictTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, float threshold_factor);
extern void BuildGroup0_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor);
extern void BuildGroup1_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor);
extern void BuildGroup2_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor);
extern void BuildGroup3_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, float threshold_factor);
extern  __global__ void CheckGroup0_Pocket(unsigned char* pConflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int* pcombidx, int  pocket_atom_cnt);
extern  __global__ void CheckGroup1_Pocket(unsigned char* pConflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int* pcombidx, int  pocket_atom_cnt);
extern  __global__ void CheckGroup2_Pocket(unsigned char* pConflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int* pcombidx, int  pocket_atom_cnt);
extern  __global__ void CheckGroup3_Pocket(unsigned char* pConflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int* pcombidx, int  pocket_atom_cnt);

unsigned long long  Stack[M];
unsigned int Sum = 0; // The number of solutions of proposed algorithm
unsigned int Sum_ = 0; // The number of solutions of Naive DFS
int NoPruningCnt[FRAG_CNT] = { 0 };

inline bool CPUIsOnBond(const BOND* bond, int bond_cnt, int frag1_idx, int frag2_idx, int atom1_idx, int atom2_idx)
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
inline bool CheckFragsConflicted(FRAGMENT** pfrag, BOND* pbond, int frag1_idx, int frag2_idx, unsigned long long frag1_pose, unsigned long long frag2_pose, float threshold_fact)
{
    for(int i=0; i< pfrag[frag1_idx]->atom_cnt; i++)
        for (int j = 0; j < pfrag[frag2_idx]->atom_cnt; j++)
        {
            if (CPUIsOnBond(pbond, 9, frag1_idx, frag2_idx, i, j)) continue;
            float dx = pfrag[frag1_idx][frag1_pose].atom[i].X - pfrag[frag2_idx][frag2_pose].atom[j].X;
            float dy = pfrag[frag1_idx][frag1_pose].atom[i].Y - pfrag[frag2_idx][frag2_pose].atom[j].Y;
            float dz = pfrag[frag1_idx][frag1_pose].atom[i].Z - pfrag[frag2_idx][frag2_pose].atom[j].Z;
            if(sqrt(dx*dx+dy*dy+dz*dz) < threshold_fact *VanderWaalsHost[pfrag[frag1_idx][frag1_pose].atom[i].van_table_idx][pfrag[frag2_idx][frag2_pose].atom[j].van_table_idx])
                return true;
        }
    return false;
}
inline bool CheckFragPocketConflicted(FRAGMENT** pfrag, ATOM* ppocket_atom, int frag1_idx, unsigned long long frag1_pose, int pocket_atom_cnt, float threshold_fact)
{
    for (int j = 0; j < pfrag[frag1_idx][frag1_pose].atom_cnt; j++)
    {
        for (int i = 0; i < pocket_atom_cnt; i++)
        {
            float dx = ppocket_atom[i].X - pfrag[frag1_idx][frag1_pose].atom[j].X;
            float dy = ppocket_atom[i].Y - pfrag[frag1_idx][frag1_pose].atom[j].Y;
            float dz = ppocket_atom[i].Z - pfrag[frag1_idx][frag1_pose].atom[j].Z;

            int vanidx1 = ppocket_atom[i].van_table_idx;
            int vanidx2 = pfrag[frag1_idx][frag1_pose].atom[j].van_table_idx;
           
            if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_fact * VanderWaalsHost[vanidx1][vanidx2])
                return true;
        }
    }
    return false;
}

//Function: DFS stage of proposed algorithm
//  [in]1.unsigned char** pPocketConflictTHost:the conflict table of groups & pocket
//  [in]2.unsigned char** pGroupsConflictTHost:the conflict table of groups & groups
//  [in]3.GROUP* group:the data of groups
//  [in]4.int k:the current search depth
//Return value:true--the end depth has been reached, false--backtracking
bool CPU_DFS(unsigned char** pPocketConflictTHost, unsigned char** pGroupsConflictTHost, GROUP* group, int k)
{
    int m = 0;
    if (k >= 4)
    {
        return true;
    }
    for (unsigned long long i = 0; i < group[k].comb_cnt; i++)
    {
        if (k > 0)
        {
            unsigned long long pre_pose = group[k].pcombidx[i] /  ConfStepCnt[2+(k==1)]; //片段3的姿态
            if     (pre_pose < (group[k - 1].pcombidx[Stack[k - 1]])) continue;
            else if(pre_pose > (group[k - 1].pcombidx[Stack[k - 1]])) break;
        }

        for (m = 0; m < k; m++)
        {
            if (pGroupsConflictTHost[m+k-(m<1)][i]) break;
        }
        if (k == m )
        {
            if (pPocketConflictTHost[k][i]) continue;
            Stack[k] = i;
            if (CPU_DFS(pPocketConflictTHost, pGroupsConflictTHost, group, k + 1))
            {
                Sum++;
            }
        }
    }
    return false;
}

//Function: Naive DFS
//  [in]1.FRAGMENT** pfrag: data of rigid fragments
//  [in]2.ATOM* ppocket_atom: the data of pocket atoms 
//  [in]3.BOND* pbond:the data of bond  
//  [in]4.int pocket_atom_cnt: the number of pocket atoms
//  [in]5.int k:the current search depth
//  [in]6.double threshold_factor:collision threshold factor
//Return value:true--the end depth has been reached, false--backtracking
bool CPU_NaiveDFS(FRAGMENT** pfrag, ATOM* ppocket_atom, BOND* pbond, int pocket_atom_cnt, int k, float threshold_fact)
{
    int m = 0;
    if (k >= 10)
    {
        return true;
    }
    for (unsigned long long i = 0; i < (unsigned long long)(pfrag[k]->pose_cnt); i++)
    {
        if (k > 0)
        {
            unsigned long long pre_pose = i / CONF_STEP_CNT;      //pre_pose应该是片段0的姿态
            if      (pre_pose < Stack[k-1]) continue;
            else if (pre_pose > Stack[k-1]) break;
        }
        for (m = 0; m < k; m++)
        {
            if (CheckFragsConflicted(pfrag, pbond, k,m, i, Stack[m], threshold_fact))  break;
        }
        if (m == k)
        {
            if (CheckFragPocketConflicted(pfrag, ppocket_atom, k, i, pocket_atom_cnt, threshold_fact))   continue;
            Stack[k] = i;
            NoPruningCnt[k]++;
            if (CPU_NaiveDFS(pfrag, ppocket_atom, pbond, pocket_atom_cnt, k + 1, threshold_fact))
            {
                Sum_++;
            }
        }
    }
    return false;
}

//Function: Calculate & Display the runtime of the two algorithms
//Arguments:
//  [out]1.const tm* time: time structure
//  [in ]int start_time_idx:the start moment
//  [int] end_time_idx:the end moment
//Return value:the runtime in seconds
int CalcAndDisplayTime_s(const tm* time, int start_time_idx, int end_time_idx)
{
    int dsec = time[end_time_idx].tm_hour * 3600 + time[end_time_idx].tm_min * 60 + time[end_time_idx].tm_sec - time[start_time_idx].tm_hour * 3600 - time[start_time_idx].tm_min * 60 - time[start_time_idx].tm_sec;
    int dh = dsec / 3600;
    int dm = dsec / 60 % 60;
    int ds = dsec % 60;
    cout << dh << "(h):" << dm << "(m):" << ds << "(s)" << endl;
    return dsec;
}

//Function: Display the statics
//  [in]1.FRAGMENT** phost_frag:
//  [in]2.int normal_secs:the runtime of proposed algorithm
//  [in]3.int dbd_secs:the runtime of proposed algorithm
//Return value:
void StaticsAndDisplay(FRAGMENT * *phost_frag, int normal_secs, int dbd_secs)
{
    unsigned long long aver_pruning_depth = 0;
    unsigned long long pruning_cnt = 0;
    unsigned long long total_node_cnt = 0;
    for (int k = 0; k < FRAG_CNT; k++)
    {
        total_node_cnt += phost_frag[k]->pose_cnt;
        aver_pruning_depth += ((k + 1) * (phost_frag[k]->pose_cnt - NoPruningCnt[k]));
        pruning_cnt += (phost_frag[k]->pose_cnt - NoPruningCnt[k]);
    }
    cout << "Pruning Ratio:" << (float)pruning_cnt / total_node_cnt << endl;
    cout << "Pruning Depth:" << (float)aver_pruning_depth / pruning_cnt << endl;
    cout << "Speedup      :" << (float)normal_secs / dbd_secs << endl;
}

void Solve(GROUP* group, FRAGMENT** pdst_frag, FRAGMENT** phost_frag, BOND* bond_dev, BOND* bond_host, ATOM* pocket_atom_dev, ATOM* pocket_atom_host,unsigned int* frag_poses_cnt, unsigned int** pcombidx, float threshold_fact, int atom_cnt)
{
    static int cnt = 0;
    unsigned char* pGroupsCrashTHost[6];
    unsigned char* pPocketCrashTHost[4];

    tm time[4];
    int normal_secs, int dbd_secs;

    cout << "Proposed algorithm running..." << endl;
    cout << "============ Process" << " " << cnt++ << " ============" << endl;

    /**************************************** DBD algorithm ***********************************************************/
    std::cout << "BFS+DFS passing ratio:" << std::endl;
    _getsystime(&time[0]); // 记录起始时间

    //把原问题划分为4组
    LoadGroup(group[0], pdst_frag, frag_poses_cnt, pcombidx[0], 3, 0, bond_dev, threshold_fact);
    LoadGroup(group[1], pdst_frag, frag_poses_cnt, pcombidx[1], 3, 1, bond_dev, threshold_fact);
    LoadGroup(group[2], pdst_frag, frag_poses_cnt, pcombidx[2], 2, 2, bond_dev, threshold_fact);
    LoadGroup(group[3], pdst_frag, frag_poses_cnt, pcombidx[3], 2, 3, bond_dev, threshold_fact);

    ////构建每对组之间的冲突表
    BuildGroups01ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[1].comb_cnt, threshold_fact);
    BuildGroups02ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[2].comb_cnt, threshold_fact);
    BuildGroups03ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[3].comb_cnt, threshold_fact);
    BuildGroups12ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[2].comb_cnt, threshold_fact);
    BuildGroups13ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[3].comb_cnt, threshold_fact);
    BuildGroups23ConflictT(pGroupsCrashTHost, pdst_frag, bond_dev, group, (unsigned long long)group[3].comb_cnt, threshold_fact);

    ////构建每组与口袋原子之间的冲突表
    BuildGroup0_PocketConflictT(pPocketCrashTHost, pocket_atom_dev, pdst_frag, group, atom_cnt, threshold_fact);
    BuildGroup1_PocketConflictT(pPocketCrashTHost, pocket_atom_dev, pdst_frag, group, atom_cnt, threshold_fact);
    BuildGroup2_PocketConflictT(pPocketCrashTHost, pocket_atom_dev, pdst_frag, group, atom_cnt, threshold_fact);
    BuildGroup3_PocketConflictT(pPocketCrashTHost, pocket_atom_dev, pdst_frag, group, atom_cnt, threshold_fact);

    //DFS 
    CPU_DFS(pPocketCrashTHost, pGroupsCrashTHost, group, 0);
    _getsystime(&time[1]);

    //打印结果
    cout << "BFS+DFS   : " << Sum << endl; // 输出解数
    dbd_secs = CalcAndDisplayTime_s(time, 0, 1);//计算并显示耗时
    cout << "----------------------------------------------------" << endl; // 显示分割

    /**************************************** Naive DFS ***********************************************************/

    cout << "Naive DFS running..." << endl;
    _getsystime(&time[2]);
    CPU_NaiveDFS(phost_frag, pocket_atom_host, bond_host, atom_cnt, 0, threshold_fact);
    _getsystime(&time[3]);

    //打印结果
    cout << "Naive DFS : " << Sum_ << endl; // 输出解数
    normal_secs = CalcAndDisplayTime_s(time, 2, 3);//计算并显示耗时
    cout << "----------------------------------------------------" << endl; // 显示分割

    /******************************************* Statics ***********************************************************/
    StaticsAndDisplay(phost_frag, normal_secs, dbd_secs);
    cout << "----------------------------------------------------" << endl; // 显示分割

    for (int i = 0; i < 6; i++)
    {
        delete [] pGroupsCrashTHost[i];
        if (i < 4)
            delete[]pPocketCrashTHost[i];
    }

}
int main()
{
    //程序运行所需数据和内存区
    FILE* pfile = NULL;
    char* pbuf = new char[4096 * 4096];
    ATOM* patom = new ATOM[4096 * 4096];
    RESIDUE residue;
    FRAGMENT fragment[FRAG_CNT];
    BOND bond[10];
    float pocket_cx = 0.0;
    float pocket_cy = 0.0;
    float pocket_cz = 0.0;
    int atom_cnt = 4096 * 4096;
    float scope_minX, scope_maxX, scope_minY, scope_maxY, scope_minZ, scope_maxZ;
    FRAGMENT* psrc_frag;
    FRAGMENT** pdst_frag;
    FRAGMENT* ptmp_frag[FRAG_CNT];
    unsigned int frag_poses_cnt[FRAG_CNT];

    cout << "Paclitaxel" << endl;
    cout << "Reading file..." << endl;
    //初始化
    ATOM* pocket_atom = new ATOM[atom_cnt];
    Init(&pfile, pbuf, patom, pocket_atom, residue, fragment, bond, pocket_cx, pocket_cy, pocket_cz,
        scope_minX, scope_maxX, scope_minY, scope_maxY, scope_minZ, scope_maxZ, atom_cnt, frag_poses_cnt);
   
    delete [] pbuf;
    delete [] patom;

    cout << "Allocating memory..." << endl;

    //刚性片段数据存储空间建立
    cudaMalloc(&psrc_frag, sizeof(FRAGMENT) * FRAG_CNT);
    cudaMalloc(&pdst_frag, sizeof(FRAGMENT*) * FRAG_CNT);

    for (int i = 0; i < FRAG_CNT; i++)
    {
        cudaMalloc(&ptmp_frag[i], sizeof(FRAGMENT) * frag_poses_cnt[i]);
    }
    cudaMemcpy(pdst_frag, ptmp_frag, sizeof(FRAGMENT*) * FRAG_CNT, cudaMemcpyHostToDevice);
    cudaMemcpy(psrc_frag, fragment , sizeof(FRAGMENT) * FRAG_CNT, cudaMemcpyHostToDevice);

    cout << "Calculate poses..." << endl;
    //枚举所有片段所有原子的所有朝向旋转
    Rotate(psrc_frag, pdst_frag, frag_poses_cnt, pocket_cx, pocket_cy, pocket_cz);

    cout << "Allocating memory..." << endl;

    FRAGMENT* phost_frag[FRAG_CNT];
    for (int i = 0; i < FRAG_CNT; i++)
    {
        phost_frag[i] = new FRAGMENT[frag_poses_cnt[i]];
        cudaMemcpy(phost_frag[i], ptmp_frag[i], frag_poses_cnt[i] * sizeof(FRAGMENT), cudaMemcpyDeviceToHost);
    }

    ATOM* pocket_atom_dev;
    cudaMalloc(&pocket_atom_dev, atom_cnt * sizeof(ATOM));
    cudaMemcpy(pocket_atom_dev, pocket_atom, atom_cnt * sizeof(ATOM), cudaMemcpyHostToDevice);

    GROUP group[4];
    unsigned int * pcombidx[4] = {0};

    BOND* bond_dev;
    cudaMalloc(&bond_dev, (FRAG_CNT-1) * sizeof(BOND));
    cudaMemcpy(bond_dev, bond, (FRAG_CNT - 1) * sizeof(BOND), cudaMemcpyHostToDevice);

    float threshold_factor;
    char buf[20] = {0};
    while (1)
    {
        pcombidx[0] = new unsigned int[frag_poses_cnt[2]];
        pcombidx[1] = new unsigned int[frag_poses_cnt[5]];
        pcombidx[2] = new unsigned int[frag_poses_cnt[7]];
        pcombidx[3] = new unsigned int[frag_poses_cnt[9]];

        if (pcombidx[0] == 0 || pcombidx[1] == 0 || pcombidx[2] == 0 || pcombidx[3] == 0)
        {
            cout << "Mem Allocation Error!" << endl;
            return -1;
        }

        cout << endl;
        cout << "Input threshold factor:";
        scanf("%s", buf);
        if (!strcmp(buf, "q") || !strcmp(buf, "Q")) break;
        if (threshold_factor = (float)atof(buf), (threshold_factor > 1 || threshold_factor <= 0)) break;

        Solve(group, pdst_frag, phost_frag, bond_dev, bond, pocket_atom_dev, pocket_atom, frag_poses_cnt, pcombidx, threshold_factor, atom_cnt);

        cudaError err = cudaGetLastError();;
        switch (err)
        {
        case 0: break;
        case 2:cout << "Mem Allocation Error!" << endl; return -1;
        case 700:cout << "Invalid Access!" << endl; return -1;
        default:cout << "Other Err!" << endl; return -1;
        }

        Sum = 0;
        Sum_ = 0;
        memset(NoPruningCnt, 0, sizeof(int)*FRAG_CNT);
        cout << "Release memory...";
        delete[] pcombidx[0];
        delete[] pcombidx[1];
        delete[] pcombidx[2];
        delete[] pcombidx[3];

        pcombidx[0] = 0;
        pcombidx[1] = 0;
        pcombidx[2] = 0;
        pcombidx[3] = 0;
    }

    return 0;
}
