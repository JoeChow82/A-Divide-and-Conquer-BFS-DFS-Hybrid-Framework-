#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>
#include "struct.h"
#include <corecrt_memcpy_s.h>
#include <stdlib.h>
/*----------------------------------------Get init data-----------------------------------------------*/
//Function:Read PDB file to get original data
//Argument:
//[in ]1.FILE* pfile: file stream to operate the pdb file
//[out]2.char* pbuf: memory buffer to contain file data
//Return value:the number of bytes of data from the file
int CPUReadPDBFile(FILE* pfile, char* pbuf)
{
    if (pfile == NULL)
    {
        std::cout << "文件无法打开!" << std::endl;
        return -1;
    }
    if (pbuf == NULL)
    {
        std::cout << "内存分配失败!" << std::endl;
        return -1;
    }
    unsigned int data_len;
    data_len = fread_s(pbuf, 1024 * 1024, 1, 1024 * 1024, pfile);
    if (0 == data_len)
    {
        std::cout << "文件为空!" << std::endl;
    }
    return data_len;
}

//Function:get data of atoms
//Argument:
//[in ]1.const char* pbuf:the memory buffer of original data 
//[out]2.ATOM* patom:the data of atoms
//[in ]3.unsigned int max_buf_len:maximum size of the memory buffer
int CPUGetAtomsInfo(const char* pbuf, ATOM* patom, unsigned int max_buf_len)
{
    int flag = 0;
    const char* pdref = "DBREF";
    const char* patom_ref = "ATOM";
    const char* phetatm_ref = "HETATM";
    const char* pter_ref = "TER";
    int atom_cnt = 0;
    if (patom == NULL)
    {
        std::cout << "原子坐标缓冲区为空！" << std::endl;
        return -1;
    }
    for (unsigned int idx = 0; idx < max_buf_len; idx++)
    {
        if (0 == flag && pbuf[idx] == 'D')
        {
            int i = 1;
            for (; i <= 4; i++)
            {
                if (pbuf[idx + i] != pdref[i]) break;
            }
            if (i > 4)
            {
                idx += i;
                flag = 1;
            }
        }
        else if (1 == flag)
        {
            if (pbuf[idx] == 'A')
            {
                int i = 1;
                for (; i <= 3; i++)
                {
                    if (pbuf[idx + i] != patom_ref[i]) break;
                }
                if (i > 3)
                {
                    if (pbuf[idx + 10] >= '0' && pbuf[idx + 10] <= '9')
                    {
                        flag = 2;//找到原子数据
                    }
                }
            }
            else if (pbuf[idx] == 'H')
            {
                int i = 1;
                for (; i <= 5; i++)
                {
                    if (pbuf[idx + i] != phetatm_ref[i]) break;
                }
                if (i > 5)
                {
                    if (pbuf[idx + 7] <= '9' && pbuf[idx + 7] >= '0')
                    {
                        flag = 3;//找到残基数据
                    }
                }
            }
        }
        if (2 == flag)
        {
            char tmpbuf[30] = { 0 };
            if (pbuf[idx] == 'T')
            {
                int i = 1;
                for (; i <= 2; i++)
                {
                    if (pbuf[idx + i] != pter_ref[i]) break;
                }
                if (i > 2)
                {
                    idx += 80;
                    flag = 1;//找到多肽链
                }
            }
            else
            {
                if (0 != memcpy_s(tmpbuf, 11, &pbuf[idx + 27], 11))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                patom[atom_cnt].X = (float)atof(tmpbuf);

                if (0 != memcpy_s(tmpbuf, 8, &pbuf[idx + 39], 8))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                patom[atom_cnt].Y = (float)atof(tmpbuf);

                if (0 != memcpy_s(tmpbuf, 8, &pbuf[idx + 46], 8))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                patom[atom_cnt].Z = (float)atof(tmpbuf);

                if (0 != memcpy_s(tmpbuf, 5, &pbuf[idx + 6], 5))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                tmpbuf[5] = 0;
                patom[atom_cnt].num = atoi(tmpbuf);

                if (0 != memcpy_s(patom[atom_cnt].residue_name, 3, &pbuf[idx + 17], 3))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                if (0 != memcpy_s(patom[atom_cnt].name, 2, &pbuf[idx + 76], 2))
                {
                    //std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }

                if (!strcmp(patom[atom_cnt].name, " H"))
                    patom[atom_cnt].van_table_idx = H_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " C"))
                    patom[atom_cnt].van_table_idx = C_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " N"))
                    patom[atom_cnt].van_table_idx = N_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " O"))
                    patom[atom_cnt].van_table_idx = O_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " F"))
                    patom[atom_cnt].van_table_idx = F_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " P"))
                    patom[atom_cnt].van_table_idx = P_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, " S"))
                    patom[atom_cnt].van_table_idx = S_VAN_ATOM_IDX;
                else if (!strcmp(patom[atom_cnt].name, "CL"))
                    patom[atom_cnt].van_table_idx = CL_VAN_ATOM_IDX;
                atom_cnt++;
                idx += 80;
            }
        }
        if (3 == flag)
        {
            char tmpbuf[30] = { 0 };
            if (pbuf[idx] != 'H')   break;
            if (0 != memcpy_s(tmpbuf, 12, &pbuf[idx + 27], 12))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }
            tmpbuf[12] = 0;
            patom[atom_cnt].X = (float)atof(tmpbuf);
            if (0 != memcpy_s(tmpbuf, 12, &pbuf[idx + 39], 12))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }
            tmpbuf[12] = 0;
            patom[atom_cnt].Y = (float)atof(tmpbuf);
            if (0 != memcpy_s(tmpbuf, 8, &pbuf[idx + 46], 8))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }
            tmpbuf[8] = 0;
            patom[atom_cnt].Z = (float)atof(tmpbuf);

            if (0 != memcpy_s(patom[atom_cnt].name, 2, &pbuf[idx + 76], 2))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }

            if (0 != memcpy_s(tmpbuf, 5, &pbuf[idx + 6], 5))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }
            tmpbuf[5] = 0;
            patom[atom_cnt].num = atoi(tmpbuf);
            if (0 != memcpy_s(patom[atom_cnt].residue_name, 3, &pbuf[idx + 17], 3))
            {
                //std::cout << "文件格式错误！" << std::endl;
                return -1;
            }

            if (!strcmp(patom[atom_cnt].name, " H"))
                patom[atom_cnt].van_table_idx = H_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " C"))
                patom[atom_cnt].van_table_idx = C_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " N"))
                patom[atom_cnt].van_table_idx = N_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " O"))
                patom[atom_cnt].van_table_idx = O_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " F"))
                patom[atom_cnt].van_table_idx = F_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " P"))
                patom[atom_cnt].van_table_idx = P_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, " S"))
                patom[atom_cnt].van_table_idx = S_VAN_ATOM_IDX;
            else if (!strcmp(patom[atom_cnt].name, "CL"))
                patom[atom_cnt].van_table_idx = CL_VAN_ATOM_IDX;
            atom_cnt++;
            idx += 80;
        }
    }
    return atom_cnt;
}

//Function:get data of residue
//Argument:
//[in ]1.const char* pbuf:the memory buffer of original data 
//[in ]2.char* residueName:the name of the residue to obtain
//[out]2.RESIDUE& residue:the data of residue
//[in ]3.unsigned int max_buf_len:maximum size of the memory buffer
//Return value:the number of atoms on the residue
int CPUGetResidueInfo(const char* pbuf, char* residueName, RESIDUE& residue, unsigned int max_buf_len)
{
    const char* p = "DBREF";
    const char* phetam = "HETATM";
    int dest_resdue_cnt = 0;
    if (pbuf == NULL)
    {
       // std::cout << "无效数据" << std::endl;
        return -1;
    }
    if (max_buf_len == 0)
    {
       // std::cout << "数据为空" << std::endl;
        return -1;
    }
    if (residueName == NULL)
    {
        //std::cout << "无效关键字" << std::endl;
        return -1;
    }
    char flag = 0;
    for (unsigned int idx = 0; idx < max_buf_len; idx++)
    {
        if (0 == flag && pbuf[idx] == 'D')
        {
            int i = 1;
            for (; i <= 4; i++)
            {
                if (pbuf[idx + i] != p[i]) break;
            }
            if (i > 4)
            {
                idx += i;
                flag = 1;
            }
        }
        else if (1 == flag && pbuf[idx] == 'H')
        {
            int i = 1;
            for (; i <= 5; i++)
            {
                if (pbuf[idx + i] != phetam[i]) break;
            }
            if (i > 5)
            {
                if (pbuf[idx + 7] <= '9' && pbuf[idx + 7] >= '0')
                {
                    flag = 2;//找到残基数据
                }
            }
        }
        else if (2 == flag)
        {
            if (0 == memcmp(&pbuf[idx + 16], residueName, strlen(residueName))) // 是目标残基
            {
                char tmpbuf[15] = { 0 };
                dest_resdue_cnt = 1;
                if (0 != memcpy_s(residue.residueName, strlen(residueName), residueName, strlen(residueName)))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                if (0 != memcpy_s(tmpbuf, 4, &pbuf[idx + 21], 4))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.residueNum = atoi(tmpbuf);
                if (0 != memcpy_s(tmpbuf, 5, &pbuf[idx + 6], 5))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.atomNum[residue.atoms] = atoi(tmpbuf);
                if (0 != memcpy_s(residue.atomName[residue.atoms], 4, &pbuf[idx + 75], 4))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                if (!strcmp(residue.atomName[residue.atoms], " H"))
                    residue.van_idx[residue.atoms] = H_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " C  "))
                    residue.van_idx[residue.atoms] = C_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " N  "))
                    residue.van_idx[residue.atoms] = N_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " O  "))
                    residue.van_idx[residue.atoms] = O_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " F  "))
                    residue.van_idx[residue.atoms] = H_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " P  "))
                    residue.van_idx[residue.atoms] = C_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], " S  "))
                    residue.van_idx[residue.atoms] = N_VAN_ATOM_IDX;
                else if (!strcmp(residue.atomName[residue.atoms], "Cl  "))
                    residue.van_idx[residue.atoms] = O_VAN_ATOM_IDX;
                if (0 != memcpy_s(residue.link, 1, &pbuf[idx + 20], 1))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                if (0 != memcpy_s(tmpbuf, 7, &pbuf[idx + 53], 7))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.occupyRatio[residue.atoms] = (float)atof(tmpbuf);
                if (0 != memcpy_s(tmpbuf, 5, &pbuf[idx + 60], 5))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.temperature[residue.atoms] = (float)atof(tmpbuf);
                if (0 != memcpy_s(tmpbuf, 12, &pbuf[idx + 26], 12))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.atomX[residue.atoms] = (float)atof(tmpbuf);
                if (0 != memcpy_s(tmpbuf, 12, &pbuf[idx + 38], 12))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.atomY[residue.atoms] = (float)atof(tmpbuf);
                if (0 != memcpy_s(tmpbuf, 8, &pbuf[idx + 46], 8))
                {
                   // std::cout << "文件格式错误！" << std::endl;
                    return -1;
                }
                residue.atomZ[residue.atoms] = (float)atof(tmpbuf);
                residue.atoms++;
            }
            else  if (1 == dest_resdue_cnt)
            {
                return residue.atoms;
            }
            idx += 80;
        }
    }
    return residue.atoms;
}

//Function:get the atoms data of pocket 
//Argument:
//  [in ]1.ATOM* psrc_data:original atoms data 
//  [out]2.ATOM* pdst_patom:the data of atoms of pocket
//  [in ]3.RESIDUE* exclude_resdue:the residue of which the atoms are also in the pocket will be ignored
//  [in ]4.unsigned int src_len:orginal data bytes count
//  [in ]5.unsigned int max_atom_cnt:maximum count of atoms
//  [in ]6.float scope_minX:the min X value of geometry scope of the pocket 
//  [in ]7.float scope_maxX:the max X value of geometry scope of the pocket 
//  [in ]8.float scope_minY:the min Y value of geometry scope of the pocket 
//  [in ]9.float scope_maxY:the max Y value of geometry scope of the pocket 
//  [in ]10.float scope_minZ:the min Z value of geometry scope of the pocket 
//  [in ]11.float scope_maxZ:the max Z value of geometry scope of the pocket 
//  [in ]12.int exclude_resdue_cnt:the number of excluded residue 
//Return value:the number of atoms in the pocket 
int CPUGetAtomsFromPocket(ATOM* psrc_data, ATOM* pdst_patom, RESIDUE* exclude_resdue, unsigned int src_len, unsigned int max_atom_cnt,
float scope_minX, float scope_maxX, float scope_minY, float scope_maxY, float scope_minZ, float scope_maxZ, int exclude_resdue_cnt)
{
    int cnt = 0;
    if (psrc_data == NULL || pdst_patom == NULL) return 0;
    if (max_atom_cnt <= 0 || src_len <= 0) return 0;
    if (scope_minX > scope_maxX || scope_minY > scope_maxY || scope_minZ > scope_maxZ) return 0;

    for (unsigned int i = 0; i < src_len; i++)
    {
        if (psrc_data[i].X >= scope_minX && psrc_data[i].X <= scope_maxX
            && psrc_data[i].Y >= scope_minY && psrc_data[i].Y <= scope_maxY
            && psrc_data[i].Z >= scope_minZ && psrc_data[i].Z <= scope_maxZ)
        {
            for (int k = 0; k < exclude_resdue_cnt; k++)
            {
                if (memcmp(psrc_data[i].residue_name, exclude_resdue[k].residueName, 3))
                {
                    memcpy(&pdst_patom[cnt++], &psrc_data[i], sizeof(psrc_data[i]));

                }
            }
        }
    }
    return cnt;
}

//Function: initialize the Vander Waals table index of atoms 
//Arguments:
//  [out] 1.FRAGMENT* pfrag:the data of every fragment
//  [in ] 2.int frag_cnt:the number fragments
//Return value:
void CPUSetAtomVanTableIndex(FRAGMENT* pfrag, int frag_cnt)
{
    for (int i = 0; i < frag_cnt; i++)
    {
        for (int j = 0; j < pfrag[i].atom_cnt; j++)
        {
            if (!memcmp(pfrag[i].atom[j].name, " H", 2))
            {
                pfrag[i].atom[j].van_table_idx = H_VAN_ATOM_IDX;
            }
            else if (!memcmp(pfrag[i].atom[j].name, " C", 2))   pfrag[i].atom[j].van_table_idx = C_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " N", 2))   pfrag[i].atom[j].van_table_idx = N_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " O", 2))   pfrag[i].atom[j].van_table_idx = O_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " F", 2))   pfrag[i].atom[j].van_table_idx = F_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " P", 2))   pfrag[i].atom[j].van_table_idx = P_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " S", 2))   pfrag[i].atom[j].van_table_idx = S_VAN_ATOM_IDX;
            else if (!memcmp(pfrag[i].atom[j].name, " CL", 2))  pfrag[i].atom[j].van_table_idx = CL_VAN_ATOM_IDX;
        }
    }
}

//Function:Initialize every fragment with the data from residue buffer
//Arguments:
//  [out]1.FRAGMENT* fragment:the detination fragment 
//  [in ]2.const RESIDUE& residue:supply the source data
//Return value:
void CPULoadFragment(FRAGMENT* fragment, const RESIDUE& residue)
{
    //片段0：C13侧链末端羟基-羰基段
    {
        fragment[0].index = 0;
        fragment[0].atom_cnt = 4;
        memcpy_s(fragment[0].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[0].atom[0].name, sizeof(residue.atomName[0]), residue.atomName[0], sizeof(residue.atomName[0]));

        fragment[0].atom[0].num = residue.atomNum[0];
        fragment[0].atom[0].X = residue.atomX[0];
        fragment[0].atom[0].Y = residue.atomY[0];
        fragment[0].atom[0].Z = residue.atomZ[0];

        memcpy_s(fragment[0].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[0].atom[1].name, sizeof(residue.atomName[1]), residue.atomName[1], sizeof(residue.atomName[1]));
        fragment[0].atom[1].num = residue.atomNum[1];
        fragment[0].atom[1].X = residue.atomX[1];
        fragment[0].atom[1].Y = residue.atomY[1];
        fragment[0].atom[1].Z = residue.atomZ[1];

        memcpy_s(fragment[0].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[0].atom[2].name, sizeof(residue.atomName[2]), residue.atomName[2], sizeof(residue.atomName[2]));
        fragment[0].atom[2].num = residue.atomNum[2];
        fragment[0].atom[2].X = residue.atomX[2];
        fragment[0].atom[2].Y = residue.atomY[2];
        fragment[0].atom[2].Z = residue.atomZ[2];

        memcpy_s(fragment[0].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[0].atom[3].name, sizeof(residue.atomName[3]), residue.atomName[3], sizeof(residue.atomName[3]));
        fragment[0].atom[3].num = residue.atomNum[3];
        fragment[0].atom[3].X = residue.atomX[3];
        fragment[0].atom[3].Y = residue.atomY[3];
        fragment[0].atom[3].Z = residue.atomZ[3];
    }
    //片段1：紫杉烷核心六元环段
    {
        fragment[1].index = 0;
        fragment[1].atom_cnt = 8;

        memcpy_s(fragment[1].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[0].name, sizeof(residue.atomName[4]), residue.atomName[4], sizeof(residue.atomName[4]));
        fragment[1].atom[0].num = residue.atomNum[4];
        fragment[1].atom[0].X = residue.atomX[4];
        fragment[1].atom[0].Y = residue.atomY[4];
        fragment[1].atom[0].Z = residue.atomZ[4];

        memcpy_s(fragment[1].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[1].name, sizeof(residue.atomName[5]), residue.atomName[5], sizeof(residue.atomName[5]));
        fragment[1].atom[1].num = residue.atomNum[5];
        fragment[1].atom[1].X = residue.atomX[5];
        fragment[1].atom[1].Y = residue.atomY[5];
        fragment[1].atom[1].Z = residue.atomZ[5];

        memcpy_s(fragment[1].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[2].name, sizeof(residue.atomName[6]), residue.atomName[6], sizeof(residue.atomName[6]));
        fragment[1].atom[2].num = residue.atomNum[6];
        fragment[1].atom[2].X = residue.atomX[6];
        fragment[1].atom[2].Y = residue.atomY[6];
        fragment[1].atom[2].Z = residue.atomZ[6];

        memcpy_s(fragment[1].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[3].name, sizeof(residue.atomName[7]), residue.atomName[7], sizeof(residue.atomName[7]));
        fragment[1].atom[3].num = residue.atomNum[7];
        fragment[1].atom[3].X = residue.atomX[7];
        fragment[1].atom[3].Y = residue.atomY[7];
        fragment[1].atom[3].Z = residue.atomZ[7];

        memcpy_s(fragment[1].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[4].name, sizeof(residue.atomName[8]), residue.atomName[8], sizeof(residue.atomName[8]));
        fragment[1].atom[4].num = residue.atomNum[8];
        fragment[1].atom[4].X = residue.atomX[8];
        fragment[1].atom[4].Y = residue.atomY[8];
        fragment[1].atom[4].Z = residue.atomZ[8];

        memcpy_s(fragment[1].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[5].name, sizeof(residue.atomName[9]), residue.atomName[9], sizeof(residue.atomName[9]));
        fragment[1].atom[5].num = residue.atomNum[9];
        fragment[1].atom[5].X = residue.atomX[9];
        fragment[1].atom[5].Y = residue.atomY[9];
        fragment[1].atom[5].Z = residue.atomZ[9];

        memcpy_s(fragment[1].atom[6].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[6].name, sizeof(residue.atomName[10]), residue.atomName[10], sizeof(residue.atomName[10]));
        fragment[1].atom[6].num = residue.atomNum[10];
        fragment[1].atom[6].X = residue.atomX[10];
        fragment[1].atom[6].Y = residue.atomY[10];
        fragment[1].atom[6].Z = residue.atomZ[10];

        memcpy_s(fragment[1].atom[7].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[1].atom[7].name, sizeof(residue.atomName[11]), residue.atomName[11], sizeof(residue.atomName[11]));
        fragment[1].atom[7].num = residue.atomNum[11];
        fragment[1].atom[7].X = residue.atomX[11];
        fragment[1].atom[7].Y = residue.atomY[11];
        fragment[1].atom[7].Z = residue.atomZ[11];
    }

    //片段2：C4乙酰基连接段
    {
        fragment[2].index = 0;
        fragment[2].atom_cnt = 6;

        memcpy_s(fragment[2].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[0].name, sizeof(residue.atomName[12]), residue.atomName[12], sizeof(residue.atomName[12]));
        fragment[2].atom[0].num = residue.atomNum[12];
        fragment[2].atom[0].X = residue.atomX[12];
        fragment[2].atom[0].Y = residue.atomY[12];
        fragment[2].atom[0].Z = residue.atomZ[12];

        memcpy_s(fragment[2].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[1].name, sizeof(residue.atomName[13]), residue.atomName[13], sizeof(residue.atomName[13]));
        fragment[2].atom[1].num = residue.atomNum[13];
        fragment[2].atom[1].X = residue.atomX[13];
        fragment[2].atom[1].Y = residue.atomY[13];
        fragment[2].atom[1].Z = residue.atomZ[13];

        memcpy_s(fragment[2].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[2].name, sizeof(residue.atomName[14]), residue.atomName[14], sizeof(residue.atomName[14]));
        fragment[2].atom[2].num = residue.atomNum[14];
        fragment[2].atom[2].X = residue.atomX[14];
        fragment[2].atom[2].Y = residue.atomY[14];
        fragment[2].atom[2].Z = residue.atomZ[14];

        memcpy_s(fragment[2].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[3].name, sizeof(residue.atomName[15]), residue.atomName[15], sizeof(residue.atomName[15]));
        fragment[2].atom[3].num = residue.atomNum[15];
        fragment[2].atom[3].X = residue.atomX[15];
        fragment[2].atom[3].Y = residue.atomY[15];
        fragment[2].atom[3].Z = residue.atomZ[15];

        memcpy_s(fragment[2].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[4].name, sizeof(residue.atomName[16]), residue.atomName[16], sizeof(residue.atomName[16]));
        fragment[2].atom[4].num = residue.atomNum[16];
        fragment[2].atom[4].X = residue.atomX[16];
        fragment[2].atom[4].Y = residue.atomY[16];
        fragment[2].atom[4].Z = residue.atomZ[16];

        memcpy_s(fragment[2].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[2].atom[5].name, sizeof(residue.atomName[17]), residue.atomName[17], sizeof(residue.atomName[17]));
        fragment[2].atom[5].num = residue.atomNum[17];
        fragment[2].atom[5].X = residue.atomX[17];
        fragment[2].atom[5].Y = residue.atomY[17];
        fragment[2].atom[5].Z = residue.atomZ[17];
    }
    //片段3：C13侧链羟基段
    {
        fragment[3].index = 0;
        fragment[3].atom_cnt = 3;

        memcpy_s(fragment[3].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[3].atom[0].name, sizeof(residue.atomName[18]), residue.atomName[18], sizeof(residue.atomName[18]));
        fragment[3].atom[0].num = residue.atomNum[18];
        fragment[3].atom[0].X = residue.atomX[18];
        fragment[3].atom[0].Y = residue.atomY[18];
        fragment[3].atom[0].Z = residue.atomZ[18];

        memcpy_s(fragment[3].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[3].atom[1].name, sizeof(residue.atomName[19]), residue.atomName[19], sizeof(residue.atomName[19]));
        fragment[3].atom[1].num = residue.atomNum[19];
        fragment[3].atom[1].X = residue.atomX[19];
        fragment[3].atom[1].Y = residue.atomY[19];
        fragment[3].atom[1].Z = residue.atomZ[19];

        memcpy_s(fragment[3].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[3].atom[2].name, sizeof(residue.atomName[20]), residue.atomName[20], sizeof(residue.atomName[20]));
        fragment[3].atom[2].num = residue.atomNum[20];
        fragment[3].atom[2].X = residue.atomX[20];
        fragment[3].atom[2].Y = residue.atomY[20];
        fragment[3].atom[2].Z = residue.atomZ[20];
    }
    //片段4：C13侧链五元环段	
    {
        fragment[4].index = 0;
        fragment[4].atom_cnt = 5;

        memcpy_s(fragment[4].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[4].atom[0].name, sizeof(residue.atomName[21]), residue.atomName[21], sizeof(residue.atomName[21]));
        fragment[4].atom[0].num = residue.atomNum[21];
        fragment[4].atom[0].X = residue.atomX[21];
        fragment[4].atom[0].Y = residue.atomY[21];
        fragment[4].atom[0].Z = residue.atomZ[21];

        memcpy_s(fragment[4].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[4].atom[1].name, sizeof(residue.atomName[22]), residue.atomName[22], sizeof(residue.atomName[22]));
        fragment[4].atom[1].num = residue.atomNum[22];
        fragment[4].atom[1].X = residue.atomX[22];
        fragment[4].atom[1].Y = residue.atomY[22];
        fragment[4].atom[1].Z = residue.atomZ[22];

        memcpy_s(fragment[4].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[4].atom[2].name, sizeof(residue.atomName[23]), residue.atomName[23], sizeof(residue.atomName[23]));
        fragment[4].atom[2].num = residue.atomNum[23];
        fragment[4].atom[2].X = residue.atomX[23];
        fragment[4].atom[2].Y = residue.atomY[23];
        fragment[4].atom[2].Z = residue.atomZ[23];

        memcpy_s(fragment[4].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[4].atom[3].name, sizeof(residue.atomName[24]), residue.atomName[24], sizeof(residue.atomName[24]));
        fragment[4].atom[3].num = residue.atomNum[24];
        fragment[4].atom[3].X = residue.atomX[24];
        fragment[4].atom[3].Y = residue.atomY[24];
        fragment[4].atom[3].Z = residue.atomZ[24];

        memcpy_s(fragment[4].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[4].atom[4].name, sizeof(residue.atomName[25]), residue.atomName[25], sizeof(residue.atomName[25]));
        fragment[4].atom[4].num = residue.atomNum[25];
        fragment[4].atom[4].X = residue.atomX[25];
        fragment[4].atom[4].Y = residue.atomY[25];
        fragment[4].atom[4].Z = residue.atomZ[25];
    }
    //片段5：C2苯甲酰基连接段		
    {
        fragment[5].index = 0;  //初始姿态序号
        fragment[5].atom_cnt = 7;
        
        memcpy_s(fragment[5].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[0].name, sizeof(residue.atomName[26]), residue.atomName[26], sizeof(residue.atomName[26]));
        fragment[5].atom[0].num = residue.atomNum[26];
        fragment[5].atom[0].X = residue.atomX[26];
        fragment[5].atom[0].Y = residue.atomY[26];
        fragment[5].atom[0].Z = residue.atomZ[26];

        memcpy_s(fragment[5].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[1].name, sizeof(residue.atomName[27]), residue.atomName[27], sizeof(residue.atomName[27]));
        fragment[5].atom[1].num = residue.atomNum[27];
        fragment[5].atom[1].X = residue.atomX[27];
        fragment[5].atom[1].Y = residue.atomY[27];
        fragment[5].atom[1].Z = residue.atomZ[27];

        memcpy_s(fragment[5].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[2].name, sizeof(residue.atomName[28]), residue.atomName[28], sizeof(residue.atomName[28]));
        fragment[5].atom[2].num = residue.atomNum[28];
        fragment[5].atom[2].X = residue.atomX[28];
        fragment[5].atom[2].Y = residue.atomY[28];
        fragment[5].atom[2].Z = residue.atomZ[28];

        memcpy_s(fragment[5].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[3].name, sizeof(residue.atomName[29]), residue.atomName[29], sizeof(residue.atomName[29]));
        fragment[5].atom[3].num = residue.atomNum[29];
        fragment[5].atom[3].X = residue.atomX[29];
        fragment[5].atom[3].Y = residue.atomY[29];
        fragment[5].atom[3].Z = residue.atomZ[29];

        memcpy_s(fragment[5].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[4].name, sizeof(residue.atomName[30]), residue.atomName[30], sizeof(residue.atomName[30]));
        fragment[5].atom[4].num = residue.atomNum[30];
        fragment[5].atom[4].X = residue.atomX[30];
        fragment[5].atom[4].Y = residue.atomY[30];
        fragment[5].atom[4].Z = residue.atomZ[30];

        memcpy_s(fragment[5].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[5].name, sizeof(residue.atomName[31]), residue.atomName[31], sizeof(residue.atomName[31]));
        fragment[5].atom[5].num = residue.atomNum[31];
        fragment[5].atom[5].X = residue.atomX[31];
        fragment[5].atom[5].Y = residue.atomY[31];
        fragment[5].atom[5].Z = residue.atomZ[31];

        memcpy_s(fragment[5].atom[6].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[5].atom[6].name, sizeof(residue.atomName[32]), residue.atomName[32], sizeof(residue.atomName[32]));
        fragment[5].atom[6].num = residue.atomNum[32];
        fragment[5].atom[6].X = residue.atomX[32];
        fragment[5].atom[6].Y = residue.atomY[32];
        fragment[5].atom[6].Z = residue.atomZ[32];
    }
    //片段6：C1羟基侧链段
    {
        fragment[6].index = 0;
        fragment[6].atom_cnt = 9;

        memcpy_s(fragment[6].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[0].name, sizeof(residue.atomName[33]), residue.atomName[33], sizeof(residue.atomName[33]));
        fragment[6].atom[0].num = residue.atomNum[33];
        fragment[6].atom[0].X = residue.atomX[33];
        fragment[6].atom[0].Y = residue.atomY[33];
        fragment[6].atom[0].Z = residue.atomZ[33];

        memcpy_s(fragment[6].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[1].name, sizeof(residue.atomName[34]), residue.atomName[34], sizeof(residue.atomName[34]));
        fragment[6].atom[1].num = residue.atomNum[34];
        fragment[6].atom[1].X = residue.atomX[34];
        fragment[6].atom[1].Y = residue.atomY[34];
        fragment[6].atom[1].Z = residue.atomZ[34];

        memcpy_s(fragment[6].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[2].name, sizeof(residue.atomName[35]), residue.atomName[35], sizeof(residue.atomName[35]));
        fragment[6].atom[2].num = residue.atomNum[35];
        fragment[6].atom[2].X = residue.atomX[35];
        fragment[6].atom[2].Y = residue.atomY[35];
        fragment[6].atom[2].Z = residue.atomZ[35];

        memcpy_s(fragment[6].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[3].name, sizeof(residue.atomName[36]), residue.atomName[36], sizeof(residue.atomName[36]));
        fragment[6].atom[3].num = residue.atomNum[36];
        fragment[6].atom[3].X = residue.atomX[36];
        fragment[6].atom[3].Y = residue.atomY[36];
        fragment[6].atom[3].Z = residue.atomZ[36];

        memcpy_s(fragment[6].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[4].name, sizeof(residue.atomName[37]), residue.atomName[37], sizeof(residue.atomName[37]));
        fragment[6].atom[4].num = residue.atomNum[37];
        fragment[6].atom[4].X = residue.atomX[37];
        fragment[6].atom[4].Y = residue.atomY[37];
        fragment[6].atom[4].Z = residue.atomZ[37];

        memcpy_s(fragment[6].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[5].name, sizeof(residue.atomName[38]), residue.atomName[38], sizeof(residue.atomName[38]));
        fragment[6].atom[5].num = residue.atomNum[38];
        fragment[6].atom[5].X = residue.atomX[38];
        fragment[6].atom[5].Y = residue.atomY[38];
        fragment[6].atom[5].Z = residue.atomZ[38];

        memcpy_s(fragment[6].atom[6].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[6].name, sizeof(residue.atomName[39]), residue.atomName[39], sizeof(residue.atomName[39]));
        fragment[6].atom[6].num = residue.atomNum[39];
        fragment[6].atom[6].X = residue.atomX[39];
        fragment[6].atom[6].Y = residue.atomY[39];
        fragment[6].atom[6].Z = residue.atomZ[39];

        memcpy_s(fragment[6].atom[7].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[7].name, sizeof(residue.atomName[40]), residue.atomName[40], sizeof(residue.atomName[40]));
        fragment[6].atom[7].num = residue.atomNum[40];
        fragment[6].atom[7].X = residue.atomX[40];
        fragment[6].atom[7].Y = residue.atomY[40];
        fragment[6].atom[7].Z = residue.atomZ[40];

        memcpy_s(fragment[6].atom[8].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[6].atom[8].name, sizeof(residue.atomName[41]), residue.atomName[41], sizeof(residue.atomName[41]));
        fragment[6].atom[8].num = residue.atomNum[41];
        fragment[6].atom[8].X = residue.atomX[41];
        fragment[6].atom[8].Y = residue.atomY[41];
        fragment[6].atom[8].Z = residue.atomZ[41];
    }

    //片段7：酰胺+末端苯环段
    {
        fragment[7].index = 0;
        fragment[7].atom_cnt = 9;

        memcpy_s(fragment[7].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[0].name, sizeof(residue.atomName[42]), residue.atomName[42], sizeof(residue.atomName[42]));
        fragment[7].atom[0].num = residue.atomNum[42];
        fragment[7].atom[0].X = residue.atomX[42];
        fragment[7].atom[0].Y = residue.atomY[42];
        fragment[7].atom[0].Z = residue.atomZ[42];

        memcpy_s(fragment[7].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[1].name, sizeof(residue.atomName[43]), residue.atomName[43], sizeof(residue.atomName[43]));
        fragment[7].atom[1].num = residue.atomNum[43];
        fragment[7].atom[1].X = residue.atomX[43];
        fragment[7].atom[1].Y = residue.atomY[43];
        fragment[7].atom[1].Z = residue.atomZ[43];

        memcpy_s(fragment[7].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[2].name, sizeof(residue.atomName[44]), residue.atomName[44], sizeof(residue.atomName[44]));
        fragment[7].atom[2].num = residue.atomNum[44];
        fragment[7].atom[2].X = residue.atomX[44];
        fragment[7].atom[2].Y = residue.atomY[44];
        fragment[7].atom[2].Z = residue.atomZ[44];

        memcpy_s(fragment[7].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[3].name, sizeof(residue.atomName[45]), residue.atomName[45], sizeof(residue.atomName[45]));
        fragment[7].atom[3].num = residue.atomNum[45];
        fragment[7].atom[3].X = residue.atomX[45];
        fragment[7].atom[3].Y = residue.atomY[45];
        fragment[7].atom[3].Z = residue.atomZ[45];

        memcpy_s(fragment[7].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[4].name, sizeof(residue.atomName[46]), residue.atomName[46], sizeof(residue.atomName[46]));
        fragment[7].atom[4].num = residue.atomNum[46];
        fragment[7].atom[4].X = residue.atomX[46];
        fragment[7].atom[4].Y = residue.atomY[46];
        fragment[7].atom[4].Z = residue.atomZ[46];

        memcpy_s(fragment[7].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[5].name, sizeof(residue.atomName[47]), residue.atomName[47], sizeof(residue.atomName[47]));
        fragment[7].atom[5].num = residue.atomNum[47];
        fragment[7].atom[5].X = residue.atomX[47];
        fragment[7].atom[5].Y = residue.atomY[47];
        fragment[7].atom[5].Z = residue.atomZ[47];

        memcpy_s(fragment[7].atom[6].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[6].name, sizeof(residue.atomName[48]), residue.atomName[48], sizeof(residue.atomName[48]));
        fragment[7].atom[6].num = residue.atomNum[48];
        fragment[7].atom[6].X = residue.atomX[48];
        fragment[7].atom[6].Y = residue.atomY[48];
        fragment[7].atom[6].Z = residue.atomZ[48];

        memcpy_s(fragment[7].atom[7].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[7].name, sizeof(residue.atomName[49]), residue.atomName[49], sizeof(residue.atomName[49]));
        fragment[7].atom[7].num = residue.atomNum[49];
        fragment[7].atom[7].X = residue.atomX[49];
        fragment[7].atom[7].Y = residue.atomY[49];
        fragment[7].atom[7].Z = residue.atomZ[49];

        memcpy_s(fragment[7].atom[8].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[7].atom[8].name, sizeof(residue.atomName[50]), residue.atomName[50], sizeof(residue.atomName[50]));
        fragment[7].atom[8].num = residue.atomNum[50];
        fragment[7].atom[8].X = residue.atomX[50];
        fragment[7].atom[8].Y = residue.atomY[50];
        fragment[7].atom[8].Z = residue.atomZ[50];
    }
    //片段8：苯环侧链段
    {
        fragment[8].index = 0;
        fragment[8].atom_cnt = 6;

        memcpy_s(fragment[8].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[0].name, sizeof(residue.atomName[51]), residue.atomName[51], sizeof(residue.atomName[51]));
        fragment[8].atom[0].num = residue.atomNum[51];
        fragment[8].atom[0].X = residue.atomX[51];
        fragment[8].atom[0].Y = residue.atomY[51];
        fragment[8].atom[0].Z = residue.atomZ[51];

        memcpy_s(fragment[8].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[1].name, sizeof(residue.atomName[52]), residue.atomName[52], sizeof(residue.atomName[52]));
        fragment[8].atom[1].num = residue.atomNum[52];
        fragment[8].atom[1].X = residue.atomX[52];
        fragment[8].atom[1].Y = residue.atomY[52];
        fragment[8].atom[1].Z = residue.atomZ[52];

        memcpy_s(fragment[8].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[2].name, sizeof(residue.atomName[53]), residue.atomName[53], sizeof(residue.atomName[53]));
        fragment[8].atom[2].num = residue.atomNum[53];
        fragment[8].atom[2].X = residue.atomX[53];
        fragment[8].atom[2].Y = residue.atomY[53];
        fragment[8].atom[2].Z = residue.atomZ[53];

        memcpy_s(fragment[8].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[3].name, sizeof(residue.atomName[54]), residue.atomName[54], sizeof(residue.atomName[54]));
        fragment[8].atom[3].num = residue.atomNum[54];
        fragment[8].atom[3].X = residue.atomX[54];
        fragment[8].atom[3].Y = residue.atomY[54];
        fragment[8].atom[3].Z = residue.atomZ[54];

        memcpy_s(fragment[8].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[4].name, sizeof(residue.atomName[55]), residue.atomName[55], sizeof(residue.atomName[55]));
        fragment[8].atom[4].num = residue.atomNum[55];
        fragment[8].atom[4].X = residue.atomX[55];
        fragment[8].atom[4].Y = residue.atomY[55];
        fragment[8].atom[4].Z = residue.atomZ[55];

        memcpy_s(fragment[8].atom[5].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[8].atom[5].name, sizeof(residue.atomName[56]), residue.atomName[56], sizeof(residue.atomName[56]));
        fragment[8].atom[5].num = residue.atomNum[56];
        fragment[8].atom[5].X = residue.atomX[56];
        fragment[8].atom[5].Y = residue.atomY[56];
        fragment[8].atom[5].Z = residue.atomZ[56];
    }
    //片段9：C13侧链末端段
    {
        fragment[9].index = 0;
        fragment[9].atom_cnt = 5;

        memcpy_s(fragment[9].atom[0].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[9].atom[0].name, sizeof(residue.atomName[57]), residue.atomName[57], sizeof(residue.atomName[57]));
        fragment[9].atom[0].num = residue.atomNum[57];
        fragment[9].atom[0].X = residue.atomX[57];
        fragment[9].atom[0].Y = residue.atomY[57];
        fragment[9].atom[0].Z = residue.atomZ[57];

        memcpy_s(fragment[9].atom[1].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[9].atom[1].name, sizeof(residue.atomName[58]), residue.atomName[58], sizeof(residue.atomName[58]));
        fragment[9].atom[1].num = residue.atomNum[58];
        fragment[9].atom[1].X = residue.atomX[58];
        fragment[9].atom[1].Y = residue.atomY[58];
        fragment[9].atom[1].Z = residue.atomZ[58];

        memcpy_s(fragment[9].atom[2].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[9].atom[2].name, sizeof(residue.atomName[59]), residue.atomName[59], sizeof(residue.atomName[59]));
        fragment[9].atom[2].num = residue.atomNum[59];
        fragment[9].atom[2].X = residue.atomX[59];
        fragment[9].atom[2].Y = residue.atomY[59];
        fragment[9].atom[2].Z = residue.atomZ[59];

        memcpy_s(fragment[9].atom[3].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[9].atom[3].name, sizeof(residue.atomName[60]), residue.atomName[60], sizeof(residue.atomName[60]));
        fragment[9].atom[3].num = residue.atomNum[60];
        fragment[9].atom[3].X = residue.atomX[60];
        fragment[9].atom[3].Y = residue.atomY[60];
        fragment[9].atom[3].Z = residue.atomZ[60];

        memcpy_s(fragment[9].atom[4].residue_name, sizeof(residue.residueName), residue.residueName, sizeof(residue.residueName));
        memcpy_s(fragment[9].atom[4].name, sizeof(residue.atomName[61]), residue.atomName[61], sizeof(residue.atomName[61]));
        fragment[9].atom[4].num = residue.atomNum[61];
        fragment[9].atom[4].X = residue.atomX[61];
        fragment[9].atom[4].Y = residue.atomY[61];
        fragment[9].atom[4].Z = residue.atomZ[61];
    }
}

//Function: initialize the geometry information of the pocket
//Arguments:
//  [in ]1.FRAGMENT* pfrag:fragments data
//  [in ]2.int frag_cnt: number of fragments
//  [out]3.float& cx: the coordinate of X of the pocket center
//  [out]4.float& cy: the coordinate of Y of the pocket center
//  [out]5.float& cz: the coordinate of Z of the pocket center
//  [out]6.float& scope_minX: the min X coordinate value of the pocket scope
//  [out]7.float& scope_maxX: the max X coordinate value of the pocket scope
//  [out]8.float& scope_minY: the min Y coordinate value of the pocket scope
//  [out]9.float& scope_maxY: the max Y coordinate value of the pocket scope
//  [out]10.float& scope_minZ: the min Z coordinate value of the pocket scope
//  [out]11.float& scope_maxZ: the max Z coordinate value of the pocket scope
//Return value:true--the function finished normally ;false--the function failed
bool CPUGetPocketGeoInfo(FRAGMENT* pfrag, int frag_cnt, float& cx, float& cy, float& cz, float& scope_minX, float& scope_maxX, float& scope_minY, float& scope_maxY, float& scope_minZ, float& scope_maxZ)
{
    if (pfrag == NULL) return false;
    if (frag_cnt == 0) return false;

    int atom_cnt = 0;
    cx = 0.0;
    cy = 0.0;
    cz = 0.0;
    for (int i = 0; i < frag_cnt; i++)
    {
        atom_cnt += pfrag[i].atom_cnt;
        for (int j = 0; j < pfrag[i].atom_cnt; j++)
        {
            cx = cx + (float)pfrag[i].atom[j].X;
            cy = cy + (float)pfrag[i].atom[j].Y;
            cz = cz + (float)pfrag[i].atom[j].Z;
        }
    }
    cx /= atom_cnt;
    cy /= atom_cnt;
    cz /= atom_cnt;

    scope_minX = cx - 15;
    scope_maxX = cx + 15;
    scope_minY = cy - 15;
    scope_maxY = cy + 15;
    scope_minZ = cz - 15;
    scope_maxZ = cz + 15;
    return true;
}

//Function:convert the PDB coordinates to pocket coordinates
//Arguments:
//[in]1.FRAGMENT* fragment:framents data
//[in]2.ATOM* pocket_atom:the atoms data of the pocket 
//[in]3.int frag_cnt: the count of the fragments
//[out]4.float cx: the X coordinate of the pocket center point 
//[out]5.float cy: the Y coordinate of the pocket center point 
//[out]6.float cz: the Z coordinate of the pocket center point 
//Return value:
void PDBCor2PocketCor(FRAGMENT* fragment, ATOM* pocket_atom, int frag_cnt, int atom_cnt, float cx, float cy, float cz)
{
    for (int i = 0; i < frag_cnt; i++)
    {
        for (int j = 0; j < fragment[i].atom_cnt; j++)
        {
            fragment[i].atom[j].X -= cx;
            fragment[i].atom[j].Y -= cy;
            fragment[i].atom[j].Z -= cz;

        }
    }
    for (int i = 0; i < atom_cnt; i++)
    {
        pocket_atom[i].X -= cx;
        pocket_atom[i].Y -= cy;
        pocket_atom[i].Z -= cx;
    }
}

//Function:Build bond information
//Arguments:
//  [out]1.BOND* bond:memory buffer containing bond information
//Return value:
void LoadBond(BOND* bond)
{
    //frag0-frag1
    bond[0].frag_idx[0] = 0;
    bond[0].frag_idx[1] = 1;
    bond[0].atom_idx[0] = 3;
    bond[0].atom_idx[1] = 0;

    //frag1-frag2
    bond[1].frag_idx[0] = 1;
    bond[1].frag_idx[1] = 2;
    bond[1].atom_idx[0] = 7;
    bond[1].atom_idx[1] = 0;

    //frag2-frag3
    bond[2].frag_idx[0] = 2;
    bond[2].frag_idx[1] = 3;
    bond[2].atom_idx[0] = 5;
    bond[2].atom_idx[1] = 6;

    //frag3-frag4
    bond[3].frag_idx[0] = 3;
    bond[3].frag_idx[1] = 4;
    bond[3].atom_idx[0] = 2;
    bond[3].atom_idx[1] = 0;

    //frag4-frag5
    bond[4].frag_idx[0] = 4;
    bond[4].frag_idx[1] = 5;
    bond[4].atom_idx[0] = 4;
    bond[4].atom_idx[1] = 0;

    //frag5-frag6
    bond[5].frag_idx[0] = 5;
    bond[5].frag_idx[1] = 6;
    bond[5].atom_idx[0] = 6;
    bond[5].atom_idx[1] = 0;

    //frag6-frag7
    bond[6].frag_idx[0] = 6;
    bond[6].frag_idx[1] = 7;
    bond[6].atom_idx[0] = 8;
    bond[6].atom_idx[1] = 0;

    //frag7-frag8
    bond[7].frag_idx[0] = 7;
    bond[7].frag_idx[1] = 8;
    bond[7].atom_idx[0] = 8;
    bond[7].atom_idx[1] = 0;

    //frag8-frag9
    bond[8].frag_idx[0] = 7;
    bond[8].frag_idx[1] = 9;
    bond[8].atom_idx[0] = 5;
    bond[8].atom_idx[1] = 0;

}

//Function:global initialization
//Arguments:
//  [in ]1.FILE** pfile:file stream intended for obtaining the source data
//  [out]2.char* pbuf:memory buffer used to contain the original data from file
//  [out]3.ATOM* patom:memory buffer used to contain the atoms data
//  [out]4.ATOM* pocket_atom: memory buffer used to contain the pocket atoms data
//  [out]5.RESIDUE& residue:memory buffer used to contain the residue data
//  [out]6.FRAGMENT* fragment:memory buffer used to contain the fragments data
//  [out]7.BOND* bond:memory buffer used to contain the bonds information
//  [out]8.float& pocket_cx:recording the X coordinate of pocket center point 
//  [out]9.float& pocket_cy:recording the Y coordinate of pocket center point 
//  [out]10.float& pocket_cz:recording the Z coordinate of pocket center point 
//  [out]11.float& scope_minX:recording the min X coordinate of the pocket geometric scope 
//  [out]12.float& scope_maxX:recording the max X coordinate of the pocket geometric scope 
//  [out]13.float& scope_minY:recording the min Y coordinate of the pocket geometric scope  
//  [out]14.float& scope_maxY:recording the max Y coordinate of the pocket geometric scope 
//  [out]15.float& scope_minZ:recording the min Z coordinate of the pocket geometric scope 
//  [out]16.float& scope_maxZ:recording the max Z coordinate of the pocket geometric scope 
//  [out]17.int& atom_cnt:recording the number of pocket atoms  
//  [out]18.unsigned int* frag_pose_cnt:recording the number of poses per fragment 
//Return value:
void Init(FILE** pfile, char* pbuf, ATOM* patom, ATOM* pocket_atom, RESIDUE& residue, FRAGMENT* fragment, BOND* bond, float& pocket_cx, float& pocket_cy, float& pocket_cz,
    float& scope_minX, float& scope_maxX, float& scope_minY, float& scope_maxY, float& scope_minZ, float& scope_maxZ, int& atom_cnt, unsigned int* frag_pose_cnt)
{
    //获取原始pdb数据,填充数据结构
    //读取文件数据
    fopen_s(pfile, "D:\\1JFF.pdb", "r"); //Open file
    CPUReadPDBFile(*pfile, pbuf);        //Read file
    fclose(*pfile);                      //Close file
    
    CPUGetResidueInfo(pbuf, "TA1", residue, (unsigned int)4096 * 4096);//Get RAP residue information
    CPULoadFragment(fragment, residue);//Manually load fragment
    CPUSetAtomVanTableIndex(fragment, FRAG_CNT);//Get the Vander-Waals-table-indexes of every atom 
    atom_cnt = CPUGetAtomsInfo(pbuf, patom, (unsigned int)4096 * 4096);//Get all atoms information
    CPUGetPocketGeoInfo(fragment, FRAG_CNT, pocket_cx, pocket_cy, pocket_cz, scope_minX, scope_maxX, scope_minY, scope_maxY, scope_minZ, scope_maxZ);//Get the geometric data of protein pocket 
    atom_cnt = CPUGetAtomsFromPocket(patom, pocket_atom, &residue, atom_cnt, atom_cnt, scope_minX, scope_maxX, scope_minY, scope_maxY, scope_minZ, scope_maxZ, 1);//Get the pocket atoms data
    PDBCor2PocketCor(fragment, pocket_atom, FRAG_CNT, atom_cnt, pocket_cx, pocket_cy, pocket_cz);//Convert thr pdb coordinates to pocket coordinates
    LoadBond(bond);//Load bonds information
    //Cascade poses data
    for (int i = 0; i < FRAG_CNT; i++)
    {
        frag_pose_cnt[i] = (unsigned int)pow(EULER_STEP_CNT,3);
    }
    for (int i = 1; i < FRAG_CNT; i++)
    {
        for (int j = i; j < FRAG_CNT; j++)
        {
            frag_pose_cnt[j] *= CONF_STEP_CNT;
        }
    }
}
/*---------------------------------------------------------------------------------------------------*/
