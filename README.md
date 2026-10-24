**Case Paclitaxel:**
# A Divide-and-Conquer-BFS-DFS Hybrid Framework with Precomputed Conflict Tables for Deterministic Combinatorial Search: A Molecular Docking Case Study
A BFS-DFS hybrid search architecture that decouples conflict evaluation from search
Precompute all pairwise conflicts once, store them in lookup tables, and reduce every conflict check during search to an O(1) table lookup for combinatorial optimization proplems.

1.Supportong File:
  
    a.1JFF.pdb is the source data of Paclitaxel
  
    b.Paclitaxel.xlsx is the experiment related data covering groups and rigid fragments partition,experimental statics and figures.

2.Software Environment:
  
    a.CUDA: Cuda compilation tools, release 13.2, V13.2.78
  
    b.IDE: Visual Studio 2026; version 18.10.3, Community
  
    c.Operating System: Windows11, 25H2
  
    d.Programing Language: C/C++, C++: ISO C++20/ C: old MSVC

3.note:
  
    a.The project runs normally under the software environment above.
  
    b.The path of pdb file reading is D:\1JFF.pdb.
  
    c.The file 1JFF.pdb is from https://www.rcsb.org/structure/1JFF
