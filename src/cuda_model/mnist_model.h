extern void cu_madd(float* A, float* B, float* C, int M, int N);
extern void cu_mmelem(float* A, float* B, float* C, int M, int N);
extern void cu_mmreduce(float* A, float* B, int M, int N);
extern void cu_mm(float* A, float* B, float* C, int N_a, int M_a, int M_b);
