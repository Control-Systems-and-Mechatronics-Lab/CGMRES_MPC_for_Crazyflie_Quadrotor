#ifndef CGMRES_H
# define CGMRES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "types.h"
#include "model.h"
#include "auxil.h"

void printMatrix(const Matrix m);

enum tiny_ErrorCode tiny_computeAcBc(const Matrix* A, const Matrix* B, Matrix* Ac, Matrix* Bc, float DT);

enum tiny_ErrorCode tiny_ComputePhix(tiny_AdmmWorkspace* work, Matrix phix);

enum tiny_ErrorCode tiny_ComputeHx(tiny_AdmmWorkspace* work, int k, Matrix hx);

enum tiny_ErrorCode tiny_ComputeHu(tiny_AdmmWorkspace* work, int k, Matrix hu);

enum tiny_ErrorCode tiny_ComputeOptimalityError(tiny_AdmmWorkspace* work, const Matrix state_vec, const Matrix current_solution_vec, Matrix optimality_vec);

enum tiny_ErrorCode tiny_AxFunc(tiny_AdmmWorkspace* work, Matrix current_solution_vec, Matrix current_solution_vec_ax, Matrix direction_vec, Matrix ax);

enum tiny_ErrorCode tiny_BFunc(tiny_AdmmWorkspace* work, Matrix current_solution_vec, Matrix current_solution_vec_b, Matrix solution_update_vec, float zeta, Matrix b);

enum tiny_ErrorCode tiny_GivensRotation(tiny_AdmmWorkspace* work, Matrix matrix, int j);

enum tiny_ErrorCode tiny_ForwardDifferenceGMRES(tiny_AdmmWorkspace* work);

enum tiny_ErrorCode tiny_RollOutClosedLoop_CGMRES(tiny_AdmmWorkspace* work);

enum tiny_ErrorCode tiny_ForwardPass(tiny_AdmmWorkspace* work);

enum tiny_ErrorCode tiny_Solvecgmres(tiny_AdmmWorkspace* work);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef CGMRES_H
