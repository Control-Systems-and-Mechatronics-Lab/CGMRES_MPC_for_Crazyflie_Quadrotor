#include "cgmres.h"
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>

#define MASS 0.033
#define G 9.81
#define KT 2.245365e-6 * 65535
// #define KT 0.1154
#define KM (KT * 0.0008)
#define EL (0.046 / 1.414213562)

static const double J_body[3][3] = {
    { 1.66e-5,  0.83e-6,  0.72e-6 },
    { 0.83e-6,  1.66e-5,  1.80e-6 },
    { 0.72e-6,  1.80e-6,  2.93e-5 }
};

typedef struct {
    double data[13];
  } Vector13;
  
  typedef struct {
    double data[12];
  } Vector12;
  
  typedef struct {
    double data[3][3];
  } Matrix3;

// phix = Q_f * (x_k - x_ref)
enum tiny_ErrorCode tiny_ComputePhix(tiny_AdmmWorkspace* work, Matrix phix) {
    if (!work) return TINY_SLAP_ERROR;
        
        MatAdd(phix, work->cgmres.optix[work->data->model[0].nhorizon-1], work->data->Xref[work->data->model[0].nhorizon-1], -1);
        // slap_MatMulAB(phix, work->data->Q, phix); // 2025 04 04
        slap_MatMulAB(phix, work->data->Qf, phix); // 2025 05 29

  
        // MatScale(phix, 0.01); // 20225 03 02

    return TINY_NO_ERROR;
}


enum tiny_ErrorCode tiny_ComputeHx(tiny_AdmmWorkspace* work, int k, Matrix hx) {
    if (!work) return TINY_SLAP_ERROR;

    //    MatAdd(hx, optix[k], Xref[k], -1)  → hx = x_k - x_ref
    MatAdd(hx, work->cgmres.optix[k], work->data->Xref[k], -1);

    //    MatMulAdd(hx, Q, hx, 1, 0)       → hx = Q · (x_k - x_ref)
    MatMulAdd(hx, work->data->Q, hx, 1, 0);

    // 2. hx += A^T(k) · λ_{k+1}
    MatMulAdd(hx, work->cgmres.Ac[k], work->cgmres.lambda[k+1], 1, 1);

    return TINY_NO_ERROR;
}


enum tiny_ErrorCode tiny_ComputeHu(tiny_AdmmWorkspace* work, int k, Matrix hu) {
    if (!work) return TINY_SLAP_ERROR;

    // 1. hu = R * u_k
    MatMulAdd(hu, work->data->R, work->cgmres.optiu[k], 1, 0);
    MatMulAdd(hu, work->cgmres.Bc[k], work->cgmres.lambda[k+1], 1, 1);
        // 4. Soft constraint penalty term
    float rho = 100.0f;
    for (int i = 0; i < hu.rows; ++i) {
        float u = work->cgmres.optiu[k].data[i];
        float z = work->ZU[k].data[i];
        float y = work->soln->YU[k].data[i];
        hu.data[i] += rho * (u - z + y);
    }
    return TINY_NO_ERROR;
}

// void compute_jacobian_state_symbolic(Vector12* x, float u[4], tiny_AdmmWorkspace* work) {
void compute_jacobian_state_symbolic(tiny_AdmmWorkspace* work, Matrix x, Matrix u, int k) {

    float* A = work->cgmres.Ac[k].data;

	float phi1 = x.data[3], phi2 = x.data[4], phi3 = x.data[5];
	float phi_sq = phi1 * phi1 + phi2 * phi2 + phi3 * phi3;
	float inv_sqrt = 1.0f / sqrtf(1.0f + phi_sq);
	float q0 = inv_sqrt;
	float q1 = q0 * phi1, q2 = q0 * phi2, q3 = q0 * phi3;


	float wx = x.data[9], wy = x.data[10], wz = x.data[11];


	float q0q0 = q0 * q0, q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;
	float q_norm = q0q0 + q1q1 + q2q2 + q3q3;
	float half_qnorm = 0.5f * q_norm;
	float inv_qnorm_sq = 1.0f / (q_norm * q_norm);


	float u1p = u.data[0] - work->data->lcu.data[0], u2p = u.data[1] - work->data->lcu.data[1];
	float u3p = u.data[2] - work->data->lcu.data[2], u4p = u.data[3] - work->data->lcu.data[3];
	float usum = u1p + u2p + u3p + u4p;


	// const float c1 = 16.8171423171429f, c2 = 8.40857115857143f; // 35g
    // const float c1 =  17.8363630636364f, c2 = 8.91818153181818; // Kt 2.24
    const float c1 =  13.9878787878788f, c2 = 6.99393939393939f; // Kt 0.1154


	A[0 * 12 + 6] = 1.0f;
	A[1 * 12 + 7] = 1.0f;
	A[2 * 12 + 8] = 1.0f;
	A[3 * 12 + 4] = half_qnorm * wz;
	A[3 * 12 + 5] = -half_qnorm * wy;
	A[3 * 12 + 9] = half_qnorm;
	A[4 * 12 + 3] = -half_qnorm * wz;
	A[4 * 12 + 5] = half_qnorm * wx;
	A[4 * 12 + 10] = half_qnorm;
	A[5 * 12 + 3] = half_qnorm * wy;
	A[5 * 12 + 4] = -half_qnorm * wx;
	A[5 * 12 + 11] = half_qnorm;

	// row 6
	float t6 = usum * (
		-q0 * (c1 * q1 * (q0 * q2 + q1 * q3) - c2 * q3 * q_norm)
		+ q1 * (c1 * q0 * (q0 * q2 + q1 * q3) - c2 * q2 * q_norm)
		- q2 * (c2 * q1 * q_norm - c1 * q3 * (q0 * q2 + q1 * q3))
		+ q3 * (c2 * q0 * q_norm - c1 * q2 * (q0 * q2 + q1 * q3))
		) * inv_qnorm_sq;
	A[6 * 12 + 3] = t6;
	A[6 * 12 + 4] = c2 * (
		(q0q0 + q1q1) * usum
		- (q2q2 + q3q3) * usum
		) / q_norm;

	// row 7
	float n7 = (
		-q0q0 * usum
		+ q1q1 * usum
		- q2q2 * usum
		+ q3q3 * usum
		);
	A[7 * 12 + 3] = c2 * n7 / q_norm;
	float t7 = usum * (
		q0 * (c1 * q2 * (q0 * q1 - q2 * q3) + c2 * q3 * q_norm)
		+ q1 * (c2 * q2 * q_norm + c1 * q3 * (q0 * q1 - q2 * q3))
		- q2 * (c1 * q0 * (q0 * q1 - q2 * q3) - c2 * q1 * q_norm)
		+ q3 * (c2 * q0 * q_norm - c1 * q1 * (q0 * q1 - q2 * q3))
		) * inv_qnorm_sq;
	A[7 * 12 + 4] = t7;

	// row 8
	A[8 * 12 + 3] = c1 * (-(q0 * q1 + q2 * q3) * usum) / q_norm;
	A[8 * 12 + 4] = c1 * ((-q0 * q2 + q1 * q3) * usum) / q_norm;

	A[9 * 12 + 9] = -0.0019757139f * wx - 0.0487024361f * wy + 0.0159427956f * wz;
	A[9 * 12 + 10] = -0.0487024361f * wx - 0.2197619996f * wy - 0.7661612619f * wz;
	A[9 * 12 + 11] = 0.0159427956f * wx - 0.7661612619f * wy + 0.2217377135f * wz;
	A[10 * 12 + 9] = 0.0936074158f * wx + 0.1114817158f * wy + 0.7761372460f * wz;
	A[10 * 12 + 10] = 0.1114817158f * wx + 0.0042877438f * wy - 0.0165079879f * wz;
	A[10 * 12 + 11] = 0.7761372460f * wx - 0.0165079879f * wy - 0.0978951595f * wz;
	A[11 * 12 + 9] = -0.0623573664f * wx - 0.0056519227f * wy - 0.1095060019f * wz;
	A[11 * 12 + 10] = -0.0056519227f * wx + 0.0617921741f * wy + 0.0444146924f * wz;
	A[11 * 12 + 11] = -0.1095060019f * wx + 0.0444146924f * wy + 0.0005651929f * wz;

    }

// void compute_jacobian_input_symbolic(Vector12* x, float u[4], tiny_AdmmWorkspace* work) {
void compute_jacobian_input_symbolic(tiny_AdmmWorkspace* work, Matrix x, int k) {

    float* B = work->cgmres.Bc[k].data;


	float phi1 = x.data[3], phi2 = x.data[4], phi3 = x.data[5];
	float phi_sq = phi1 * phi1 + phi2 * phi2 + phi3 * phi3;
	float inv_sqrt = 1.0f / sqrtf(1.0f + phi_sq);
	float q0 = inv_sqrt, q1 = q0 * phi1, q2 = q0 * phi2, q3 = q0 * phi3;


	float q_norm = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
	// const float cA = 8.40857115857143f / q_norm; //35g
    const float cA = 6.99393939393939f / q_norm;
	// const float cB = 4.20428557928571f / q_norm; // 35g
    const float cB = 3.4969696969697f / q_norm;
	// row 6
	float a = q0 * q2 + q1 * q3;
	for (int i = 0; i < 4; i++) B[6 * 4 + i] = cA * a;

	// row 7
	float b = -q0 * q1 + q2 * q3;
	for (int i = 0; i < 4; i++) B[7 * 4 + i] = cA * b;

	// row 8
	float c = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
	for (int i = 0; i < 4; i++) B[8 * 4 + i] = cB * c;


	// B[9 * 4 + 0] = -226.301627f; B[9 * 4 + 1] = -249.214482f;
	// B[9 * 4 + 2] = 226.554783f;  B[9 * 4 + 3] = 248.961327f;
	// B[10 * 4 + 0] = -227.413113f;  B[10 * 4 + 1] = 250.072813f;
	// B[10 * 4 + 2] = 228.121950f;  B[10 * 4 + 3] = -250.781649f;
	// B[11 * 4 + 0] = 16.229760f;  B[11 * 4 + 1] = -5.936786f;
	// B[11 * 4 + 2] = -22.883540f;  B[11 * 4 + 3] = 12.590566f;
    B[9 * 4 + 0] =  -215.162647337513f; B[9 * 4 + 1] = -238.586122704622f;
	B[9 * 4 + 2] =   216.963698701890f;  B[9 * 4 + 3] =  236.785071340246f;
	B[10 * 4 + 0] = -214.819718508824f;  B[10 * 4 + 1] =  236.442142511556f;
	B[10 * 4 + 2] = 219.862662329078f;  B[10 * 4 + 3] = -241.485086331811f;
	B[11 * 4 + 0] = -5.00739595225625f;  B[11 * 4 + 1] = 14.8291963353764f;
	B[11 * 4 + 2] = -42.3302374081127f;  B[11 * 4 + 3] = 32.5084370249926f;
}

//


void quad_dynamics_rp(tiny_AdmmWorkspace* work, Vector12 *x, float u[4], Vector12 *dx) {

    float u_hover[4];
    float u_eff[4];
    for (int i = 0; i < 4; ++i) {
        u_hover[i] = -work->data->lcu.data[i];      // u_hover = -lcu
        u_eff[i]   = u[i] + u_hover[i]; 
    }

    float px = x->data[0];
    float py = x->data[1];
    float pz = x->data[2];


    float r1 = x->data[3];
    float r2 = x->data[4];
    float r3 = x->data[5];


    float vx = x->data[6];
    float vy = x->data[7];
    float vz = x->data[8];


    float wx = x->data[9];
    float wy = x->data[10];
    float wz = x->data[11];


    dx->data[0] = vx;
    dx->data[1] = vy;
    dx->data[2] = vz;


    float hat_r[3][3] = {
        {  0.0f,  -r3,   r2 },
        {   r3,  0.0f,  -r1 },
        {  -r2,   r1,  0.0f }
    };

    // 3-2) ĥ(r)² = ĥ(r) · ĥ(r)
    float hat_r2[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            hat_r2[i][j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                hat_r2[i][j] += hat_r[i][k] * hat_r[k][j];
            }
        }
    }

    //     I₃ + ĥ(r) + r·rᵀ = M
    float M[3][3];
    float rrT[3][3] = {
        { r1*r1, r1*r2, r1*r3 },
        { r2*r1, r2*r2, r2*r3 },
        { r3*r1, r3*r2, r3*r3 }
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float Iij = (i == j) ? 1.0f : 0.0f;
            M[i][j] = Iij + hat_r[i][j] + rrT[i][j];
        }
    }
    // 3-4) ṙ = ½· M · ω
    dx->data[3] = 0.5f * ( M[0][0]*wx + M[0][1]*wy + M[0][2]*wz );
    dx->data[4] = 0.5f * ( M[1][0]*wx + M[1][1]*wy + M[1][2]*wz );
    dx->data[5] = 0.5f * ( M[2][0]*wx + M[2][1]*wy + M[2][2]*wz );

    // 4-1) ||r||²
    float r_norm_sq = r1*r1 + r2*r2 + r3*r3;

    float factor = 2.0f / (1.0f + r_norm_sq);

    // 4-3) R = I + factor·( ĥ(r) + ĥ(r)² )
    float R[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float Iij = (i == j) ? 1.0f : 0.0f;
            R[i][j] = Iij + factor * ( hat_r[i][j] + hat_r2[i][j] );
        }
    }


    float F = KT * (u_eff[0] + u_eff[1] + u_eff[2] + u_eff[3]);

    float term0 =       R[0][2] * F;
    float term1 =       R[1][2] * F;
    float term2 =       R[2][2] * F;

    dx->data[6] = (1.0f / MASS) * term0;                     
    dx->data[7] = (1.0f / MASS) * term1;                     
    dx->data[8] = (1.0f / MASS) * term2 - G;                   


  
    float tau[3];
    tau[0] = EL * KT * (-u_eff[0] - u_eff[1] + u_eff[2] + u_eff[3]); 
    tau[1] = EL * KT * (-u_eff[0] + u_eff[1] + u_eff[2] - u_eff[3]);  
    tau[2] =     KM * (-u_eff[0] + u_eff[1] - u_eff[2] + u_eff[3]);   


    float Jw[3] = {
        J_body[0][0]*wx + J_body[0][1]*wy + J_body[0][2]*wz,
        J_body[1][0]*wx + J_body[1][1]*wy + J_body[1][2]*wz,
        J_body[2][0]*wx + J_body[2][1]*wy + J_body[2][2]*wz
    };

    // 6-3) ω × (J ω)
    float omega_cross_Jw[3];
    omega_cross_Jw[0] = wy * Jw[2] - wz * Jw[1];
    omega_cross_Jw[1] = wz * Jw[0] - wx * Jw[2];
    omega_cross_Jw[2] = wx * Jw[1] - wy * Jw[0];

    // 6-4) net = −[ω×(Jω)] + τ
    float net[3];
    for (int i = 0; i < 3; i++) {
        net[i] = -omega_cross_Jw[i] + tau[i];
    }


    dx->data[9]  = net[0] / J_body[0][0];   // ω̇ₓ
    dx->data[10] = net[1] / J_body[1][1];   // ω̇ᵧ
    dx->data[11] = net[2] / J_body[2][2];   // ω̇𝓏
}

    
enum tiny_ErrorCode tiny_ComputeOptimalityError(tiny_AdmmWorkspace* work, const Matrix state_vec, const Matrix current_solution_vec, Matrix optimality_vec) {
    if (!work) return TINY_SLAP_ERROR;
    int N = work->data->model[0].nhorizon; 
    tiny_Model* model = work->data->model;
    // //printf("sdfg");
    
    work->cgmres.optix[0] = state_vec;

    // work->soln->X[0] = state_vec;
    // //printf("sdfg\n");
    for (int i=0; i<N-1; i++){
        for (int j = 0; j < 4; j++){
            // work->soln->U[i].data[j] = current_solution_vec.data[i*4 + j];
            work->cgmres.optiu[i].data[j] = current_solution_vec.data[i*4 + j];
        }

        // // printf("computeoptimalityerror : ");
        compute_jacobian_state_symbolic(work, work->cgmres.optix[i], work->cgmres.optiu[i], i);
        compute_jacobian_input_symbolic(work, work->cgmres.optix[i], i);

        Vector12 dynamicxce;
        Vector12 dynamicxdotce;

        for (int k = 0; k < 12; k++){
            dynamicxce.data[k] = work->cgmres.optix[i].data[k];
        }

        quad_dynamics_rp(
            work,
            &dynamicxce,      
            work->cgmres.optiu[i].data,    
            &dynamicxdotce     
        );

        for (int k = 0; k < 12; k++){
            work->cgmres.xdot[i].data[k] = dynamicxdotce.data[k];
        }


        MatScale(work->cgmres.xdot[i], 0.01);
        for (int idx = 0; idx < 12; idx++) {
            // Vector12이므로 rows*cols = 12
            work->cgmres.optix[i+1].data[idx] = work->cgmres.optix[i].data[idx]
                                          + work->cgmres.xdot[i].data[idx];
        }

    }

    // printMatrix(work->cgmres.Ac[2]);
    // printMatrix(work->cgmres.Bc[2]);

    // tiny_ComputePhix(work, &work->cgmres.phix);
    tiny_ComputePhix(work, work->cgmres.lambda[N-1]);

    for(int i=N-2; i>=0; i--){
        tiny_ComputeHx(work, i, work->cgmres.hx[0]);


        // MatScale(work->cgmres.hx[0], 0.02);
        MatScale(work->cgmres.hx[0], 0.01);
        // MatScale(work->cgmres.hx[0], 0.01);

        MatAdd(work->cgmres.lambda[i], work->cgmres.lambda[i+1], work->cgmres.hx[0], 1);


        tiny_ComputeHu(work, i, work->cgmres.hu[0]);

        for(int j = 0; j < 4; j++){
            optimality_vec.data[i*4+j] = work->cgmres.hu[0].data[j];
        }

    }

    return TINY_NO_ERROR;
}

enum tiny_ErrorCode tiny_AxFunc(tiny_AdmmWorkspace* work, Matrix current_solution_vec, Matrix current_solution_vec_ax, Matrix direction_vec, Matrix ax) { // most important to change
    if (!work) return TINY_SLAP_ERROR;

    int Nc = work->data->model[0].nhorizon - 1;
    float difference_increment = 1e-4f;


    MatAdd(current_solution_vec_ax, current_solution_vec, direction_vec, difference_increment);
    // //printMatrix(current_solution_vec_ax);
    tiny_ComputeOptimalityError(work, work->cgmres.X_perturb[0], current_solution_vec_ax, work->cgmres.eu_2ax[0]);


    MatAdd(ax, work->cgmres.eu_2ax[0], work->cgmres.eu_1[0], -1);
    MatScale(ax, 1/difference_increment);    

    return TINY_NO_ERROR;
}

enum tiny_ErrorCode tiny_BFunc(tiny_AdmmWorkspace* work, Matrix current_solution_vec, Matrix current_solution_vec_b, Matrix solution_update_vec, float zeta, Matrix b) {
    if (!work) return TINY_SLAP_ERROR;
    // tiny_Model* model = work->data->model;
    float difference_increment = 1e-4f;
    int Nc = work->data->model[0].nhorizon - 1;

    tiny_ComputeOptimalityError(work, work->soln->X[0], current_solution_vec, work->cgmres.eu[0]);

    tiny_ComputeOptimalityError(work, work->cgmres.X_perturb[0], current_solution_vec, work->cgmres.eu_1[0]);

    MatAdd(current_solution_vec_b, current_solution_vec, solution_update_vec, difference_increment);
    
    tiny_ComputeOptimalityError(work, work->cgmres.X_perturb[0], current_solution_vec_b, work->cgmres.eu_2b[0]);

    //20250228
    MatScale(work->cgmres.eu[0], (1/difference_increment - zeta));
    //printMatrix(work->cgmres.eu[0]);
    MatAdd(b, work->cgmres.eu[0], work->cgmres.eu_2b[0], -1/difference_increment);

    return TINY_NO_ERROR;
}

/* --- tiny_GivensRotation, tiny_DotProduct, tiny_VectorNorm, tiny_ScaleVector 구현 --- */

enum tiny_ErrorCode tiny_GivensRotation(tiny_AdmmWorkspace* work, Matrix column_vec, int j) {

    float tmp1;
    float tmp2;

    tmp1 = work->cgmres.givens_c[0].data[j] * column_vec.data[j] - work->cgmres.givens_s[0].data[j] * column_vec.data[j + 1];
    tmp2 = work->cgmres.givens_s[0].data[j] * column_vec.data[j] + work->cgmres.givens_c[0].data[j] * column_vec.data[j + 1];

    // //printf("tmp1 : %f\n", tmp1);
    // //printf("tmp2 : %f\n", tmp2);
    column_vec.data[j] = tmp1;
    column_vec.data[j + 1] = tmp2;
    
    return TINY_NO_ERROR;
}


enum tiny_ErrorCode tiny_ForwardDifferenceGMRES(tiny_AdmmWorkspace* work) {

    if (!work) return TINY_SLAP_ERROR;
    int N = work->data->model[0].nhorizon - 1; 


    for (int i = 0; i <= max_krylov_dim; i++) {
        work->cgmres.givens_c[0].data[i] = 0.0f;
        work->cgmres.givens_s[0].data[i] = 0.0f;
        work->cgmres.g_vec[0].data[i] = 0.0f;

    }

        Vector12 dynamicxpertub;
        Vector12 dynamicxdotpertub;

        for (int k = 0; k < 12; k++){
            dynamicxpertub.data[k] = work->soln->X[0].data[k];
        }

        quad_dynamics_rp(
            work,
            &dynamicxpertub,       
            work->soln->U[0].data,   
            &dynamicxdotpertub    
        );

        for (int k = 0; k < 12; k++){
            work->cgmres.xdot[0].data[k] = dynamicxdotpertub.data[k];
        }

        MatScale(work->cgmres.xdot[0], 1e-4f);  // xdot_k ← 0.01 * xdot_k
    
        for (int idx = 0; idx < 12; idx++) {

            work->cgmres.X_perturb[0].data[idx] = work->cgmres.optix[0].data[idx]
                                          + work->cgmres.xdot[0].data[idx];
        }

    tiny_BFunc(work, work->cgmres.current_solution_vec[0], work->cgmres.current_solution_vec_b[0], work->cgmres.solution_update_vec[0], 50.0f, work->cgmres.basis_mat[0]); //////////////////// zeta
     
    work->cgmres.g_vec[0].data[0] = slap_NormTwo(work->cgmres.basis_mat[0]);
    




    MatScale(work->cgmres.basis_mat[0], 1.0 / work->cgmres.g_vec[0].data[0]);
    

    int k;    

    for (k = 0; k < max_krylov_dim; k++) {


        tiny_AxFunc(work, work->cgmres.current_solution_vec[0], work->cgmres.current_solution_vec_ax[0], work->cgmres.basis_mat[k], work->cgmres.basis_mat[k+1]);
     
        for (int j = 0; j <= k; j++) { // compute hessenberg matrix

            work->cgmres.hessenberg_mat[k].data[j] = slap_InnerProduct(work->cgmres.basis_mat[k+1], work->cgmres.basis_mat[j]); // no problem..
            


            MatAdd(work->cgmres.basis_mat[k+1], work->cgmres.basis_mat[k+1], work->cgmres.basis_mat[j], -work->cgmres.hessenberg_mat[k].data[j]); // no problem...



        }

        work->cgmres.hessenberg_mat[k].data[k+1] = slap_NormTwo(work->cgmres.basis_mat[k+1]); //hessenberg_mat_(k+1,k) = basis_mat_.col(k+1).norm();
        

        if (work->cgmres.hessenberg_mat[k].data[k+1] != 0) {
            MatScale(work->cgmres.basis_mat[k+1], 1.0 / work->cgmres.hessenberg_mat[k].data[k+1]);

        } else {
            //printf("The modified Gram-Schmidt breakdown at k=%d\n", k);        
            break;
        }


        for (int j = 0; j < k; j++) {

            tiny_GivensRotation(work, work->cgmres.hessenberg_mat[k], j);        
        }
        

        float nu = sqrt(work->cgmres.hessenberg_mat[k].data[k] * work->cgmres.hessenberg_mat[k].data[k] + work->cgmres.hessenberg_mat[k].data[k+1] * work->cgmres.hessenberg_mat[k].data[k+1]);  //double nu = std::sqrt(hessenberg_mat[0]_(k,k)*hessenberg_mat[0]_(k,k) + hessenberg_mat[0]_(k+1,k)*hessenberg_mat[0]_(k+1,k));
        //printf("nu %f\n", nu);
        if (nu != 0) {
            work->cgmres.givens_c[0].data[k] = work->cgmres.hessenberg_mat[k].data[k] / nu;
            work->cgmres.givens_s[0].data[k] = - work->cgmres.hessenberg_mat[k].data[k+1] / nu;
            work->cgmres.hessenberg_mat[k].data[k] = work->cgmres.givens_c[0].data[k] * work->cgmres.hessenberg_mat[k].data[k] - work->cgmres.givens_s[0].data[k] * work->cgmres.hessenberg_mat[k].data[k+1];
            work->cgmres.hessenberg_mat[k].data[k+1] = 0.0f;
            tiny_GivensRotation(work, work->cgmres.g_vec[0], k);

        }



        for (int j = 0; j <= k; j++) {
            float inner_product = slap_InnerProduct(work->cgmres.basis_mat[k+1], work->cgmres.basis_mat[j]);

        }
        

    }

    for (int i = k - 1; i >= 0; i--) {
        float tmp = work->cgmres.g_vec[0].data[i];

        for (int j = i + 1; j < k; j++) {
            tmp -= (work->cgmres.hessenberg_mat[j].data[i]) * work->cgmres.givens_c[0].data[j];
        }

        work->cgmres.givens_c[0].data[i] = tmp / (work->cgmres.hessenberg_mat[i].data[i]); 
    }


    for (int i = 0; i < (N * 4); i++) {
        float tmp = 0.0f;
        for (int j = 0; j < k; j++) {

            tmp += work->cgmres.basis_mat[j].data[i] * work->cgmres.givens_c[0].data[j];//tmp += basis_mat_(i,j) * givens_c[0]_vec_(j);

        }
        
        work->cgmres.solution_update_vec[0].data[i] += tmp; //solution_update_vec(i) += tmp;    
    } 

    work->cgmres.optix[0] = work->soln->X[0];

    // MatAdd(work->cgmres.current_solution_vec[0], work->cgmres.current_solution_vec[0], work->cgmres.solution_update_vec[0], 0.02);
    MatAdd(work->cgmres.current_solution_vec[0], work->cgmres.current_solution_vec[0], work->cgmres.solution_update_vec[0], 0.01);
    // MatAdd(work->cgmres.current_solution_vec[0], work->cgmres.current_solution_vec[0], work->cgmres.solution_update_vec[0], 0.01);

    for (int h=0; h < N; h++){
        for (int i=0; i<4; i++){

            float tmp = work->cgmres.current_solution_vec[0].data[4*h+i];
            if(tmp < work->data->lcu.data[i]){
                tmp = work->data->lcu.data[i];
            }
            else if(tmp > (work->data->ucu.data[i])){
                tmp = work->data->ucu.data[i];
            }
            work->cgmres.current_solution_vec[0].data[4*h+i] = tmp;

            work->soln->U[h].data[i] = work->cgmres.current_solution_vec[0].data[4*h+i];

        }
    }
    for (int k = 0; k < N - 1; ++k) {
        for (int i = 0; i < 4; ++i) {
            float u = work->soln->U[k].data[i];
            float y = work->soln->YU[k].data[i];
            float z_proj = T_MIN(T_MAX(u + y, work->data->lcu.data[i]), work->data->ucu.data[i]);
    
            work->ZU[k].data[i] = z_proj;
            work->soln->YU[k].data[i] = y + u - z_proj;  // Dual update
        }
    }

        return TINY_NO_ERROR;
}

enum tiny_ErrorCode tiny_RollOutClosedLoop_CGMRES(tiny_AdmmWorkspace* work) {
    tiny_Model* model = work->data->model;
    int N = model[0].nhorizon;

    for (int k = 0; k < N - 1; ++k) {

        Vector12 dynamicx;
        Vector12 dynamicxdot;

        for (int i = 0; i < 12; i++){
            dynamicx.data[i] = work->soln->X[k].data[i];
        }

        quad_dynamics_rp(
            work,
            &dynamicx,       
            work->soln->U[k].data,    
            &dynamicxdot     
        );

        for (int i = 0; i < 12; i++){
            work->cgmres.xdot[k].data[i] = dynamicxdot.data[i];
        }


        MatScale(work->cgmres.xdot[k], 0.01);
        for (int idx = 0; idx < work->soln->X[k].rows * work->soln->X[k].cols; idx++) {
 
            work->soln->X[k+1].data[idx] = work->soln->X[k].data[idx]
                                          + work->cgmres.xdot[k].data[idx];
        }
        for (int i = 0; i < 12; i++){
            work->cgmres.optix[k].data[i] = work->soln->X[k].data[i];
        }

    }

    return TINY_NO_ERROR;
}

enum tiny_ErrorCode tiny_ForwardPass(tiny_AdmmWorkspace* work) {
    MatCpy(work->soln->X[0], work->data->x0);

    tiny_RollOutClosedLoop_CGMRES(work);
    return TINY_NO_ERROR;
}

enum tiny_ErrorCode tiny_Solvecgmres(tiny_AdmmWorkspace* work) {
    if (!work) return TINY_SLAP_ERROR;
    MatCpy(work->soln->X[0], work->data->x0);

    tiny_ForwardDifferenceGMRES(work);
    return TINY_NO_ERROR;
}

