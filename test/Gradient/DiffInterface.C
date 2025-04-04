// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oGradientDiffInterface.out 2>&1 | %filecheck %s
// RUN: ./GradientDiffInterface.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oGradientDiffInterface.out
// RUN: ./GradientDiffInterface.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

double f_1(double x, double y, double z) {
  return 0 * x + 1 * y + 2 * z;
}

// all
//CHECK:   void f_1_grad(double x, double y, double z, double *_d_x, double *_d_y, double *_d_z) {
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 0 * 1;
//CHECK-NEXT:           *_d_y += 1 * 1;
//CHECK-NEXT:           *_d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// x
//CHECK:   void f_1_grad_0(double x, double y, double z, double *_d_x) {
//CHECK-NEXT:       double _d_y = 0.;
//CHECK-NEXT:       double _d_z = 0.;
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 0 * 1;
//CHECK-NEXT:           _d_y += 1 * 1;
//CHECK-NEXT:           _d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// y
//CHECK:   void f_1_grad_1(double x, double y, double z, double *_d_y) {
//CHECK-NEXT:       double _d_x = 0.;
//CHECK-NEXT:       double _d_z = 0.;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_x += 0 * 1;
//CHECK-NEXT:           *_d_y += 1 * 1;
//CHECK-NEXT:           _d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// z
//CHECK:   void f_1_grad_2(double x, double y, double z, double *_d_z) {
//CHECK-NEXT:       double _d_x = 0.;
//CHECK-NEXT:       double _d_y = 0.;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_x += 0 * 1;
//CHECK-NEXT:           _d_y += 1 * 1;
//CHECK-NEXT:           *_d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// x, y
//CHECK:   void f_1_grad_0_1(double x, double y, double z, double *_d_x, double *_d_y) {
//CHECK-NEXT:       double _d_z = 0.;
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += 0 * 1;
//CHECK-NEXT:           *_d_y += 1 * 1;
//CHECK-NEXT:           _d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// y, z
//CHECK:   void f_1_grad_1_2(double x, double y, double z, double *_d_y, double *_d_z) {
//CHECK-NEXT:       double _d_x = 0.;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_x += 0 * 1;
//CHECK-NEXT:           *_d_y += 1 * 1;
//CHECK-NEXT:           *_d_z += 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

#define TEST(F, ...)                                                           \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    result[2] = 0;                                                             \
    F.execute(0, 0, 0, __VA_ARGS__);                                           \
    printf("{%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]);           \
  }

int main () {
  double result[3];

  auto f1_grad_all = clad::gradient(f_1);
  TEST(f1_grad_all,
       &result[0],
       &result[1],
       &result[2]); // CHECK-EXEC: {0.00, 1.00, 2.00}

  auto f1_grad_x = clad::gradient(f_1, "x");
  TEST(f1_grad_x, &result[0]); // CHECK-EXEC: {0.00, 0.00, 0.00}

  auto f1_grad_y = clad::gradient(f_1, "y");
  TEST(f1_grad_y, &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_0 = clad::gradient(f_1, "1");
  TEST(f1_grad_0, &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_z = clad::gradient(f_1, "z");
  TEST(f1_grad_z, &result[2]); // CHECK-EXEC: {0.00, 0.00, 2.00}

  auto f1_grad_xy = clad::gradient(f_1, "x, y");
  TEST(f1_grad_xy, &result[0], &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_0y = clad::gradient(f_1, "0, y");
  TEST(f1_grad_0y, &result[0], &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_10 = clad::gradient(f_1, "1, 0");
  TEST(f1_grad_10, &result[0], &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_yx = clad::gradient(f_1, "y, x");
  TEST(f1_grad_yx, &result[0], &result[1]); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_yz = clad::gradient(f_1, "y, z");
  TEST(f1_grad_yz, &result[1], &result[2]); // CHECK-EXEC: {0.00, 1.00, 2.00}

  auto f1_grad_xyz = clad::gradient(f_1, "x, y, z");
  TEST(f1_grad_xyz,
       &result[0],
       &result[1],
       &result[2]); // CHECK-EXEC: {0.00, 1.00, 2.00}

  auto f1_grad_zyx = clad::gradient(f_1, "z,y,x");
  TEST(f1_grad_zyx,
       &result[0],
       &result[1],
       &result[2]); // CHECK-EXEC: {0.00, 1.00, 2.00}

  return 0;
}
