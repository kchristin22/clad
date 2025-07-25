// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTFormula.out 2>&1 | %filecheck %s
// RUN: ./TFormula.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTFormula.out
// RUN: ./TFormula.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

using Double_t = double;

namespace TMath {
  Double_t Abs(Double_t x) { return ::std::abs(x); }
  Double_t Exp(Double_t x) { return ::std::exp(x); }
  Double_t Sin(Double_t x) { return ::std::sin(x); }
}

// We do not need to add custom derivatives here.
// Clad should automatically generate these functions.
namespace clad {
namespace custom_derivatives {
namespace TMath {
clad::ValueAndPushforward<Double_t, Double_t> Abs_pushforward(Double_t x, Double_t d_x) {
  return std::abs_pushforward(x, d_x);
}
clad::ValueAndPushforward<Double_t, Double_t> Exp_pushforward(Double_t x, Double_t d_x) {
  return std::exp_pushforward(x, d_x);
}
clad::ValueAndPushforward<Double_t, Double_t> Sin_pushforward(Double_t x, Double_t d_x) {
  return std::cos_pushforward(x, d_x);
}
} // namespace TMath
} // namespace custom_derivatives
} // namespace clad

Double_t TFormula_example(const Double_t* x, Double_t* p) {
  return x[0]*(p[0] + p[1] + p[2]) + TMath::Exp(-p[0]) + TMath::Abs(p[1]);
}
// _grad = { x[0] + (-1) * Exp_darg0(-p[0]), x[0] + Abs_darg0(p[1]), x[0] }

void TFormula_example_grad_1(const Double_t* x, Double_t* p, Double_t* _d_p);
//CHECK:   void TFormula_example_grad_1(const Double_t *x, Double_t *p, Double_t *_d_p) {
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += x[0] * 1;
//CHECK-NEXT:           _d_p[1] += x[0] * 1;
//CHECK-NEXT:           _d_p[2] += x[0] * 1;
//CHECK-NEXT:           Double_t _r0 = 0.;
//CHECK-NEXT:           _r0 += 1 * clad::custom_derivatives::TMath::Exp_pushforward(-p[0], 1.).pushforward;
//CHECK-NEXT:           _d_p[0] += -_r0;
//CHECK-NEXT:           Double_t _r1 = 0.;
//CHECK-NEXT:           _r1 += 1 * clad::custom_derivatives::TMath::Abs_pushforward(p[1], 1.).pushforward;
//CHECK-NEXT:           _d_p[1] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_0(const Double_t *x, Double_t *p) {
//CHECK-NEXT:       {{double|Double_t}} _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       clad::ValueAndPushforward<Double_t, Double_t> _t1 = clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -1.);
//CHECK-NEXT:       return 0. * _t0 + x[0] * (1. + 0. + 0.) + _t1.pushforward + 0.;
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_1(const Double_t *x, Double_t *p) {
//CHECK-NEXT:       {{double|Double_t}} _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       clad::ValueAndPushforward<Double_t, Double_t> _t1 = clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -0.);
//CHECK-NEXT:       clad::ValueAndPushforward<Double_t, Double_t> _t2 = clad::custom_derivatives::TMath::Abs_pushforward(p[1], 1.);
//CHECK-NEXT:       return 0. * _t0 + x[0] * (0. + 1. + 0.) + _t1.pushforward + _t2.pushforward;
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_2(const Double_t *x, Double_t *p) {
//CHECK-NEXT:       {{double|Double_t}} _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       clad::ValueAndPushforward<Double_t, Double_t> _t1 = clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -0.);
//CHECK-NEXT:       return 0. * _t0 + x[0] * (0. + 0. + 1.) + _t1.pushforward + 0.;
//CHECK-NEXT:   }

Double_t TFormula_hess1(const Double_t *x, Double_t *p) {
    return x[0] * std::sin(p[0]) - x[1] * std::cos(p[1]);
}

int main() {
  Double_t x[] = { 3 };
  Double_t p[] = { -std::log(2), -1, 3 };
  Double_t result[3] = { 0 };

  auto gradient = clad::gradient(TFormula_example, "p");
  gradient.execute(x, p, result);
  printf("Result is = {%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: Result is = {1.00, 2.00, 3.00}

  auto differentiation0 = clad::differentiate(TFormula_example, "p[0]");
  printf("Result is = {%.2f}\n", differentiation0.execute(x, p)); // CHECK-EXEC: Result is = {1.00}

  auto differentiation1 = clad::differentiate(TFormula_example, "p[1]");
  printf("Result is = {%.2f}\n", differentiation1.execute(x, p)); // CHECK-EXEC: Result is = {2.00}

  auto differentiation2 = clad::differentiate(TFormula_example, "p[2]");
  printf("Result is = {%.2f}\n", differentiation2.execute(x, p)); // CHECK-EXEC: Result is = {3.00}

  {
     Double_t x[] = { 1, 2 };
     Double_t p[] = { 30, 60 };

     auto hess1 = clad::hessian(TFormula_hess1, "p[0:1]");
     Double_t hess_result[4] = { 0 };
     hess1.execute(x, p, hess_result);
     // hes_result[0] = -1 * x[0] * std::sin(p[0]);
     // hes_result[1] = 0;
     // hes_result[2] = 0;
     // hes_result[0] = x[1] * std::cos(p[1]);
     printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n",
            hess_result[0], hess_result[1], hess_result[2], hess_result[3]); // CHECK-EXEC: Result is = {0.99, 0.00, 0.00, -1.90}
  }
}
