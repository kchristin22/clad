#include "rsbench.cuh"
#include "clad/Differentiator/Differentiator.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor GPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, RSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

void calculate_micro_xs_doppler_pullback(
    double *micro_xs, int nuc, double E, Input input, const int *n_windows,
    const double *pseudo_K0RS, const Window *windows, const Pole *poles,
    int max_num_windows, int max_num_poles, double *_d_micro_xs, int *_d_nuc,
    double *_d_E, Input *_d_input, int *_d_max_num_windows,
    int *_d_max_num_poles) __attribute__((device));
void calculate_micro_xs_pullback(double *micro_xs, int nuc, double E,
                                 Input input, const int *n_windows,
                                 const double *pseudo_K0RS,
                                 const Window *windows, const Pole *poles,
                                 int max_num_windows, int max_num_poles,
                                 double *_d_micro_xs, int *_d_nuc, double *_d_E,
                                 Input *_d_input, int *_d_max_num_windows,
                                 int *_d_max_num_poles) __attribute__((device));
void calculate_macro_xs_grad_0(double *macro_xs, int mat, double E, Input input,
                               const int *num_nucs, const int *mats,
                               int max_num_nucs, const double *concs,
                               const int *n_windows, const double *pseudo_K0Rs,
                               const Window *windows, const Pole *poles,
                               int max_num_windows, int max_num_poles,
                               double *_d_macro_xs) __attribute__((device))
{
    int _d_mat = 0;
    double _d_E = 0.;
    Input _d_input = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0, 0, 0, 0};
    int _d_max_num_nucs = 0;
    int _d_max_num_windows = 0;
    int _d_max_num_poles = 0;
    int _d_i = 0;
    int i = 0;
    clad::tape<clad::array<double>> _t1 = {};
    double _d_micro_xs[4] = {0};
    clad::array<double> micro_xs(4);
    clad::tape<int> _t2 = {};
    int _d_nuc = 0;
    int nuc = 0;
    clad::tape<bool> _cond0 = {};
    clad::tape<unsigned long> _t3 = {};
    clad::tape<int> _t4 = {};
    int _d_j = 0;
    int j = 0;
    clad::tape<double> _t5 = {};
    unsigned long _t0 = 0UL;
    for (i = 0;; i++)
    {
        {
            if (!(i < num_nucs[mat]))
                break;
        }
        _t0++;
        clad::push(_t1, std::move(micro_xs)), micro_xs = {0.};
        clad::push(_t2, nuc), nuc = mats[mat * max_num_nucs + i];
        {
            clad::push(_cond0, input.doppler == 1);
            if (clad::back(_cond0))
                calculate_micro_xs_doppler(micro_xs, nuc, E, input, n_windows,
                                           pseudo_K0Rs, windows, poles,
                                           max_num_windows, max_num_poles);
            else
                calculate_micro_xs(micro_xs, nuc, E, input, n_windows,
                                   pseudo_K0Rs, windows, poles, max_num_windows,
                                   max_num_poles);
        }
        clad::push(_t3, 0UL);
        for (clad::push(_t4, j), j = 0;; j++)
        {
            {
                if (!(j < 4))
                    break;
            }
            clad::back(_t3)++;
            clad::push(_t5, micro_xs[j]);
            macro_xs[j] += clad::back(_t5) * concs[mat * max_num_nucs + i];
        }
    }
    for (;; _t0--)
    {
        {
            if (!_t0)
                break;
        }
        i--;
        {
            for (;; clad::back(_t3)--)
            {
                {
                    if (!clad::back(_t3))
                        break;
                }
                j--;
                {
                    double _r_d0 = _d_macro_xs[j];
                    _d_micro_xs[j] += _r_d0 * concs[mat * max_num_nucs + i];
                    clad::pop(_t5);
                }
            }
            {
                _d_j = 0;
                j = clad::pop(_t4);
            }
            clad::pop(_t3);
        }
        {
            if (clad::back(_cond0))
            {
                int _r0 = 0;
                double _r1 = 0.;
                Input _r2 = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0,
                             0, 0, 0};
                int _r3 = 0;
                int _r4 = 0;
                calculate_micro_xs_doppler_pullback(
                    micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows,
                    poles, max_num_windows, max_num_poles, _d_micro_xs, &_r0,
                    &_r1, &_r2, &_r3, &_r4);
                _d_nuc += _r0;
                _d_E += _r1;
                _d_max_num_windows += _r3;
                _d_max_num_poles += _r4;
            }
            else
            {
                int _r5 = 0;
                double _r6 = 0.;
                Input _r7 = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0,
                             0, 0, 0};
                int _r8 = 0;
                int _r9 = 0;
                calculate_micro_xs_pullback(
                    micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows,
                    poles, max_num_windows, max_num_poles, _d_micro_xs, &_r5,
                    &_r6, &_r7, &_r8, &_r9);
                _d_nuc += _r5;
                _d_E += _r6;
                _d_max_num_windows += _r8;
                _d_max_num_poles += _r9;
            }
            clad::pop(_cond0);
        }
        {
            _d_nuc = 0;
            nuc = clad::pop(_t2);
        }
        {
            clad::zero_init(_d_micro_xs);
            micro_xs = clad::pop(_t1);
        }
    }
}
void calculate_sig_T_pullback(int nuc, double E, Input input,
                              const double *pseudo_K0RS, RSComplex *sigTfactors,
                              int *_d_nuc, double *_d_E, Input *_d_input,
                              RSComplex *_d_sigTfactors)
    __attribute__((device));
void c_sub_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device));
void c_mul_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((host))
__attribute__((device));
void fast_nuclear_W_pullback(RSComplex Z, RSComplex _d_y, RSComplex *_d_Z)
    __attribute__((device));
void calculate_micro_xs_doppler_pullback(
    double *micro_xs, int nuc, double E, Input input, const int *n_windows,
    const double *pseudo_K0RS, const Window *windows, const Pole *poles,
    int max_num_windows, int max_num_poles, double *_d_micro_xs, int *_d_nuc,
    double *_d_E, Input *_d_input, int *_d_max_num_windows,
    int *_d_max_num_poles) __attribute__((device))
{
    bool _cond0;
    int _d_i = 0;
    int i = 0;
    clad::tape<Pole> _t1 = {};
    Pole _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    Pole pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_E_c = {0., 0.};
    RSComplex E_c = {0., 0.};
    clad::tape<RSComplex> _t3 = {};
    RSComplex _d_dopp_c = {0., 0.};
    RSComplex dopp_c = {0., 0.};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_Z = {0., 0.};
    RSComplex Z = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_faddeeva = {0., 0.};
    RSComplex faddeeva = {0., 0.};
    double _d_sigT = 0.;
    double sigT;
    double _d_sigA = 0.;
    double sigA;
    double _d_sigF = 0.;
    double sigF;
    double _d_sigE = 0.;
    double sigE;
    double _d_spacing = 0.;
    double spacing = 1. / n_windows[nuc];
    int _d_window = 0;
    int window = (int)(E / spacing);
    {
        _cond0 = window == n_windows[nuc];
        if (_cond0)
            window--;
    }
    RSComplex _d_sigTfactors[4] = {0};
    RSComplex sigTfactors[4];
    calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors);
    Window _d_w = {0., 0., 0., 0, 0};
    Window w = windows[nuc * max_num_windows + window];
    sigT = E * w.T;
    sigA = E * w.A;
    sigF = E * w.F;
    double _d_dopp = 0.;
    double dopp = 0.5;
    unsigned long _t0 = 0UL;
    for (i = w.start;; i++)
    {
        {
            if (!(i < w.end))
                break;
        }
        _t0++;
        clad::push(_t1, std::move(pole)), pole = poles[nuc * max_num_poles + i];
        clad::push(_t2, std::move(E_c)), E_c = {E, 0};
        clad::push(_t3, std::move(dopp_c)), dopp_c = {dopp, 0};
        clad::push(_t4, std::move(Z)),
            Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);
        clad::push(_t5, std::move(faddeeva)), faddeeva = fast_nuclear_W(Z);
        sigT += c_mul(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value])).r;
        sigA += c_mul(pole.MP_RA, faddeeva).r;
        sigF += c_mul(pole.MP_RF, faddeeva).r;
    }
    sigE = sigT - sigA;
    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigE;
    {
        double _r_d10 = _d_micro_xs[3];
        _d_micro_xs[3] = 0.;
        _d_sigE += _r_d10;
    }
    {
        double _r_d9 = _d_micro_xs[2];
        _d_micro_xs[2] = 0.;
        _d_sigF += _r_d9;
    }
    {
        double _r_d8 = _d_micro_xs[1];
        _d_micro_xs[1] = 0.;
        _d_sigA += _r_d8;
    }
    {
        double _r_d7 = _d_micro_xs[0];
        _d_micro_xs[0] = 0.;
        _d_sigT += _r_d7;
    }
    {
        double _r_d6 = _d_sigE;
        _d_sigE = 0.;
        _d_sigT += _r_d6;
        _d_sigA += -_r_d6;
    }
    {
        for (;; _t0--)
        {
            {
                if (!_t0)
                    break;
            }
            i--;
            double _r_d5 = _d_sigF;
            double _r_d4 = _d_sigA;
            double _r_d3 = _d_sigT;
            {
                RSComplex _r9 = {0., 0.};
                fast_nuclear_W_pullback(Z, _d_faddeeva, &_r9);
                _d_faddeeva = {0., 0.};
                faddeeva = clad::pop(_t5);
            }
            {
                RSComplex _r5 = {0., 0.};
                RSComplex _r8 = {0., 0.};
                c_mul_pullback(c_sub(E_c, pole.MP_EA), dopp_c, _d_Z, &_r5,
                               &_r8);
                RSComplex _r6 = {0., 0.};
                RSComplex _r7 = {0., 0.};
                c_sub_pullback(E_c, pole.MP_EA, _r5, &_r6, &_r7);
                _d_Z = {0., 0.};
                Z = clad::pop(_t4);
            }
            {
                _d_dopp += _d_dopp_c.r;
                _d_dopp_c = {0., 0.};
                dopp_c = clad::pop(_t3);
            }
            {
                *_d_E += _d_E_c.r;
                _d_E_c = {0., 0.};
                E_c = clad::pop(_t2);
            }
            {
                _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
                pole = clad::pop(_t1);
            }
        }
        _d_w.start += _d_i;
    }
    {
        double _r_d2 = _d_sigF;
        _d_sigF = 0.;
        *_d_E += _r_d2 * w.F;
        _d_w.F += E * _r_d2;
    }
    {
        double _r_d1 = _d_sigA;
        _d_sigA = 0.;
        *_d_E += _r_d1 * w.A;
        _d_w.A += E * _r_d1;
    }
    {
        double _r_d0 = _d_sigT;
        _d_sigT = 0.;
        *_d_E += _r_d0 * w.T;
        _d_w.T += E * _r_d0;
    }
    {
        int _r2 = 0;
        double _r3 = 0.;
        Input _r4 = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0, 0, 0, 0};
        calculate_sig_T_pullback(nuc, E, input, pseudo_K0RS, sigTfactors, &_r2,
                                 &_r3, &_r4, _d_sigTfactors);
        *_d_nuc += _r2;
        *_d_E += _r3;
    }
    {
        *_d_E += _d_window / spacing;
        double _r1 = _d_window * -(E / (spacing * spacing));
        _d_spacing += _r1;
    }
    double _r0 = _d_spacing * -(1. / (n_windows[nuc] * n_windows[nuc]));
}
void c_div_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device));
inline constexpr void operator_equal_pullback(RSComplex &this_1,
                                              RSComplex &&arg, RSComplex _d_y,
                                              RSComplex *_d_this,
                                              RSComplex *_d_arg) noexcept;
inline constexpr clad::ValueAndAdjoint<RSComplex &, RSComplex &>
operator_equal_forw(RSComplex &this_1, RSComplex &&arg, RSComplex *_d_this,
                    RSComplex &&_d_arg) noexcept;
void calculate_micro_xs_pullback(double *micro_xs, int nuc, double E,
                                 Input input, const int *n_windows,
                                 const double *pseudo_K0RS,
                                 const Window *windows, const Pole *poles,
                                 int max_num_windows, int max_num_poles,
                                 double *_d_micro_xs, int *_d_nuc, double *_d_E,
                                 Input *_d_input, int *_d_max_num_windows,
                                 int *_d_max_num_poles) __attribute__((device))
{
    bool _cond0;
    int _d_i = 0;
    int i = 0;
    clad::tape<RSComplex> _t1 = {};
    RSComplex _d_PSIIKI = {0., 0.};
    RSComplex PSIIKI = {0., 0.};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_CDUM = {0., 0.};
    RSComplex CDUM = {0., 0.};
    clad::tape<Pole> _t3 = {};
    Pole _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    Pole pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_t1 = {0., 0.};
    RSComplex t1 = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_t2 = {0., 0.};
    RSComplex t2 = {0., 0.};
    clad::tape<RSComplex> _t6 = {};
    clad::tape<RSComplex> _t8 = {};
    RSComplex _d_E_c = {0., 0.};
    RSComplex E_c = {0., 0.};
    clad::tape<RSComplex> _t9 = {};
    double _d_sigT = 0.;
    double sigT;
    double _d_sigA = 0.;
    double sigA;
    double _d_sigF = 0.;
    double sigF;
    double _d_sigE = 0.;
    double sigE;
    double _d_spacing = 0.;
    double spacing = 1. / n_windows[nuc];
    int _d_window = 0;
    int window = (int)(E / spacing);
    {
        _cond0 = window == n_windows[nuc];
        if (_cond0)
            window--;
    }
    RSComplex _d_sigTfactors[4] = {0};
    RSComplex sigTfactors[4] = {{0., /*implicit*/ (double)0}};
    calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors);
    Window _d_w = {0., 0., 0., 0, 0};
    Window w = windows[nuc * max_num_windows + window];
    sigT = E * w.T;
    sigA = E * w.A;
    sigF = E * w.F;
    unsigned long _t0 = 0UL;
    for (i = w.start;; i++)
    {
        {
            if (!(i < w.end))
                break;
        }
        _t0++;
        clad::push(_t1, std::move(PSIIKI)), PSIIKI = {0., 0.};
        clad::push(_t2, std::move(CDUM)), CDUM = {0., 0.};
        clad::push(_t3, std::move(pole)), pole = poles[nuc * max_num_poles + i];
        clad::push(_t4, std::move(t1)), t1 = {0, 1};
        clad::push(_t5, std::move(t2)), t2 = {sqrt(E), 0};
        clad::push(_t6, PSIIKI);
        clad::ValueAndAdjoint<RSComplex &, RSComplex &> _t7 =
            operator_equal_forw(clad::back(_t6),
                                c_div(t1, c_sub(pole.MP_EA, t2)), &_d_PSIIKI,
                                {0., 0.});
        clad::push(_t8, std::move(E_c)), E_c = {E, 0};
        clad::push(_t9, CDUM);
        clad::ValueAndAdjoint<RSComplex &, RSComplex &> _t10 =
            operator_equal_forw(clad::back(_t9), c_div(PSIIKI, E_c), &_d_CDUM,
                                {0., 0.});
        sigT += c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])).r;
        sigA += c_mul(pole.MP_RA, CDUM).r;
        sigF += c_mul(pole.MP_RF, CDUM).r;
    }
    sigE = sigT - sigA;
    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigE;
    {
        double _r_d10 = _d_micro_xs[3];
        _d_micro_xs[3] = 0.;
        _d_sigE += _r_d10;
    }
    {
        double _r_d9 = _d_micro_xs[2];
        _d_micro_xs[2] = 0.;
        _d_sigF += _r_d9;
    }
    {
        double _r_d8 = _d_micro_xs[1];
        _d_micro_xs[1] = 0.;
        _d_sigA += _r_d8;
    }
    {
        double _r_d7 = _d_micro_xs[0];
        _d_micro_xs[0] = 0.;
        _d_sigT += _r_d7;
    }
    {
        double _r_d6 = _d_sigE;
        _d_sigE = 0.;
        _d_sigT += _r_d6;
        _d_sigA += -_r_d6;
    }
    {
        for (;; _t0--)
        {
            {
                if (!_t0)
                    break;
            }
            i--;
            double _r_d5 = _d_sigF;
            double _r_d4 = _d_sigA;
            double _r_d3 = _d_sigT;
            {
                RSComplex _r11 = {0., 0.};
                operator_equal_pullback(clad::back(_t9), c_div(PSIIKI, E_c),
                                        {0., 0.}, &_d_CDUM, &_r11);
                RSComplex _r12 = {0., 0.};
                RSComplex _r13 = {0., 0.};
                c_div_pullback(PSIIKI, E_c, _r11, &_r12, &_r13);
                clad::pop(_t9);
            }
            {
                *_d_E += _d_E_c.r;
                _d_E_c = {0., 0.};
                E_c = clad::pop(_t8);
            }
            {
                RSComplex _r6 = {0., 0.};
                operator_equal_pullback(clad::back(_t6),
                                        c_div(t1, c_sub(pole.MP_EA, t2)),
                                        {0., 0.}, &_d_PSIIKI, &_r6);
                RSComplex _r7 = {0., 0.};
                RSComplex _r8 = {0., 0.};
                c_div_pullback(t1, c_sub(pole.MP_EA, t2), _r6, &_r7, &_r8);
                RSComplex _r9 = {0., 0.};
                RSComplex _r10 = {0., 0.};
                c_sub_pullback(pole.MP_EA, t2, _r8, &_r9, &_r10);
                clad::pop(_t6);
            }
            {
                double _r5 = 0.;
                _r5 +=
                    _d_t2.r * clad::custom_derivatives::sqrt_pushforward(E, 1.)
                                  .pushforward;
                *_d_E += _r5;
                _d_t2 = {0., 0.};
                t2 = clad::pop(_t5);
            }
            {
                _d_t1 = {0., 0.};
                t1 = clad::pop(_t4);
            }
            {
                _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
                pole = clad::pop(_t3);
            }
            {
                _d_CDUM = {0., 0.};
                CDUM = clad::pop(_t2);
            }
            {
                _d_PSIIKI = {0., 0.};
                PSIIKI = clad::pop(_t1);
            }
        }
        _d_w.start += _d_i;
    }
    {
        double _r_d2 = _d_sigF;
        _d_sigF = 0.;
        *_d_E += _r_d2 * w.F;
        _d_w.F += E * _r_d2;
    }
    {
        double _r_d1 = _d_sigA;
        _d_sigA = 0.;
        *_d_E += _r_d1 * w.A;
        _d_w.A += E * _r_d1;
    }
    {
        double _r_d0 = _d_sigT;
        _d_sigT = 0.;
        *_d_E += _r_d0 * w.T;
        _d_w.T += E * _r_d0;
    }
    {
        int _r2 = 0;
        double _r3 = 0.;
        Input _r4 = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0, 0, 0, 0};
        calculate_sig_T_pullback(nuc, E, input, pseudo_K0RS, sigTfactors, &_r2,
                                 &_r3, &_r4, _d_sigTfactors);
        *_d_nuc += _r2;
        *_d_E += _r3;
    }
    {
        *_d_E += _d_window / spacing;
        double _r1 = _d_window * -(E / (spacing * spacing));
        _d_spacing += _r1;
    }
    double _r0 = _d_spacing * -(1. / (n_windows[nuc] * n_windows[nuc]));
}
void calculate_sig_T_pullback(int nuc, double E, Input input,
                              const double *pseudo_K0RS, RSComplex *sigTfactors,
                              int *_d_nuc, double *_d_E, Input *_d_input,
                              RSComplex *_d_sigTfactors) __attribute__((device))
{
    int _d_i = 0;
    int i = 0;
    clad::tape<double> _t1 = {};
    clad::tape<double> _t2 = {};
    clad::tape<bool> _cond0 = {};
    clad::tape<double> _t3 = {};
    clad::tape<bool> _cond1 = {};
    clad::tape<double> _t4 = {};
    clad::tape<double> _t5 = {};
    clad::tape<bool> _cond2 = {};
    clad::tape<double> _t6 = {};
    clad::tape<double> _t7 = {};
    clad::tape<double> _t8 = {};
    double _d_phi = 0.;
    double phi;
    unsigned long _t0 = 0UL;
    for (i = 0;; i++)
    {
        {
            if (!(i < 4))
                break;
        }
        _t0++;
        clad::push(_t1, phi);
        phi = pseudo_K0RS[nuc * input.numL + i] * clad::push(_t2, sqrt(E));
        {
            clad::push(_cond0, i == 1);
            if (clad::back(_cond0))
            {
                clad::push(_t3, phi);
                phi -= -atan(phi);
            }
            else
            {
                clad::push(_cond1, i == 2);
                if (clad::back(_cond1))
                {
                    clad::push(_t4, phi);
                    phi -= atan(3. * phi / clad::push(_t5, (3. - phi * phi)));
                }
                else
                {
                    clad::push(_cond2, i == 3);
                    if (clad::back(_cond2))
                    {
                        clad::push(_t6, phi);
                        phi -= atan(phi * (15. - phi * phi) /
                                    clad::push(_t7, (15. - 6. * phi * phi)));
                    }
                }
            }
        }
        clad::push(_t8, phi);
        phi *= 2.;
        sigTfactors[i].r = cos(phi);
        sigTfactors[i].i = -sin(phi);
    }
    for (;; _t0--)
    {
        {
            if (!_t0)
                break;
        }
        i--;
        {
            double _r_d6 = _d_sigTfactors[i].i;
            _d_sigTfactors[i].i = 0.;
            double _r7 = 0.;
            _r7 +=
                -_r_d6 *
                clad::custom_derivatives::sin_pushforward(phi, 1.).pushforward;
            _d_phi += _r7;
        }
        {
            double _r_d5 = _d_sigTfactors[i].r;
            _d_sigTfactors[i].r = 0.;
            double _r6 = 0.;
            _r6 +=
                _r_d5 *
                clad::custom_derivatives::cos_pushforward(phi, 1.).pushforward;
            _d_phi += _r6;
        }
        {
            phi = clad::pop(_t8);
            double _r_d4 = _d_phi;
            _d_phi = 0.;
            _d_phi += _r_d4 * 2.;
        }
        {
            if (clad::back(_cond0))
            {
                phi = clad::pop(_t3);
                double _r_d1 = _d_phi;
                double _r1 = 0.;
                clad::custom_derivatives::atan_pullback(phi, _r_d1, &_r1);
                _d_phi += _r1;
            }
            else
            {
                if (clad::back(_cond1))
                {
                    phi = clad::pop(_t4);
                    double _r_d2 = _d_phi;
                    double _r2 = 0.;
                    clad::custom_derivatives::atan_pullback(
                        3. * phi / clad::push(_t5, (3. - phi * phi)), -_r_d2,
                        &_r2);
                    _d_phi += 3. * _r2 / clad::push(_t5, (3. - phi * phi));
                    double _r3 = _r2 * -(3. * phi /
                                         (clad::push(_t5, (3. - phi * phi)) *
                                          clad::push(_t5, (3. - phi * phi))));
                    _d_phi += -_r3 * phi;
                    _d_phi += phi * -_r3;
                }
                else
                {
                    if (clad::back(_cond2))
                    {
                        phi = clad::pop(_t6);
                        double _r_d3 = _d_phi;
                        double _r4 = 0.;
                        clad::custom_derivatives::atan_pullback(
                            phi * (15. - phi * phi) /
                                clad::push(_t7, (15. - 6. * phi * phi)),
                            -_r_d3, &_r4);
                        _d_phi += _r4 /
                                  clad::push(_t7, (15. - 6. * phi * phi)) *
                                  (15. - phi * phi);
                        _d_phi += -phi * _r4 /
                                  clad::push(_t7, (15. - 6. * phi * phi)) * phi;
                        _d_phi += phi * -phi * _r4 /
                                  clad::push(_t7, (15. - 6. * phi * phi));
                        double _r5 =
                            _r4 * -(phi * (15. - phi * phi) /
                                    (clad::push(_t7, (15. - 6. * phi * phi)) *
                                     clad::push(_t7, (15. - 6. * phi * phi))));
                        _d_phi += 6. * -_r5 * phi;
                        _d_phi += 6. * phi * -_r5;
                    }
                    clad::pop(_cond2);
                }
                clad::pop(_cond1);
            }
            clad::pop(_cond0);
        }
        {
            phi = clad::pop(_t1);
            double _r_d0 = _d_phi;
            _d_phi = 0.;
            double _r0 = 0.;
            _r0 +=
                pseudo_K0RS[nuc * input.numL + i] * _r_d0 *
                clad::custom_derivatives::sqrt_pushforward(E, 1.).pushforward;
            *_d_E += _r0;
        }
    }
}
void c_sub_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device))
{
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = A.r - B.r;
    C.i = A.i - B.i;
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        (*_d_A).i += _r_d1;
        (*_d_B).i += -_r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        (*_d_A).r += _r_d0;
        (*_d_B).r += -_r_d0;
    }
}
void c_mul_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((host))
__attribute__((device))
{
    double _d_a = 0.;
    double a = A.r;
    double _d_b = 0.;
    double b = A.i;
    double _d_c = 0.;
    double c = B.r;
    double _d_d = 0.;
    double d = B.i;
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = (a * c) - (b * d);
    C.i = (a * d) + (b * c);
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        _d_a += _r_d1 * d;
        _d_d += a * _r_d1;
        _d_b += _r_d1 * c;
        _d_c += b * _r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        _d_a += _r_d0 * c;
        _d_c += a * _r_d0;
        _d_b += -_r_d0 * d;
        _d_d += b * -_r_d0;
    }
    (*_d_B).i += _d_d;
    (*_d_B).r += _d_c;
    (*_d_A).i += _d_b;
    (*_d_A).r += _d_a;
}
void fast_cexp_pullback(RSComplex z, RSComplex _d_y, RSComplex *_d_z)
    __attribute__((device));
void c_add_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device));
void fast_nuclear_W_pullback(RSComplex Z, RSComplex _d_y, RSComplex *_d_Z)
    __attribute__((device))
{
    bool _cond0;
    RSComplex _d_prefactor = {0., 0.};
    RSComplex prefactor = {0., 0.};
    double _d_an[10] = {0};
    clad::array<double> an(10UL);
    double _d_neg_1n[10] = {0};
    clad::array<double> neg_1n(10UL);
    double _d_denominator_left[10] = {0};
    clad::array<double> denominator_left(10UL);
    RSComplex _d_t1 = {0., 0.};
    RSComplex t1 = {0., 0.};
    RSComplex _d_t2 = {0., 0.};
    RSComplex t2 = {0., 0.};
    RSComplex _d_i = {0., 0.};
    RSComplex i = {0., 0.};
    RSComplex _d_one = {0., 0.};
    RSComplex one = {0., 0.};
    RSComplex _d_W = {0., 0.};
    RSComplex W = {0., 0.};
    RSComplex _d_sum = {0., 0.};
    RSComplex sum = {0., 0.};
    unsigned long _t0;
    int _d_n = 0;
    int n = 0;
    clad::tape<RSComplex> _t1 = {};
    RSComplex _d_t3 = {0., 0.};
    RSComplex t3 = {0., 0.};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_top = {0., 0.};
    RSComplex top = {0., 0.};
    clad::tape<RSComplex> _t3 = {};
    RSComplex _d_t4 = {0., 0.};
    RSComplex t4 = {0., 0.};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_t5 = {0., 0.};
    RSComplex t5 = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_bot = {0., 0.};
    RSComplex bot = {0., 0.};
    clad::tape<RSComplex> _t6 = {};
    RSComplex _d_t6 = {0., 0.};
    RSComplex t6 = {0., 0.};
    clad::tape<RSComplex> _t7 = {};
    RSComplex _t9;
    RSComplex _d_a = {0., 0.};
    RSComplex a = {0., 0.};
    RSComplex _d_b = {0., 0.};
    RSComplex b = {0., 0.};
    RSComplex _d_c = {0., 0.};
    RSComplex c = {0., 0.};
    RSComplex _d_d = {0., 0.};
    RSComplex d = {0., 0.};
    RSComplex _d_i0 = {0., 0.};
    RSComplex i0 = {0., 0.};
    RSComplex _d_Z2 = {0., 0.};
    RSComplex Z2 = {0., 0.};
    RSComplex _d_W0 = {0., 0.};
    RSComplex W0 = {0., 0.};
    {
        _cond0 = c_abs(Z) < 6.;
        if (_cond0)
        {
            prefactor = {0, 81.243300000000005};
            an = {0.27584019999999998,
                  0.224574,
                  0.1594149,
                  0.09866577,
                  0.053244140000000002,
                  0.025052149999999999,
                  0.01027747,
                  0.003676164,
                  0.0011464940000000001,
                  3.1175700000000002E-4};
            neg_1n = {-1., 1., -1., 1., -1., 1., -1., 1., -1., 1.};
            denominator_left = {9.8696040000000007, 39.47842,
                                88.826440000000005, 157.91370000000001,
                                246.74010000000001, 355.30579999999998,
                                483.61059999999998, 631.65470000000005,
                                799.43799999999999, 986.96040000000005};
            t1 = {0, 12};
            t2 = {12, 0};
            i = {0, 1};
            one = {1, 0};
            W = c_div(c_mul(i, c_sub(one, fast_cexp(c_mul(t1, Z)))),
                      c_mul(t2, Z));
            sum = {0, 0};
            _t0 = 0UL;
            for (n = 0;; n++)
            {
                {
                    if (!(n < 10))
                        break;
                }
                _t0++;
                clad::push(_t1, std::move(t3)), t3 = {neg_1n[n], 0};
                clad::push(_t2, std::move(top)),
                    top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
                clad::push(_t3, std::move(t4)), t4 = {denominator_left[n], 0};
                clad::push(_t4, std::move(t5)), t5 = {144, 0};
                clad::push(_t5, std::move(bot)),
                    bot = c_sub(t4, c_mul(t5, c_mul(Z, Z)));
                clad::push(_t6, std::move(t6)), t6 = {an[n], 0};
                clad::push(_t7, sum);
                clad::ValueAndAdjoint<RSComplex &, RSComplex &> _t8 =
                    operator_equal_forw(clad::back(_t7),
                                        c_add(sum, c_mul(t6, c_div(top, bot))),
                                        &_d_sum, {0., 0.});
            }
            _t9 = W;
            clad::ValueAndAdjoint<RSComplex &, RSComplex &> _t10 =
                operator_equal_forw(_t9,
                                    c_add(W, c_mul(prefactor, c_mul(Z, sum))),
                                    &_d_W, {0., 0.});
            goto _label0;
        }
        else
        {
            a = {0.51242422475476845, 0};
            b = {0.27525512860841095, 0};
            c = {0.051765358792987826, 0};
            d = {2.7247448713915889, 0};
            i0 = {0, 1};
            Z2 = c_mul(Z, Z);
            W0 = c_mul(c_mul(Z, i0),
                       c_add(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d))));
            goto _label1;
        }
    }
    if (_cond0)
    {
    _label0:;
        {
            RSComplex _r31 = {0., 0.};
            operator_equal_pullback(_t9,
                                    c_add(W, c_mul(prefactor, c_mul(Z, sum))),
                                    {0., 0.}, &_d_W, &_r31);
            RSComplex _r32 = {0., 0.};
            RSComplex _r33 = {0., 0.};
            c_add_pullback(W, c_mul(prefactor, c_mul(Z, sum)), _r31, &_r32,
                           &_r33);
            RSComplex _r34 = {0., 0.};
            RSComplex _r35 = {0., 0.};
            c_mul_pullback(prefactor, c_mul(Z, sum), _r33, &_r34, &_r35);
            RSComplex _r36 = {0., 0.};
            RSComplex _r37 = {0., 0.};
            c_mul_pullback(Z, sum, _r35, &_r36, &_r37);
        }
        for (;; _t0--)
        {
            {
                if (!_t0)
                    break;
            }
            n--;
            {
                RSComplex _r24 = {0., 0.};
                operator_equal_pullback(clad::back(_t7),
                                        c_add(sum, c_mul(t6, c_div(top, bot))),
                                        {0., 0.}, &_d_sum, &_r24);
                RSComplex _r25 = {0., 0.};
                RSComplex _r26 = {0., 0.};
                c_add_pullback(sum, c_mul(t6, c_div(top, bot)), _r24, &_r25,
                               &_r26);
                RSComplex _r27 = {0., 0.};
                RSComplex _r28 = {0., 0.};
                c_mul_pullback(t6, c_div(top, bot), _r26, &_r27, &_r28);
                RSComplex _r29 = {0., 0.};
                RSComplex _r30 = {0., 0.};
                c_div_pullback(top, bot, _r28, &_r29, &_r30);
                clad::pop(_t7);
            }
            {
                _d_an[n] += _d_t6.r;
                _d_t6 = {0., 0.};
                t6 = clad::pop(_t6);
            }
            {
                RSComplex _r18 = {0., 0.};
                RSComplex _r19 = {0., 0.};
                c_sub_pullback(t4, c_mul(t5, c_mul(Z, Z)), _d_bot, &_r18,
                               &_r19);
                RSComplex _r20 = {0., 0.};
                RSComplex _r21 = {0., 0.};
                c_mul_pullback(t5, c_mul(Z, Z), _r19, &_r20, &_r21);
                RSComplex _r22 = {0., 0.};
                RSComplex _r23 = {0., 0.};
                c_mul_pullback(Z, Z, _r21, &_r22, &_r23);
                _d_bot = {0., 0.};
                bot = clad::pop(_t5);
            }
            {
                _d_t5 = {0., 0.};
                t5 = clad::pop(_t4);
            }
            {
                _d_denominator_left[n] += _d_t4.r;
                _d_t4 = {0., 0.};
                t4 = clad::pop(_t3);
            }
            {
                RSComplex _r11 = {0., 0.};
                RSComplex _r17 = {0., 0.};
                c_sub_pullback(c_mul(t3, fast_cexp(c_mul(t1, Z))), one, _d_top,
                               &_r11, &_r17);
                RSComplex _r12 = {0., 0.};
                RSComplex _r13 = {0., 0.};
                c_mul_pullback(t3, fast_cexp(c_mul(t1, Z)), _r11, &_r12, &_r13);
                RSComplex _r14 = {0., 0.};
                fast_cexp_pullback(c_mul(t1, Z), _r13, &_r14);
                RSComplex _r15 = {0., 0.};
                RSComplex _r16 = {0., 0.};
                c_mul_pullback(t1, Z, _r14, &_r15, &_r16);
                _d_top = {0., 0.};
                top = clad::pop(_t2);
            }
            {
                _d_neg_1n[n] += _d_t3.r;
                _d_t3 = {0., 0.};
                t3 = clad::pop(_t1);
            }
        }
        {
            RSComplex _r0 = {0., 0.};
            RSComplex _r8 = {0., 0.};
            c_div_pullback(c_mul(i, c_sub(one, fast_cexp(c_mul(t1, Z)))),
                           c_mul(t2, Z), _d_W, &_r0, &_r8);
            RSComplex _r1 = {0., 0.};
            RSComplex _r2 = {0., 0.};
            c_mul_pullback(i, c_sub(one, fast_cexp(c_mul(t1, Z))), _r0, &_r1,
                           &_r2);
            RSComplex _r3 = {0., 0.};
            RSComplex _r4 = {0., 0.};
            c_sub_pullback(one, fast_cexp(c_mul(t1, Z)), _r2, &_r3, &_r4);
            RSComplex _r5 = {0., 0.};
            fast_cexp_pullback(c_mul(t1, Z), _r4, &_r5);
            RSComplex _r6 = {0., 0.};
            RSComplex _r7 = {0., 0.};
            c_mul_pullback(t1, Z, _r5, &_r6, &_r7);
            RSComplex _r9 = {0., 0.};
            RSComplex _r10 = {0., 0.};
            c_mul_pullback(t2, Z, _r8, &_r9, &_r10);
        }
    }
    else
    {
    _label1:;
        {
            RSComplex _r40 = {0., 0.};
            RSComplex _r43 = {0., 0.};
            c_mul_pullback(
                c_mul(Z, i0),
                c_add(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d))), _d_W0,
                &_r40, &_r43);
            RSComplex _r41 = {0., 0.};
            RSComplex _r42 = {0., 0.};
            c_mul_pullback(Z, i0, _r40, &_r41, &_r42);
            RSComplex _r44 = {0., 0.};
            RSComplex _r49 = {0., 0.};
            c_add_pullback(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d)), _r43,
                           &_r44, &_r49);
            RSComplex _r45 = {0., 0.};
            RSComplex _r46 = {0., 0.};
            c_div_pullback(a, c_sub(Z2, b), _r44, &_r45, &_r46);
            RSComplex _r47 = {0., 0.};
            RSComplex _r48 = {0., 0.};
            c_sub_pullback(Z2, b, _r46, &_r47, &_r48);
            RSComplex _r50 = {0., 0.};
            RSComplex _r51 = {0., 0.};
            c_div_pullback(c, c_sub(Z2, d), _r49, &_r50, &_r51);
            RSComplex _r52 = {0., 0.};
            RSComplex _r53 = {0., 0.};
            c_sub_pullback(Z2, d, _r51, &_r52, &_r53);
        }
        {
            RSComplex _r38 = {0., 0.};
            RSComplex _r39 = {0., 0.};
            c_mul_pullback(Z, Z, _d_Z2, &_r38, &_r39);
        }
    }
}
void c_div_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device))
{
    double _d_a = 0.;
    double a = A.r;
    double _d_b = 0.;
    double b = A.i;
    double _d_c = 0.;
    double c = B.r;
    double _d_d = 0.;
    double d = B.i;
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    double _d_denom = 0.;
    double denom = c * c + d * d;
    C.r = ((a * c) + (b * d)) / denom;
    C.i = ((b * c) - (a * d)) / denom;
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        _d_b += _r_d1 / denom * c;
        _d_c += b * _r_d1 / denom;
        _d_a += -_r_d1 / denom * d;
        _d_d += a * -_r_d1 / denom;
        double _r1 = _r_d1 * -(((b * c) - (a * d)) / (denom * denom));
        _d_denom += _r1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        _d_a += _r_d0 / denom * c;
        _d_c += a * _r_d0 / denom;
        _d_b += _r_d0 / denom * d;
        _d_d += b * _r_d0 / denom;
        double _r0 = _r_d0 * -(((a * c) + (b * d)) / (denom * denom));
        _d_denom += _r0;
    }
    {
        _d_c += _d_denom * c;
        _d_c += c * _d_denom;
        _d_d += _d_denom * d;
        _d_d += d * _d_denom;
    }
    (*_d_B).i += _d_d;
    (*_d_B).r += _d_c;
    (*_d_A).i += _d_b;
    (*_d_A).r += _d_a;
}
inline constexpr void operator_equal_pullback(RSComplex &this_1,
                                              RSComplex &&arg, RSComplex _d_y,
                                              RSComplex *_d_this,
                                              RSComplex *_d_arg) noexcept
{
    this_1.r = arg.r;
    this_1.i = arg.i;
    {
        double _r_d1 = (*_d_this).i;
        (*_d_this).i = 0.;
        (*_d_arg).i += _r_d1;
    }
    {
        double _r_d0 = (*_d_this).r;
        (*_d_this).r = 0.;
        (*_d_arg).r += _r_d0;
    }
}
inline constexpr clad::ValueAndAdjoint<RSComplex &, RSComplex &>
operator_equal_forw(RSComplex &this_1, RSComplex &&arg, RSComplex *_d_this,
                    RSComplex &&_d_arg) noexcept
{
    double _t0 = this_1.r;
    this_1.r = arg.r;
    double _t1 = this_1.i;
    this_1.i = arg.i;
    return {this_1, (*_d_this)};
}
void fast_exp_pullback(double x, double _d_y, double *_d_x)
    __attribute__((device));
void fast_cexp_pullback(RSComplex z, RSComplex _d_y, RSComplex *_d_z)
    __attribute__((device))
{
    double _d_x = 0.;
    double x = z.r;
    double _d_y0 = 0.;
    double y = z.i;
    double _d_t1 = 0.;
    double t1 = fast_exp(x);
    double _d_t2 = 0.;
    double t2 = cos(y);
    double _d_t3 = 0.;
    double t3 = sin(y);
    RSComplex _d_t4 = {0., 0.};
    RSComplex t4 = {t2, t3};
    RSComplex _d_t5 = {0., 0.};
    RSComplex t5 = {t1, 0};
    RSComplex _d_result = {0., 0.};
    RSComplex result = c_mul(t5, t4);
    {
        RSComplex _r3 = {0., 0.};
        RSComplex _r4 = {0., 0.};
        c_mul_pullback(t5, t4, _d_result, &_r3, &_r4);
    }
    _d_t1 += _d_t5.r;
    {
        _d_t2 += _d_t4.r;
        _d_t3 += _d_t4.i;
    }
    {
        double _r2 = 0.;
        _r2 += _d_t3 *
               clad::custom_derivatives::sin_pushforward(y, 1.).pushforward;
        _d_y0 += _r2;
    }
    {
        double _r1 = 0.;
        _r1 += _d_t2 *
               clad::custom_derivatives::cos_pushforward(y, 1.).pushforward;
        _d_y0 += _r1;
    }
    {
        double _r0 = 0.;
        fast_exp_pullback(x, _d_t1, &_r0);
        _d_x += _r0;
    }
    (*_d_z).i += _d_y0;
    (*_d_z).r += _d_x;
}
void c_add_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A,
                    RSComplex *_d_B) __attribute__((device))
{
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = A.r + B.r;
    C.i = A.i + B.i;
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        (*_d_A).i += _r_d1;
        (*_d_B).i += _r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        (*_d_A).r += _r_d0;
        (*_d_B).r += _r_d0;
    }
}
void fast_exp_pullback(double x, double _d_y, double *_d_x)
    __attribute__((device))
{
    x = 1. + x * 2.44140625E-4;
    double _t0 = x;
    x *= x;
    double _t1 = x;
    x *= x;
    double _t2 = x;
    x *= x;
    double _t3 = x;
    x *= x;
    double _t4 = x;
    x *= x;
    double _t5 = x;
    x *= x;
    double _t6 = x;
    x *= x;
    double _t7 = x;
    x *= x;
    double _t8 = x;
    x *= x;
    double _t9 = x;
    x *= x;
    double _t10 = x;
    x *= x;
    double _t11 = x;
    x *= x;
    *_d_x += _d_y;
    {
        x = _t11;
        double _r_d12 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d12 * x;
        *_d_x += x * _r_d12;
    }
    {
        x = _t10;
        double _r_d11 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d11 * x;
        *_d_x += x * _r_d11;
    }
    {
        x = _t9;
        double _r_d10 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d10 * x;
        *_d_x += x * _r_d10;
    }
    {
        x = _t8;
        double _r_d9 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d9 * x;
        *_d_x += x * _r_d9;
    }
    {
        x = _t7;
        double _r_d8 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d8 * x;
        *_d_x += x * _r_d8;
    }
    {
        x = _t6;
        double _r_d7 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d7 * x;
        *_d_x += x * _r_d7;
    }
    {
        x = _t5;
        double _r_d6 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d6 * x;
        *_d_x += x * _r_d6;
    }
    {
        x = _t4;
        double _r_d5 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d5 * x;
        *_d_x += x * _r_d5;
    }
    {
        x = _t3;
        double _r_d4 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d4 * x;
        *_d_x += x * _r_d4;
    }
    {
        x = _t2;
        double _r_d3 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d3 * x;
        *_d_x += x * _r_d3;
    }
    {
        x = _t1;
        double _r_d2 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d2 * x;
        *_d_x += x * _r_d2;
    }
    {
        x = _t0;
        double _r_d1 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d1 * x;
        *_d_x += x * _r_d1;
    }
    {
        double _r_d0 = *_d_x;
        *_d_x = 0.;
        *_d_x += _r_d0 * 2.44140625E-4;
    }
}

void run_event_based_simulation(Input input, SimulationData GSD, unsigned long * vhash_result )
{
	////////////////////////////////////////////////////////////////////////////////
	// Configure & Launch Simulation Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Running baseline event-based simulation on device...\n");

	int nthreads = 256;
	int nblocks = ceil( (double) input.lookups / (double) nthreads);

	xs_lookup_kernel_baseline<<<nblocks, nthreads>>>( input, GSD );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	////////////////////////////////////////////////////////////////////////////////
	// Reduce Verification Results
	////////////////////////////////////////////////////////////////////////////////
	printf("Reducing verification results...\n");

	unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + input.lookups, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	*vhash_result = verification_scalar;

	#ifdef VERIFY
	#if FORWARD

	double *dout = (double*)malloc(sizeof(double));
	gpuErrchk( cudaMemcpy(dout, GSD.dout, sizeof(double), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("dout=%f\n", *dout);

	#else

	size_t sz = 10;//GSD.length_poles;
    Pole here[sz];// = 1.23456;
    gpuErrchk( cudaMemcpy(&here, GSD.d_poles, sz * sizeof(Pole), cudaMemcpyDeviceToHost) );

	for (int i=0; i < sz; ++i)
 	   printf("here[%d]=%f %f %f %f %f %f %f %f\n", i, here[i].MP_EA.r, here[i].MP_EA.i, here[i].MP_RT.r, here[i].MP_RT.i, here[i].MP_RA.r, here[i].MP_RA.i, here[i].MP_RF.r, here[i].MP_RF.i);

	#endif
	#endif
}

// In this kernel, we perform a single lookup with each thread. Threads within a warp
// do not really have any relation to each other, and divergence due to high nuclide count fuel
// material lookups are costly. This kernel constitutes baseline performance.
__global__ void xs_lookup_kernel_baseline(Input in, SimulationData GSD )
{
	// The lookup ID. Used to set the seed, and to store the verification value
	const int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= in.lookups )
		return;

	// Set the initial seed value
	uint64_t seed = STARTING_SEED;	

	// Forward seed to lookup index (we need 2 samples per lookup)
	seed = fast_forward_LCG(seed, 2*i);

	// Randomly pick an energy and material for the particle
	double E = LCG_random_double(&seed);
	int mat  = pick_mat(&seed);

	double macro_xs[4] = {0};

	#ifdef FORWARD
	calculate_macro_xs( macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles, GSD.max_num_windows, GSD.max_num_poles );

	#ifdef VERIFY
	double macro_xs2[4] = {0};
	calculate_macro_xs( macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles, GSD.max_num_windows, GSD.max_num_poles );
	printf("zz=%f %f %f %f\n", macro_xs2[0], macro_xs[0], GSD.poles[0].MP_EA.r , GSD.d_poles[0].MP_EA.r  );
	atomicAdd(GSD.dout, (macro_xs2[0] - macro_xs[0]) / 1e-3 );
	#endif

    #else

	double d_macro_xs[4] = {0};
	d_macro_xs[0] = 1.0;

    // size_t sz = GSD.length_poles * sizeof(Pole);
    // Pole *d_poles = new Pole[GSD.length_poles];
    // memcpy(d_poles, GSD.d_poles, sz);

    // auto grad = clad::gradient(calculate_macro_xs, "macro_xs");
    // grad.execute(macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs,
                //  GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows,
                //  GSD.poles, GSD.max_num_windows, GSD.max_num_poles, d_macro_xs);
	calculate_macro_xs_grad_0(macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs,
		GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows,
		GSD.poles, GSD.max_num_windows, GSD.max_num_poles,
		d_macro_xs);
	// free(d_poles);

    // __enzyme_autodiff((void*)calculate_macro_xs,
	// 			enzyme_dup,   macro_xs, d_macro_xs,
	// 			enzyme_const, mat,
	// 			enzyme_const, E, 
	// 			enzyme_const, in, 
	// 			enzyme_const, GSD.num_nucs, 
	// 			enzyme_const, GSD.mats, 
	// 			enzyme_const, GSD.max_num_nucs, 
	// 			enzyme_const, GSD.concs, 
	// 			enzyme_const, GSD.n_windows, 
	// 			enzyme_const, GSD.pseudo_K0RS, 
	// 			enzyme_const, GSD.windows, 
	// 			//enzyme_const,   GSD.poles,
	// 			enzyme_dup,   GSD.poles, GSD.d_poles,
	// 			enzyme_const, GSD.max_num_windows, 
	// 			enzyme_const, GSD.max_num_poles
	// 		);
    #endif

	// For verification, and to prevent the compiler from optimizing
	// all work out, we interrogate the returned macro_xs_vector array
	// to find its maximum value index, then increment the verification
	// value by that index. In this implementation, we write to a global
	// verification array that will get reduced after this kernel comples.
	double max = -DBL_MAX;
	int max_idx = 0;
	for(int x = 0; x < 4; x++ )
	{
		if( macro_xs[x] > max )
		{
			max = macro_xs[x];
			max_idx = x;
		}
	}
	GSD.verification[i] = max_idx+1;
}

__device__ void calculate_macro_xs( double * macro_xs, int mat, double E, Input input, const int * num_nucs, const int * mats, int max_num_nucs, const double * concs, const int * n_windows, const double * pseudo_K0Rs, const Window * windows, const Pole * poles, int max_num_windows, int max_num_poles ) 
{
	// zero out macro vector
	// for( int i = 0; i < 4; i++ )
	// 	macro_xs[i] = 0;

	// for nuclide in mat
	for( int i = 0; i < num_nucs[mat]; i++ )
	{
		double micro_xs[4] = {0.};
		int nuc = mats[mat * max_num_nucs + i];

		if( input.doppler == 1 )
			calculate_micro_xs_doppler( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
		else
			calculate_micro_xs( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

		for( int j = 0; j < 4; j++ )
		{
			macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
		}
		// Debug
		/*
		printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
		E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
		*/
	}

	// Debug
	/*
	printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
	E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
	*/
}

// No Temperature dependence (i.e., 0K evaluation)
__device__ void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, const Pole * poles, int max_num_windows, int max_num_poles)
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4] = {0.}; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = windows[nuc * max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		RSComplex PSIIKI = {0.0, 0.0};
		RSComplex CDUM = {0.0, 0.0};
		Pole pole = poles[nuc * max_num_poles + i];
		RSComplex t1 = {0, 1};
		RSComplex t2 = {sqrt(E), 0 };
		PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );
		RSComplex E_c = {E, 0};
		CDUM = c_div(PSIIKI, E_c);
		sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;
		sigA += (c_mul( pole.MP_RA, CDUM)).r;
		sigF += (c_mul(pole.MP_RF, CDUM)).r;
	}

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
}

// Temperature Dependent Variation of Kernel
// (This involves using the Complex Faddeeva function to
// Doppler broaden the poles within the window)
__device__ void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, const Pole * poles, int max_num_windows, int max_num_poles )
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = windows[nuc * max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	double dopp = 0.5;

	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		Pole pole = poles[nuc * max_num_poles + i];

		// Prep Z
		RSComplex E_c = {E, 0};
		RSComplex dopp_c = {dopp, 0};
		RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

		// Evaluate Fadeeva Function
		RSComplex faddeeva = fast_nuclear_W( Z );

		// Update W
		sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;
		sigA += (c_mul( pole.MP_RA , faddeeva)).r;
		sigF += (c_mul( pole.MP_RF , faddeeva)).r;
	}

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
}

// picks a material based on a probabilistic distribution
__device__ int pick_mat( uint64_t * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a 
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies

	double roll = LCG_random_double(seed);

	// makes a pick based on the distro
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}

	return 0;
}

__device__ void calculate_sig_T( int nuc, double E, Input input, const double * pseudo_K0RS, RSComplex * sigTfactors )
{
	double phi;

	for( int i = 0; i < 4; i++ )
	{
		phi = pseudo_K0RS[nuc * input.numL + i] * sqrt(E);

		if( i == 1 )
			phi -= - atan( phi );
		else if( i == 2 )
			phi -= atan( 3.0 * phi / (3.0 - phi*phi));
		else if( i == 3 )
			phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));

		phi *= 2.0;

		sigTfactors[i].r = cos(phi);
		sigTfactors[i].i = -sin(phi);
	}
}

// This function uses a combination of the Abrarov Approximation
// and the QUICK_W three term asymptotic expansion.
// Only expected to use Abrarov ~0.5% of the time.
__device__ RSComplex fast_nuclear_W( RSComplex Z )
{
	// Abrarov 
	if( c_abs(Z) < 6.0 )
	{
		// Precomputed parts for speeding things up
		// (N = 10, Tm = 12.0)
		RSComplex prefactor = {0, 8.124330e+01};
		double an[10] = {
			2.758402e-01,
			2.245740e-01,
			1.594149e-01,
			9.866577e-02,
			5.324414e-02,
			2.505215e-02,
			1.027747e-02,
			3.676164e-03,
			1.146494e-03,
			3.117570e-04
		};
		double neg_1n[10] = {
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0
		};

		double denominator_left[10] = {
			9.869604e+00,
			3.947842e+01,
			8.882644e+01,
			1.579137e+02,
			2.467401e+02,
			3.553058e+02,
			4.836106e+02,
			6.316547e+02,
			7.994380e+02,
			9.869604e+02
		};

		RSComplex t1 = {0, 12};
		RSComplex t2 = {12, 0};
		RSComplex i = {0,1};
		RSComplex one = {1, 0};
		RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));
		RSComplex sum = {0,0};
		for( int n = 0; n < 10; n++ )
		{
			RSComplex t3 = {neg_1n[n], 0};
			RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
			RSComplex t4 = {denominator_left[n], 0};
			RSComplex t5 = {144, 0};
			RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));
			RSComplex t6 = {an[n], 0};
			sum = c_add(sum, c_mul(t6, c_div(top,bot)));
		}
		W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
		return W;
	}
	else
	{
		// QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
		// Pre-computed parameters
		RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};
		RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};
		RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};
		RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};

		RSComplex i = {0,1};
		RSComplex Z2 = c_mul(Z, Z);
		// Three Term Asymptotic Expansion
		RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));

		return W;
	}
}

__host__ __device__ double LCG_random_double(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}	

__host__ __device__ uint64_t LCG_random_int(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return *seed;
}	

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0) 
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;
}

// Complex arithmetic functions

__device__ RSComplex c_add( RSComplex A, RSComplex B)
{
    RSComplex C;
    C.r = A.r + B.r;
	C.i = A.i + B.i;
	return C;
}

__device__ RSComplex c_sub( RSComplex A, RSComplex B)
{
    RSComplex C;
    C.r = A.r - B.r;
	C.i = A.i - B.i;
	return C;
}

__host__ __device__ RSComplex c_mul( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
    RSComplex C;
    C.r = (a*c) - (b*d);
	C.i = (a*d) + (b*c);
	return C;
}

__device__ RSComplex c_div( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
    RSComplex C;
    double denom = c*c + d*d;
	C.r = ( (a*c) + (b*d) ) / denom;
	C.i = ( (b*c) - (a*d) ) / denom;
	return C;
}

__device__ double c_abs( RSComplex A)
{
	return sqrt(A.r*A.r + A.i * A.i);
}


// Fast (but inaccurate) exponential function
// Written By "ACMer":
// https://codingforspeed.com/using-faster-exponential-approximation/
// We use our own to avoid small differences in compiler specific
// exp() intrinsic implementations that make it difficult to verify
// if the code is working correctly or not.
__device__ double fast_exp(double x)
{
  x = 1.0 + x * 0.000244140625;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

// Implementation based on:
// z = x + iy
// cexp(z) = e^x * (cos(y) + i * sin(y))
__device__ RSComplex fast_cexp( RSComplex z )
{
	double x = z.r;
	double y = z.i;

	// For consistency across architectures, we
	// will use our own exponetial implementation
	//double t1 = exp(x);
	double t1 = fast_exp(x);
	double t2 = cos(y);
	double t3 = sin(y);
	RSComplex t4 = {t2, t3};
	RSComplex t5 = {t1, 0};
	RSComplex result = c_mul(t5, (t4));
	return result;
}	

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED VARIANT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// This section contains a number of optimized variants of some of the above
// functions, which each deploy a different combination of optimizations strategies
// specific to GPU. By default, RSBench will not run any of these variants. They
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
// Optimization 6 -- Kernel Splitting + All Material Lookups + Full Sort
//                   + Energy Sort
////////////////////////////////////////////////////////////////////////////////////
// This optimization builds on optimization 4, adding in a second sort by energy.
// It is extremely fast, as now most of the threads within a warp will be hitting
// the same indices in the lookup grids. This greatly reduces thread divergence and
// greatly improves cache efficiency and re-use.
//
// However, it is unlikely that this exact optimization would be possible in a real
// application like OpenMC. One major difference is that particle objects are quite
// large, often having 50+ variable fields, such that sorting them in memory becomes
// rather expensive. Instead, the best possible option would probably be to create
// intermediate indexing (per Hamilton et. al 2019), and run the kernels indirectly.
////////////////////////////////////////////////////////////////////////////////////

__global__ void sampling_kernel(Input in, SimulationData GSD )
{
	// The lookup ID.
	const int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= in.lookups )
		return;

	// Set the initial seed value
	uint64_t seed = STARTING_SEED;	

	// Forward seed to lookup index (we need 2 samples per lookup)
	seed = fast_forward_LCG(seed, 2*i);

	// Randomly pick an energy and material for the particle
	double p_energy = LCG_random_double(&seed);
	int mat         = pick_mat(&seed); 

	// Store sample data in state array
	GSD.p_energy_samples[i] = p_energy;
	GSD.mat_samples[i] = mat;
}

__global__ void xs_lookup_kernel_optimization_1(Input in, SimulationData GSD, int m, int n_lookups, int offset )
{
	// The lookup ID. Used to set the seed, and to store the verification value
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= n_lookups )
		return;

	i += offset;

	// Check that our material type matches the kernel material
	int mat = GSD.mat_samples[i];
	if( mat != m )
		return;
	
	double macro_xs[4] = {0};

	calculate_macro_xs( macro_xs, mat, GSD.p_energy_samples[i], in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles, GSD.max_num_windows, GSD.max_num_poles );

	// For verification, and to prevent the compiler from optimizing
	// all work out, we interrogate the returned macro_xs_vector array
	// to find its maximum value index, then increment the verification
	// value by that index. In this implementation, we write to a global
	// verification array that will get reduced after this kernel comples.
	double max = -DBL_MAX;
	int max_idx = 0;
	for(int x = 0; x < 4; x++ )
	{
		if( macro_xs[x] > max )
		{
			max = macro_xs[x];
			max_idx = x;
		}
	}
	GSD.verification[i] = max_idx+1;
}

void run_event_based_simulation_optimization_1(Input in, SimulationData GSD, unsigned long * vhash_result)
{
	const char * optimization_name = "Optimization 1 - Material & Energy Sorts + Material-specific Kernels";
	
	printf("Simulation Kernel:\"%s\"\n", optimization_name);
	
	////////////////////////////////////////////////////////////////////////////////
	// Allocate Additional Data Structures Needed by Optimized Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Allocating additional device data required by kernel...\n");
	size_t sz;
	size_t total_sz = 0;

	sz = in.lookups * sizeof(double);
	gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
	total_sz += sz;
	GSD.length_p_energy_samples = in.lookups;

	sz = in.lookups * sizeof(int);
	gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
	total_sz += sz;
	GSD.length_mat_samples = in.lookups;
	
	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

	////////////////////////////////////////////////////////////////////////////////
	// Configure & Launch Simulation Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Beginning optimized simulation...\n");

	int nthreads = 32;
	int nblocks = ceil( (double) in.lookups / 32.0);
	
	sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Count the number of fuel material lookups that need to be performed (fuel id = 0)
	int n_lookups_per_material[12];
	for( int m = 0; m < 12; m++ )
		n_lookups_per_material[m] = thrust::count(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, m);

	// Sort by material first
	thrust::sort_by_key(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, GSD.p_energy_samples);

	// Now, sort each material by energy
	int offset = 0;
	for( int m = 0; m < 12; m++ )
	{
		thrust::sort_by_key(thrust::device, GSD.p_energy_samples + offset, GSD.p_energy_samples + offset + n_lookups_per_material[m], GSD.mat_samples + offset);
		offset += n_lookups_per_material[m];
	}
	
	// Launch all material kernels individually
	offset = 0;
	for( int m = 0; m < 12; m++ )
	{
		nthreads = 32;
		nblocks = ceil((double) n_lookups_per_material[m] / (double) nthreads);
		xs_lookup_kernel_optimization_1<<<nblocks, nthreads>>>( in, GSD, m, n_lookups_per_material[m], offset );
		offset += n_lookups_per_material[m];
	}
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	////////////////////////////////////////////////////////////////////////////////
	// Reduce Verification Results
	////////////////////////////////////////////////////////////////////////////////
	printf("Reducing verification results...\n");

	unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	*vhash_result = verification_scalar;
}