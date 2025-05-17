#include "clad/Differentiator/Differentiator.h"
#include "lulesh.h"

__attribute__((device)) __attribute__((always_inline)) static inline void ApplyMaterialPropertiesForElems_device_pullback(Real_t eosvmin, Real_t eosvmax, const Real_t *__restrict vnew, const Real_t *__restrict v, Real_t &vnewc, const Index_t *__restrict bad_vol, Index_t zn, Real_t *_d_eosvmin, Real_t *_d_eosvmax, Real_t *_d_vnew, Real_t *_d_v, Real_t *_d_vnewc, Index_t *_d_bad_vol, Index_t *_d_zn) {
    bool _cond0;
    bool _cond1;
    bool _cond2;
    bool _cond3;
    bool _cond4;
    bool _cond5;
    bool _cond6;
    bool _cond7;
    bool _cond8;
    vnewc = vnew[zn];
    {
        _cond0 = eosvmin != Real_t(0.);
        if (_cond0) {
            {
                _cond1 = vnewc < eosvmin;
                if (_cond1)
                    vnewc = eosvmin;
            }
        }
    }
    {
        _cond2 = eosvmax != Real_t(0.);
        if (_cond2) {
            {
                _cond3 = vnewc > eosvmax;
                if (_cond3)
                    vnewc = eosvmax;
            }
        }
    }
    Real_t _d_vc = 0.;
    Real_t vc = v[zn];
    {
        _cond4 = eosvmin != Real_t(0.);
        if (_cond4) {
            {
                _cond5 = vc < eosvmin;
                if (_cond5)
                    vc = eosvmin;
            }
        }
    }
    {
        _cond6 = eosvmax != Real_t(0.);
        if (_cond6) {
            {
                _cond7 = vc > eosvmax;
                if (_cond7)
                    vc = eosvmax;
            }
        }
    }
    {
        _cond8 = vc <= 0.;
        if (_cond8) {
            const_cast<Index_t &>(*bad_vol) = zn;
        }
    }
    if (_cond8) {
        {
            Index_t _r_d5 = *_d_bad_vol;
            *_d_bad_vol = 0;
            *_d_zn += _r_d5;
        }
    }
    if (_cond6) {
        if (_cond7) {
            Real_t _r_d4 = _d_vc;
            _d_vc = 0.;
            *_d_eosvmax += _r_d4;
        }
    }
    if (_cond4) {
        if (_cond5) {
            Real_t _r_d3 = _d_vc;
            _d_vc = 0.;
            *_d_eosvmin += _r_d3;
        }
    }
    _d_v[zn] += _d_vc;
    if (_cond2) {
        if (_cond3) {
            Real_t _r_d2 = *_d_vnewc;
            *_d_vnewc = 0.;
            *_d_eosvmax += _r_d2;
        }
    }
    if (_cond0) {
        if (_cond1) {
            Real_t _r_d1 = *_d_vnewc;
            *_d_vnewc = 0.;
            *_d_eosvmin += _r_d1;
        }
    }
    {
        Real_t _r_d0 = *_d_vnewc;
        *_d_vnewc = 0.;
        _d_vnew[zn] += _r_d0;
    }
}
__attribute__((device)) inline void giveMyRegion_pullback(const Index_t *regCSR, const Index_t i, const Index_t numReg, Index_t _d_y, Index_t *_d_regCSR, Index_t *_d_i, Index_t *_d_numReg) {
    Index_t _d_reg = 0;
    Index_t reg = 0;
    clad::tape<bool> _cond0 = {};
    unsigned long _t0 = 0UL;
    for (reg = 0; ; reg++) {
        {
            if (!(reg < numReg - 1))
                break;
        }
        _t0++;
        {
            clad::push(_cond0, i < regCSR[reg]);
            if (clad::back(_cond0))
                goto _label0;
        }
    }
    *_d_numReg += _d_y;
    for (;; _t0--) {
        {
            if (!_t0)
                break;
        }
        reg--;
        if (clad::back(_cond0))
          _label0:
            _d_reg += _d_y;
        clad::pop(_cond0);
    }
}
__attribute__((device)) __attribute__((host)) inline void FABS_pullback(real8 arg, real8 _d_y, real8 *_d_arg) {
    {
        real8 _r0 = 0.;
        _r0 += _d_y * clad::custom_derivatives::fabs_pushforward(arg, 1.).pushforward;
        *_d_arg += _r0;
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void CalcPressureForElems_device_pullback(Real_t &p_new, Real_t &bvc, Real_t &pbvc, Real_t e_old, Real_t compression, Real_t vnewc, Real_t pmin, Real_t p_cut, Real_t eosvmax, Real_t *_d_p_new, Real_t *_d_bvc, Real_t *_d_pbvc, Real_t *_d_e_old, Real_t *_d_compression, Real_t *_d_vnewc, Real_t *_d_pmin, Real_t *_d_p_cut, Real_t *_d_eosvmax) {
    bool _cond0;
    Real_t _t1;
    bool _cond1;
    Real_t _t2;
    bool _cond2;
    Real_t _t3;
    Real_t _t0 = Real_t(3.);
    Real_t _d_c1s = 0.;
    Real_t c1s = Real_t(2.) / _t0;
    Real_t _d_p_temp = 0.;
    Real_t p_temp = p_new;
    bvc = c1s * (compression + Real_t(1.));
    pbvc = c1s;
    p_temp = bvc * e_old;
    {
        _cond0 = FABS(p_temp) < p_cut;
        if (_cond0) {
            _t1 = p_temp;
            p_temp = Real_t(0.);
        }
    }
    {
        _cond1 = vnewc >= eosvmax;
        if (_cond1) {
            _t2 = p_temp;
            p_temp = Real_t(0.);
        }
    }
    {
        _cond2 = p_temp < pmin;
        if (_cond2) {
            _t3 = p_temp;
            p_temp = pmin;
        }
    }
    p_new = p_temp;
    {
        Real_t _r_d6 = *_d_p_new;
        *_d_p_new = 0.;
        _d_p_temp += _r_d6;
    }
    if (_cond2) {
        p_temp = _t3;
        Real_t _r_d5 = _d_p_temp;
        _d_p_temp = 0.;
        *_d_pmin += _r_d5;
    }
    if (_cond1) {
        p_temp = _t2;
        Real_t _r_d4 = _d_p_temp;
        _d_p_temp = 0.;
    }
    if (_cond0) {
        p_temp = _t1;
        Real_t _r_d3 = _d_p_temp;
        _d_p_temp = 0.;
    }
    {
        Real_t _r_d2 = _d_p_temp;
        _d_p_temp = 0.;
        *_d_bvc += _r_d2 * e_old;
        *_d_e_old += bvc * _r_d2;
    }
    {
        Real_t _r_d1 = *_d_pbvc;
        *_d_pbvc = 0.;
        _d_c1s += _r_d1;
    }
    {
        Real_t _r_d0 = *_d_bvc;
        *_d_bvc = 0.;
        _d_c1s += _r_d0 * (compression + Real_t(1.));
        *_d_compression += c1s * _r_d0;
    }
    *_d_p_new += _d_p_temp;
    Real_t _r0 = _d_c1s * -(Real_t(2.) / (_t0 * _t0));
}
__attribute__((device)) inline void SQRT_pullback(real8 arg, real8 _d_y, real8 *_d_arg) {
    {
        real8 _r0 = 0.;
        _r0 += _d_y * clad::custom_derivatives::sqrt_pushforward(arg, 1.).pushforward;
        *_d_arg += _r0;
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void CalcEnergyForElems_device_pullback(Real_t &p_new, Real_t &e_new, Real_t &q_new, Real_t &bvc, Real_t &pbvc, Real_t p_old, Real_t e_old, Real_t q_old, Real_t compression, Real_t compHalfStep, Real_t vnewc, Real_t work, Real_t delvc, Real_t pmin, Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin, Real_t qq, Real_t ql, Real_t rho0, Real_t eosvmax, Index_t length, Real_t *_d_p_new, Real_t *_d_e_new, Real_t *_d_q_new, Real_t *_d_bvc, Real_t *_d_pbvc, Real_t *_d_p_old, Real_t *_d_e_old, Real_t *_d_q_old, Real_t *_d_compression, Real_t *_d_compHalfStep, Real_t *_d_vnewc, Real_t *_d_work, Real_t *_d_delvc, Real_t *_d_pmin, Real_t *_d_p_cut, Real_t *_d_e_cut, Real_t *_d_q_cut, Real_t *_d_emin, Real_t *_d_qq, Real_t *_d_ql, Real_t *_d_rho0, Real_t *_d_eosvmax, Index_t *_d_length) {
    bool _cond0;
    bool _cond1;
    Real_t _d_ssc = 0.;
    Real_t ssc = 0.;
    bool _cond2;
    Real_t _t5;
    bool _cond3;
    Real_t _t7;
    bool _cond4;
    Real_t _t8;
    bool _cond5;
    Real_t _d_ssc0 = 0.;
    Real_t ssc0 = 0.;
    bool _cond6;
    Real_t _t12;
    bool _cond7;
    Real_t _t14;
    bool _cond8;
    Real_t _t15;
    bool _cond9;
    Real_t _d_ssc1 = 0.;
    Real_t ssc1 = 0.;
    bool _cond10;
    Real_t _t19;
    Real_t _t20;
    bool _cond11;
    Real_t _t21;
    Real_t _t0 = Real_t(6.);
    Real_t _d_sixth = 0.;
    const Real_t sixth = Real_t(1.) / _t0;
    Real_t _d_pHalfStep = 0.;
    Real_t pHalfStep;
    e_new = e_old - Real_t(0.5) * delvc * (p_old + q_old) + Real_t(0.5) * work;
    {
        _cond0 = e_new < emin;
        if (_cond0) {
            e_new = emin;
        }
    }
    Real_t _t1 = pHalfStep;
    Real_t _t2 = bvc;
    Real_t _t3 = pbvc;
    CalcPressureForElems_device(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc, pmin, p_cut, eosvmax);
    Real_t _t4 = (Real_t(1.) + compHalfStep);
    Real_t _d_vhalf = 0.;
    Real_t vhalf = Real_t(1.) / _t4;
    {
        _cond1 = delvc > Real_t(0.);
        if (_cond1) {
            q_new = Real_t(0.);
        } else {
            ssc = (pbvc * e_new + vhalf * vhalf * bvc * pHalfStep) / rho0;
            {
                _cond2 = ssc <= Real_t(1.1111110000000001E-37);
                if (_cond2) {
                    ssc = Real_t(3.333333E-19);
                } else {
                    _t5 = ssc;
                    ssc = SQRT(ssc);
                }
            }
            q_new = (ssc * ql + qq);
        }
    }
    Real_t _t6 = e_new;
    e_new = e_new + Real_t(0.5) * delvc * (Real_t(3.) * (p_old + q_old) - Real_t(4.) * (pHalfStep + q_new));
    e_new += Real_t(0.5) * work;
    {
        _cond3 = FABS(e_new) < e_cut;
        if (_cond3) {
            _t7 = e_new;
            e_new = Real_t(0.);
        }
    }
    {
        _cond4 = e_new < emin;
        if (_cond4) {
            _t8 = e_new;
            e_new = emin;
        }
    }
    Real_t _t9 = p_new;
    Real_t _t10 = bvc;
    Real_t _t11 = pbvc;
    CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut, eosvmax);
    Real_t _d_q_tilde = 0.;
    Real_t q_tilde;
    {
        _cond5 = delvc > Real_t(0.);
        if (_cond5) {
            q_tilde = Real_t(0.);
        } else {
            ssc0 = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
            {
                _cond6 = ssc0 <= Real_t(1.1111110000000001E-37);
                if (_cond6) {
                    ssc0 = Real_t(3.333333E-19);
                } else {
                    _t12 = ssc0;
                    ssc0 = SQRT(ssc0);
                }
            }
            q_tilde = (ssc0 * ql + qq);
        }
    }
    Real_t _t13 = e_new;
    e_new = e_new - (Real_t(7.) * (p_old + q_old) - Real_t(8.) * (pHalfStep + q_new) + (p_new + q_tilde)) * delvc * sixth;
    {
        _cond7 = FABS(e_new) < e_cut;
        if (_cond7) {
            _t14 = e_new;
            e_new = Real_t(0.);
        }
    }
    {
        _cond8 = e_new < emin;
        if (_cond8) {
            _t15 = e_new;
            e_new = emin;
        }
    }
    Real_t _t16 = p_new;
    Real_t _t17 = bvc;
    Real_t _t18 = pbvc;
    CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut, eosvmax);
    {
        _cond9 = delvc <= Real_t(0.);
        if (_cond9) {
            ssc1 = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
            {
                _cond10 = ssc1 <= Real_t(1.1111110000000001E-37);
                if (_cond10) {
                    ssc1 = Real_t(3.333333E-19);
                } else {
                    _t19 = ssc1;
                    ssc1 = SQRT(ssc1);
                }
            }
            _t20 = q_new;
            q_new = (ssc1 * ql + qq);
            {
                _cond11 = FABS(q_new) < q_cut;
                if (_cond11) {
                    _t21 = q_new;
                    q_new = Real_t(0.);
                }
            }
        }
    }
    if (_cond9) {
        if (_cond11) {
            q_new = _t21;
            Real_t _r_d20 = *_d_q_new;
            *_d_q_new = 0.;
        }
        {
            q_new = _t20;
            Real_t _r_d19 = *_d_q_new;
            *_d_q_new = 0.;
            _d_ssc1 += _r_d19 * ql;
            *_d_ql += ssc1 * _r_d19;
            *_d_qq += _r_d19;
        }
        if (_cond10) {
            {
                Real_t _r_d17 = _d_ssc1;
                _d_ssc1 = 0.;
            }
        } else {
            {
                ssc1 = _t19;
                Real_t _r_d18 = _d_ssc1;
                _d_ssc1 = 0.;
                Real_t _r25 = 0.;
                SQRT_pullback(ssc1, _r_d18, &_r25);
                _d_ssc1 += _r25;
            }
        }
        {
            *_d_pbvc += _d_ssc1 / rho0 * e_new;
            *_d_e_new += pbvc * _d_ssc1 / rho0;
            *_d_vnewc += _d_ssc1 / rho0 * p_new * bvc * vnewc;
            *_d_vnewc += vnewc * _d_ssc1 / rho0 * p_new * bvc;
            *_d_bvc += vnewc * vnewc * _d_ssc1 / rho0 * p_new;
            *_d_p_new += vnewc * vnewc * bvc * _d_ssc1 / rho0;
            Real_t _r24 = _d_ssc1 * -((pbvc * e_new + vnewc * vnewc * bvc * p_new) / (rho0 * rho0));
            *_d_rho0 += _r24;
        }
    }
    {
        p_new = _t16;
        bvc = _t17;
        pbvc = _t18;
        Real_t _r18 = 0.;
        Real_t _r19 = 0.;
        Real_t _r20 = 0.;
        Real_t _r21 = 0.;
        Real_t _r22 = 0.;
        Real_t _r23 = 0.;
        CalcPressureForElems_device_pullback(_t16, _t17, _t18, e_new, compression, vnewc, pmin, p_cut, eosvmax, &*_d_p_new, &*_d_bvc, &*_d_pbvc, &_r18, &_r19, &_r20, &_r21, &_r22, &_r23);
        *_d_e_new += _r18;
        *_d_compression += _r19;
        *_d_vnewc += _r20;
        *_d_pmin += _r21;
        *_d_p_cut += _r22;
        *_d_eosvmax += _r23;
    }
    if (_cond8) {
        {
            e_new = _t15;
            Real_t _r_d16 = *_d_e_new;
            *_d_e_new = 0.;
            *_d_emin += _r_d16;
        }
    }
    if (_cond7) {
        {
            e_new = _t14;
            Real_t _r_d15 = *_d_e_new;
            *_d_e_new = 0.;
        }
    }
    {
        e_new = _t13;
        Real_t _r_d14 = *_d_e_new;
        *_d_e_new = 0.;
        *_d_e_new += _r_d14;
        *_d_p_old += Real_t(7.) * -_r_d14 * sixth * delvc;
        *_d_q_old += Real_t(7.) * -_r_d14 * sixth * delvc;
        _d_pHalfStep += Real_t(8.) * -(-_r_d14 * sixth * delvc);
        *_d_q_new += Real_t(8.) * -(-_r_d14 * sixth * delvc);
        *_d_p_new += -_r_d14 * sixth * delvc;
        _d_q_tilde += -_r_d14 * sixth * delvc;
        *_d_delvc += (Real_t(7.) * (p_old + q_old) - Real_t(8.) * (pHalfStep + q_new) + (p_new + q_tilde)) * -_r_d14 * sixth;
        _d_sixth += (Real_t(7.) * (p_old + q_old) - Real_t(8.) * (pHalfStep + q_new) + (p_new + q_tilde)) * delvc * -_r_d14;
    }
    if (_cond5) {
        {
            Real_t _r_d10 = _d_q_tilde;
            _d_q_tilde = 0.;
        }
    } else {
        {
            Real_t _r_d13 = _d_q_tilde;
            _d_q_tilde = 0.;
            _d_ssc0 += _r_d13 * ql;
            *_d_ql += ssc0 * _r_d13;
            *_d_qq += _r_d13;
        }
        if (_cond6) {
            {
                Real_t _r_d11 = _d_ssc0;
                _d_ssc0 = 0.;
            }
        } else {
            {
                ssc0 = _t12;
                Real_t _r_d12 = _d_ssc0;
                _d_ssc0 = 0.;
                Real_t _r17 = 0.;
                SQRT_pullback(ssc0, _r_d12, &_r17);
                _d_ssc0 += _r17;
            }
        }
        {
            *_d_pbvc += _d_ssc0 / rho0 * e_new;
            *_d_e_new += pbvc * _d_ssc0 / rho0;
            *_d_vnewc += _d_ssc0 / rho0 * p_new * bvc * vnewc;
            *_d_vnewc += vnewc * _d_ssc0 / rho0 * p_new * bvc;
            *_d_bvc += vnewc * vnewc * _d_ssc0 / rho0 * p_new;
            *_d_p_new += vnewc * vnewc * bvc * _d_ssc0 / rho0;
            Real_t _r16 = _d_ssc0 * -((pbvc * e_new + vnewc * vnewc * bvc * p_new) / (rho0 * rho0));
            *_d_rho0 += _r16;
        }
    }
    {
        p_new = _t9;
        bvc = _t10;
        pbvc = _t11;
        Real_t _r10 = 0.;
        Real_t _r11 = 0.;
        Real_t _r12 = 0.;
        Real_t _r13 = 0.;
        Real_t _r14 = 0.;
        Real_t _r15 = 0.;
        CalcPressureForElems_device_pullback(_t9, _t10, _t11, e_new, compression, vnewc, pmin, p_cut, eosvmax, &*_d_p_new, &*_d_bvc, &*_d_pbvc, &_r10, &_r11, &_r12, &_r13, &_r14, &_r15);
        *_d_e_new += _r10;
        *_d_compression += _r11;
        *_d_vnewc += _r12;
        *_d_pmin += _r13;
        *_d_p_cut += _r14;
        *_d_eosvmax += _r15;
    }
    if (_cond4) {
        {
            e_new = _t8;
            Real_t _r_d9 = *_d_e_new;
            *_d_e_new = 0.;
            *_d_emin += _r_d9;
        }
    }
    if (_cond3) {
        {
            e_new = _t7;
            Real_t _r_d8 = *_d_e_new;
            *_d_e_new = 0.;
        }
    }
    {
        Real_t _r_d7 = *_d_e_new;
        *_d_work += Real_t(0.5) * _r_d7;
    }
    {
        e_new = _t6;
        Real_t _r_d6 = *_d_e_new;
        *_d_e_new = 0.;
        *_d_e_new += _r_d6;
        *_d_delvc += Real_t(0.5) * _r_d6 * (Real_t(3.) * (p_old + q_old) - Real_t(4.) * (pHalfStep + q_new));
        *_d_p_old += Real_t(3.) * Real_t(0.5) * delvc * _r_d6;
        *_d_q_old += Real_t(3.) * Real_t(0.5) * delvc * _r_d6;
        _d_pHalfStep += Real_t(4.) * -Real_t(0.5) * delvc * _r_d6;
        *_d_q_new += Real_t(4.) * -Real_t(0.5) * delvc * _r_d6;
    }
    if (_cond1) {
        {
            Real_t _r_d2 = *_d_q_new;
            *_d_q_new = 0.;
        }
    } else {
        {
            Real_t _r_d5 = *_d_q_new;
            *_d_q_new = 0.;
            _d_ssc += _r_d5 * ql;
            *_d_ql += ssc * _r_d5;
            *_d_qq += _r_d5;
        }
        if (_cond2) {
            {
                Real_t _r_d3 = _d_ssc;
                _d_ssc = 0.;
            }
        } else {
            {
                ssc = _t5;
                Real_t _r_d4 = _d_ssc;
                _d_ssc = 0.;
                Real_t _r9 = 0.;
                SQRT_pullback(ssc, _r_d4, &_r9);
                _d_ssc += _r9;
            }
        }
        {
            *_d_pbvc += _d_ssc / rho0 * e_new;
            *_d_e_new += pbvc * _d_ssc / rho0;
            _d_vhalf += _d_ssc / rho0 * pHalfStep * bvc * vhalf;
            _d_vhalf += vhalf * _d_ssc / rho0 * pHalfStep * bvc;
            *_d_bvc += vhalf * vhalf * _d_ssc / rho0 * pHalfStep;
            _d_pHalfStep += vhalf * vhalf * bvc * _d_ssc / rho0;
            Real_t _r8 = _d_ssc * -((pbvc * e_new + vhalf * vhalf * bvc * pHalfStep) / (rho0 * rho0));
            *_d_rho0 += _r8;
        }
    }
    {
        Real_t _r7 = _d_vhalf * -(Real_t(1.) / (_t4 * _t4));
        *_d_compHalfStep += _r7;
    }
    {
        pHalfStep = _t1;
        bvc = _t2;
        pbvc = _t3;
        Real_t _r1 = 0.;
        Real_t _r2 = 0.;
        Real_t _r3 = 0.;
        Real_t _r4 = 0.;
        Real_t _r5 = 0.;
        Real_t _r6 = 0.;
        CalcPressureForElems_device_pullback(_t1, _t2, _t3, e_new, compHalfStep, vnewc, pmin, p_cut, eosvmax, &_d_pHalfStep, &*_d_bvc, &*_d_pbvc, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
        *_d_e_new += _r1;
        *_d_compHalfStep += _r2;
        *_d_vnewc += _r3;
        *_d_pmin += _r4;
        *_d_p_cut += _r5;
        *_d_eosvmax += _r6;
    }
    if (_cond0) {
        {
            Real_t _r_d1 = *_d_e_new;
            *_d_e_new = 0.;
            *_d_emin += _r_d1;
        }
    }
    {
        Real_t _r_d0 = *_d_e_new;
        *_d_e_new = 0.;
        *_d_e_old += _r_d0;
        *_d_delvc += Real_t(0.5) * -_r_d0 * (p_old + q_old);
        *_d_p_old += Real_t(0.5) * delvc * -_r_d0;
        *_d_q_old += Real_t(0.5) * delvc * -_r_d0;
        *_d_work += Real_t(0.5) * _r_d0;
    }
    Real_t _r0 = _d_sixth * -(Real_t(1.) / (_t0 * _t0));
}
__attribute__((device)) __attribute__((always_inline)) static inline void CalcSoundSpeedForElems_device_pullback(Real_t vnewc, Real_t rho0, Real_t enewc, Real_t pnewc, Real_t pbvc, Real_t bvc, Real_t ss4o3, Index_t nz, const Real_t *__restrict ss, Index_t iz, Real_t *_d_vnewc, Real_t *_d_rho0, Real_t *_d_enewc, Real_t *_d_pnewc, Real_t *_d_pbvc, Real_t *_d_bvc, Real_t *_d_ss4o3, Index_t *_d_nz, Real_t *_d_ss, Index_t *_d_iz) {
    bool _cond0;
    Real_t _t0;
    Real_t _d_ssTmp = 0.;
    Real_t ssTmp = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / rho0;
    {
        _cond0 = ssTmp <= Real_t(1.1111110000000001E-37);
        if (_cond0) {
            ssTmp = Real_t(3.333333E-19);
        } else {
            _t0 = ssTmp;
            ssTmp = SQRT(ssTmp);
        }
    }
    const_cast<Real_t &>(ss[iz]) = ssTmp;
    {
        Real_t _r_d2 = _d_ss[iz];
        _d_ss[iz] = 0.;
        _d_ssTmp += _r_d2;
    }
    if (_cond0) {
        {
            Real_t _r_d0 = _d_ssTmp;
            _d_ssTmp = 0.;
        }
    } else {
        {
            ssTmp = _t0;
            Real_t _r_d1 = _d_ssTmp;
            _d_ssTmp = 0.;
            Real_t _r1 = 0.;
            SQRT_pullback(ssTmp, _r_d1, &_r1);
            _d_ssTmp += _r1;
        }
    }
    {
        *_d_pbvc += _d_ssTmp / rho0 * enewc;
        *_d_enewc += pbvc * _d_ssTmp / rho0;
        *_d_vnewc += _d_ssTmp / rho0 * pnewc * bvc * vnewc;
        *_d_vnewc += vnewc * _d_ssTmp / rho0 * pnewc * bvc;
        *_d_bvc += vnewc * vnewc * _d_ssTmp / rho0 * pnewc;
        *_d_pnewc += vnewc * vnewc * bvc * _d_ssTmp / rho0;
        Real_t _r0 = _d_ssTmp * -((pbvc * enewc + vnewc * vnewc * bvc * pnewc) / (rho0 * rho0));
        *_d_rho0 += _r0;
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void UpdateVolumesForElems_device_pullback(Index_t numElem, Real_t v_cut, const Real_t *vnew, const Real_t *v, int i, Index_t *_d_numElem, Real_t *_d_v_cut, Real_t *_d_vnew, Real_t *_d_v, int *_d_i) {
    bool _cond0;
    Real_t _t0;
    Real_t _d_tmpV = 0.;
    Real_t tmpV;
    tmpV = vnew[i];
    {
        _cond0 = FABS(tmpV - Real_t(1.)) < v_cut;
        if (_cond0) {
            _t0 = tmpV;
            tmpV = Real_t(1.);
        }
    }
    const_cast<Real_t &>(v[i]) = tmpV;
    {
        Real_t _r_d2 = _d_v[i];
        _d_v[i] = 0.;
        _d_tmpV += _r_d2;
    }
    if (_cond0) {
        tmpV = _t0;
        Real_t _r_d1 = _d_tmpV;
        _d_tmpV = 0.;
    }
    {
        Real_t _r_d0 = _d_tmpV;
        _d_tmpV = 0.;
        _d_vnew[i] += _r_d0;
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void ApplyMaterialPropertiesForElems_device_pullback(Real_t eosvmin, Real_t eosvmax, const Real_t *__restrict vnew, const Real_t *__restrict v, Real_t &vnewc, const Index_t *__restrict bad_vol, Index_t zn, Real_t *_d_eosvmin, Real_t *_d_eosvmax, Real_t *_d_vnewc, Index_t *_d_zn);
__attribute__((device)) inline void giveMyRegion_pullback(const Index_t *regCSR, const Index_t i, const Index_t numReg, Index_t _d_y, Index_t *_d_i, Index_t *_d_numReg);
__attribute__((device)) __attribute__((always_inline)) static inline void CalcSoundSpeedForElems_device_pullback(Real_t vnewc, Real_t rho0, Real_t enewc, Real_t pnewc, Real_t pbvc, Real_t bvc, Real_t ss4o3, Index_t nz, const Real_t *__restrict ss, Index_t iz, Real_t *_d_vnewc, Real_t *_d_rho0, Real_t *_d_enewc, Real_t *_d_pnewc, Real_t *_d_pbvc, Real_t *_d_bvc, Real_t *_d_ss4o3, Index_t *_d_nz, Index_t *_d_iz);
__attribute__((device)) __attribute__((always_inline)) static inline void UpdateVolumesForElems_device_pullback(Index_t numElem, Real_t v_cut, const Real_t *vnew, const Real_t *v, int i, Index_t *_d_numElem, Real_t *_d_v_cut, int *_d_i);
__attribute__((device)) void Inner_ApplyMaterialPropertiesAndUpdateVolume_kernel_grad_14(Index_t length, Real_t rho0, Real_t e_cut, Real_t emin, const Real_t *__restrict ql, const Real_t *__restrict qq, const Real_t *__restrict vnew, const Real_t *__restrict v, Real_t pmin, Real_t p_cut, Real_t q_cut, Real_t eosvmin, Real_t eosvmax, const Index_t *__restrict regElemlist, Real_t *__restrict e, const Real_t *__restrict delv, const Real_t *__restrict p, const Real_t *__restrict q, Real_t ss4o3, const Real_t *__restrict ss, Real_t v_cut, const Index_t *__restrict bad_vol, const Int_t cost, const Index_t *__restrict regCSR, const Index_t *__restrict regReps, const Index_t numReg, Real_t *_d_e) {
    Index_t _d_length = 0;
    Real_t _d_rho0 = 0.;
    Real_t _d_e_cut = 0.;
    Real_t _d_emin = 0.;
    Real_t _d_pmin = 0.;
    Real_t _d_p_cut = 0.;
    Real_t _d_q_cut = 0.;
    Real_t _d_eosvmin = 0.;
    Real_t _d_eosvmax = 0.;
    Real_t _d_ss4o3 = 0.;
    Real_t _d_v_cut = 0.;
    Int_t _d_cost = 0;
    Index_t _d_numReg = 0;
    bool _cond0;
    Index_t _d_zidx = 0;
    Index_t zidx = 0;
    Real_t _t2;
    Index_t _d_region = 0;
    Index_t region = 0;
    Index_t _d_rep = 0;
    Index_t rep = 0;
    unsigned long _t3;
    int _d_r = 0;
    int r = 0;
    clad::tape<Real_t> _t4 = {};
    clad::tape<Real_t> _t5 = {};
    clad::tape<Real_t> _t6 = {};
    clad::tape<Real_t> _t7 = {};
    clad::tape<Real_t> _t8 = {};
    clad::tape<Real_t> _t9 = {};
    clad::tape<Real_t> _t10 = {};
    Real_t _d_vchalf = 0.;
    Real_t vchalf;
    clad::tape<Real_t> _t11 = {};
    clad::tape<Real_t> _t12 = {};
    clad::tape<bool> _cond1 = {};
    clad::tape<bool> _cond2 = {};
    clad::tape<bool> _cond3 = {};
    clad::tape<bool> _cond4 = {};
    clad::tape<Real_t> _t13 = {};
    clad::tape<Real_t> _t14 = {};
    clad::tape<Real_t> _t15 = {};
    clad::tape<Real_t> _t16 = {};
    clad::tape<Real_t> _t17 = {};
    Real_t _d_e_old = 0., _d_delvc = 0., _d_p_old = 0., _d_q_old = 0., _d_e_temp = 0., _d_delvc_temp = 0., _d_p_temp = 0., _d_q_temp = 0.;
    Real_t e_old, delvc, p_old, q_old, e_temp, delvc_temp, p_temp, q_temp;
    Real_t _d_compression = 0., _d_compHalfStep = 0.;
    Real_t compression, compHalfStep;
    Real_t _d_qq_old = 0., _d_ql_old = 0., _d_qq_temp = 0., _d_ql_temp = 0., _d_work = 0.;
    Real_t qq_old, ql_old, qq_temp, ql_temp, work;
    Real_t _d_p_new = 0., _d_e_new = 0., _d_q_new = 0.;
    Real_t p_new, e_new, q_new;
    Real_t _d_bvc = 0., _d_pbvc = 0., _d_vnewc = 0.;
    Real_t bvc, pbvc, vnewc;
    unsigned int _t1 = blockDim.x;
    unsigned int _t0 = blockIdx.x;
    Index_t _d_i = 0;
    Index_t i = _t1 * _t0 + threadIdx.x;
    {
        _cond0 = i < length;
        if (_cond0) {
            zidx = regElemlist[i];
            _t2 = vnewc;
            ApplyMaterialPropertiesForElems_device(eosvmin, eosvmax, vnew, v, vnewc, bad_vol, zidx);
            region = giveMyRegion(regCSR, i, numReg);
            rep = regReps[region];
            e_temp = e[zidx];
            p_temp = p[zidx];
            q_temp = q[zidx];
            qq_temp = qq[zidx];
            ql_temp = ql[zidx];
            delvc_temp = delv[zidx];
            _t3 = 0UL;
            for (r = 0; ; r++) {
                {
                    if (!(r < rep))
                        break;
                }
                _t3++;
                // clad::push(_t4, e_old);
                e_old = e_temp;
                // clad::push(_t5, p_old);
                p_old = p_temp;
                // clad::push(_t6, q_old);
                q_old = q_temp;
                // clad::push(_t7, qq_old);
                qq_old = qq_temp;
                // clad::push(_t8, ql_old);
                ql_old = ql_temp;
                // clad::push(_t9, delvc);
                delvc = delvc_temp;
                // clad::push(_t10, work);
                work = Real_t(0.);
                // clad::push(_t11, compression);
                compression = Real_t(1.) / vnewc - Real_t(1.);
                vchalf = vnewc - delvc * Real_t(0.5);
                // clad::push(_t12, compHalfStep);
                compHalfStep = Real_t(1.) / vchalf - Real_t(1.);
                {
                    clad::push(_cond1, eosvmin != Real_t(0.));
                    if (clad::back(_cond1)) {
                        {
                            clad::push(_cond2, vnewc <= eosvmin);
                            if (clad::back(_cond2)) {
                                compHalfStep = compression;
                            }
                        }
                    }
                }
                {
                    clad::push(_cond3, eosvmax != Real_t(0.));
                    if (clad::back(_cond3)) {
                        {
                            clad::push(_cond4, vnewc >= eosvmax);
                            if (clad::back(_cond4)) {
                                p_old = Real_t(0.);
                                compression = Real_t(0.);
                                compHalfStep = Real_t(0.);
                            }
                        }
                    }
                }
                // clad::push(_t13, p_new);
                // clad::push(_t14, e_new);
                // clad::push(_t15, q_new);
                // clad::push(_t16, bvc);
                // clad::push(_t17, pbvc);
                CalcEnergyForElems_device(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old, compression, compHalfStep, vnewc, work, delvc, pmin, p_cut, e_cut, q_cut, emin, qq_old, ql_old, rho0, eosvmax, length);
            }
            const_cast<Real_t &>(p[zidx]) = p_new;
            e[zidx] = e_new;
            const_cast<Real_t &>(q[zidx]) = q_new;
            CalcSoundSpeedForElems_device(vnewc, rho0, e_new, p_new, pbvc, bvc, ss4o3, length, ss, zidx);
            UpdateVolumesForElems_device(length, v_cut, vnew, v, zidx);
        }
    }
    if (_cond0) {
        {
            Index_t _r34 = 0;
            Real_t _r35 = 0.;
            Index_t _r36 = 0;
            UpdateVolumesForElems_device_pullback(length, v_cut, vnew, v, zidx, &_r34, &_r35, &_r36);
            _d_length += _r34;
            _d_v_cut += _r35;
            _d_zidx += _r36;
        }
        {
            Real_t _r25 = 0.;
            Real_t _r26 = 0.;
            Real_t _r27 = 0.;
            Real_t _r28 = 0.;
            Real_t _r29 = 0.;
            Real_t _r30 = 0.;
            Real_t _r31 = 0.;
            Index_t _r32 = 0;
            Index_t _r33 = 0;
            CalcSoundSpeedForElems_device_pullback(vnewc, rho0, e_new, p_new, pbvc, bvc, ss4o3, length, ss, zidx, &_r25, &_r26, &_r27, &_r28, &_r29, &_r30, &_r31, &_r32, &_r33);
            _d_vnewc += _r25;
            _d_rho0 += _r26;
            _d_e_new += _r27;
            _d_p_new += _r28;
            _d_pbvc += _r29;
            _d_bvc += _r30;
            _d_ss4o3 += _r31;
            _d_length += _r32;
            _d_zidx += _r33;
        }
        {
            Real_t _r_d20 = _d_e[zidx];
            _d_e[zidx] = 0.;
            _d_e_new += _r_d20;
        }
        for (;; _t3--) {
            {
                if (!_t3)
                    break;
            }
            {
                // p_new = clad::back(_t13);
                // e_new = clad::back(_t14);
                // q_new = clad::back(_t15);
                // bvc = clad::back(_t16);
                // pbvc = clad::back(_t17);
                Real_t _r7 = 0.;
                Real_t _r8 = 0.;
                Real_t _r9 = 0.;
                Real_t _r10 = 0.;
                Real_t _r11 = 0.;
                Real_t _r12 = 0.;
                Real_t _r13 = 0.;
                Real_t _r14 = 0.;
                Real_t _r15 = 0.;
                Real_t _r16 = 0.;
                Real_t _r17 = 0.;
                Real_t _r18 = 0.;
                Real_t _r19 = 0.;
                Real_t _r20 = 0.;
                Real_t _r21 = 0.;
                Real_t _r22 = 0.;
                Real_t _r23 = 0.;
                Index_t _r24 = 0;
                CalcEnergyForElems_device_pullback(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old, compression, compHalfStep, vnewc, work, delvc, pmin, p_cut, e_cut, q_cut, emin, qq_old, ql_old, rho0, eosvmax, length, &_d_p_new, &_d_e_new, &_d_q_new, &_d_bvc, &_d_pbvc, &_r7, &_r8, &_r9, &_r10, &_r11, &_r12, &_r13, &_r14, &_r15, &_r16, &_r17, &_r18, &_r19, &_r20, &_r21, &_r22, &_r23, &_r24);
                // clad::pop(_t13);
                // clad::pop(_t14);
                // clad::pop(_t15);
                // clad::pop(_t16);
                // clad::pop(_t17);
                _d_p_old += _r7;
                _d_e_old += _r8;
                _d_q_old += _r9;
                _d_compression += _r10;
                _d_compHalfStep += _r11;
                _d_vnewc += _r12;
                _d_work += _r13;
                _d_delvc += _r14;
                _d_pmin += _r15;
                _d_p_cut += _r16;
                _d_e_cut += _r17;
                _d_q_cut += _r18;
                _d_emin += _r19;
                _d_qq_old += _r20;
                _d_ql_old += _r21;
                _d_rho0 += _r22;
                _d_eosvmax += _r23;
                _d_length += _r24;
            }
            {
                if (clad::back(_cond3)) {
                    {
                        if (clad::back(_cond4)) {
                            {
                                Real_t _r_d19 = _d_compHalfStep;
                                _d_compHalfStep = 0.;
                            }
                            {
                                Real_t _r_d18 = _d_compression;
                                _d_compression = 0.;
                            }
                            {
                                Real_t _r_d17 = _d_p_old;
                                _d_p_old = 0.;
                            }
                        }
                        clad::pop(_cond4);
                    }
                }
                clad::pop(_cond3);
            }
            {
                if (clad::back(_cond1)) {
                    {
                        if (clad::back(_cond2)) {
                            {
                                Real_t _r_d16 = _d_compHalfStep;
                                _d_compHalfStep = 0.;
                                _d_compression += _r_d16;
                            }
                        }
                        clad::pop(_cond2);
                    }
                }
                clad::pop(_cond1);
            }
            {
                // compHalfStep = clad::pop(_t12);
                Real_t _r_d15 = _d_compHalfStep;
                _d_compHalfStep = 0.;
                Real_t _r6 = _r_d15 * -(Real_t(1.) / (vchalf * vchalf));
                _d_vchalf += _r6;
            }
            {
                Real_t _r_d14 = _d_vchalf;
                _d_vchalf = 0.;
                _d_vnewc += _r_d14;
                _d_delvc += -_r_d14 * Real_t(0.5);
            }
            {
                // compression = clad::pop(_t11);
                Real_t _r_d13 = _d_compression;
                _d_compression = 0.;
                Real_t _r5 = _r_d13 * -(Real_t(1.) / (vnewc * vnewc));
                _d_vnewc += _r5;
            }
            _d_vchalf = 0.;
            {
                // work = clad::pop(_t10);
                Real_t _r_d12 = _d_work;
                _d_work = 0.;
            }
            {
                // delvc = clad::pop(_t9);
                Real_t _r_d11 = _d_delvc;
                _d_delvc = 0.;
                _d_delvc_temp += _r_d11;
            }
            {
                // ql_old = clad::pop(_t8);
                Real_t _r_d10 = _d_ql_old;
                _d_ql_old = 0.;
                _d_ql_temp += _r_d10;
            }
            {
                // qq_old = clad::pop(_t7);
                Real_t _r_d9 = _d_qq_old;
                _d_qq_old = 0.;
                _d_qq_temp += _r_d9;
            }
            {
                // q_old = clad::pop(_t6);
                Real_t _r_d8 = _d_q_old;
                _d_q_old = 0.;
                _d_q_temp += _r_d8;
            }
            {
                // p_old = clad::pop(_t5);
                Real_t _r_d7 = _d_p_old;
                _d_p_old = 0.;
                _d_p_temp += _r_d7;
            }
            {
                // e_old = clad::pop(_t4);
                Real_t _r_d6 = _d_e_old;
                _d_e_old = 0.;
                _d_e_temp += _r_d6;
            }
        }
        {
            Real_t _r_d5 = _d_delvc_temp;
            _d_delvc_temp = 0.;
        }
        {
            Real_t _r_d4 = _d_ql_temp;
            _d_ql_temp = 0.;
        }
        {
            Real_t _r_d3 = _d_qq_temp;
            _d_qq_temp = 0.;
        }
        {
            Real_t _r_d2 = _d_q_temp;
            _d_q_temp = 0.;
        }
        {
            Real_t _r_d1 = _d_p_temp;
            _d_p_temp = 0.;
        }
        {
            Real_t _r_d0 = _d_e_temp;
            _d_e_temp = 0.;
            _d_e[zidx] += _r_d0;
        }
        {
            Index_t _r3 = 0;
            Index_t _r4 = 0;
            giveMyRegion_pullback(regCSR, i, numReg, _d_region, &_r3, &_r4);
            _d_i += _r3;
            _d_numReg += _r4;
        }
        {
            vnewc = _t2;
            Real_t _r0 = 0.;
            Real_t _r1 = 0.;
            Index_t _r2 = 0;
            ApplyMaterialPropertiesForElems_device_pullback(eosvmin, eosvmax, vnew, v, _t2, bad_vol, zidx, &_r0, &_r1, &_d_vnewc, &_r2);
            _d_eosvmin += _r0;
            _d_eosvmax += _r1;
            _d_zidx += _r2;
        }
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void ApplyMaterialPropertiesForElems_device_pullback(Real_t eosvmin, Real_t eosvmax, const Real_t *__restrict vnew, const Real_t *__restrict v, Real_t &vnewc, const Index_t *__restrict bad_vol, Index_t zn, Real_t *_d_eosvmin, Real_t *_d_eosvmax, Real_t *_d_vnewc, Index_t *_d_zn) {
    bool _cond0;
    bool _cond1;
    bool _cond2;
    bool _cond3;
    bool _cond4;
    bool _cond5;
    bool _cond6;
    bool _cond7;
    bool _cond8;
    vnewc = vnew[zn];
    {
        _cond0 = eosvmin != Real_t(0.);
        if (_cond0) {
            {
                _cond1 = vnewc < eosvmin;
                if (_cond1)
                    vnewc = eosvmin;
            }
        }
    }
    {
        _cond2 = eosvmax != Real_t(0.);
        if (_cond2) {
            {
                _cond3 = vnewc > eosvmax;
                if (_cond3)
                    vnewc = eosvmax;
            }
        }
    }
    Real_t _d_vc = 0.;
    Real_t vc = v[zn];
    {
        _cond4 = eosvmin != Real_t(0.);
        if (_cond4) {
            {
                _cond5 = vc < eosvmin;
                if (_cond5)
                    vc = eosvmin;
            }
        }
    }
    {
        _cond6 = eosvmax != Real_t(0.);
        if (_cond6) {
            {
                _cond7 = vc > eosvmax;
                if (_cond7)
                    vc = eosvmax;
            }
        }
    }
    {
        _cond8 = vc <= 0.;
        if (_cond8) {
            const_cast<Index_t &>(*bad_vol) = zn;
        }
    }
    if (_cond8) {
    }
    if (_cond6) {
        if (_cond7) {
            Real_t _r_d4 = _d_vc;
            _d_vc = 0.;
            *_d_eosvmax += _r_d4;
        }
    }
    if (_cond4) {
        if (_cond5) {
            Real_t _r_d3 = _d_vc;
            _d_vc = 0.;
            *_d_eosvmin += _r_d3;
        }
    }
    if (_cond2) {
        if (_cond3) {
            Real_t _r_d2 = *_d_vnewc;
            *_d_vnewc = 0.;
            *_d_eosvmax += _r_d2;
        }
    }
    if (_cond0) {
        if (_cond1) {
            Real_t _r_d1 = *_d_vnewc;
            *_d_vnewc = 0.;
            *_d_eosvmin += _r_d1;
        }
    }
    {
        Real_t _r_d0 = *_d_vnewc;
        *_d_vnewc = 0.;
    }
}
__attribute__((device)) inline void giveMyRegion_pullback(const Index_t *regCSR, const Index_t i, const Index_t numReg, Index_t _d_y, Index_t *_d_i, Index_t *_d_numReg) {
    Index_t _d_reg = 0;
    Index_t reg = 0;
    clad::tape<bool> _cond0 = {};
    unsigned long _t0 = 0UL;
    for (reg = 0; ; reg++) {
        {
            if (!(reg < numReg - 1))
                break;
        }
        _t0++;
        {
            clad::push(_cond0, i < regCSR[reg]);
            if (clad::back(_cond0))
                goto _label0;
        }
    }
    *_d_numReg += _d_y;
    for (;; _t0--) {
        {
            if (!_t0)
                break;
        }
        reg--;
        if (clad::back(_cond0))
          _label0:
            _d_reg += _d_y;
        clad::pop(_cond0);
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void CalcSoundSpeedForElems_device_pullback(Real_t vnewc, Real_t rho0, Real_t enewc, Real_t pnewc, Real_t pbvc, Real_t bvc, Real_t ss4o3, Index_t nz, const Real_t *__restrict ss, Index_t iz, Real_t *_d_vnewc, Real_t *_d_rho0, Real_t *_d_enewc, Real_t *_d_pnewc, Real_t *_d_pbvc, Real_t *_d_bvc, Real_t *_d_ss4o3, Index_t *_d_nz, Index_t *_d_iz) {
    bool _cond0;
    Real_t _t0;
    Real_t _d_ssTmp = 0.;
    Real_t ssTmp = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / rho0;
    {
        _cond0 = ssTmp <= Real_t(1.1111110000000001E-37);
        if (_cond0) {
            ssTmp = Real_t(3.333333E-19);
        } else {
            _t0 = ssTmp;
            ssTmp = SQRT(ssTmp);
        }
    }
    const_cast<Real_t &>(ss[iz]) = ssTmp;
    if (_cond0) {
        {
            Real_t _r_d0 = _d_ssTmp;
            _d_ssTmp = 0.;
        }
    } else {
        {
            ssTmp = _t0;
            Real_t _r_d1 = _d_ssTmp;
            _d_ssTmp = 0.;
            Real_t _r1 = 0.;
            SQRT_pullback(ssTmp, _r_d1, &_r1);
            _d_ssTmp += _r1;
        }
    }
    {
        *_d_pbvc += _d_ssTmp / rho0 * enewc;
        *_d_enewc += pbvc * _d_ssTmp / rho0;
        *_d_vnewc += _d_ssTmp / rho0 * pnewc * bvc * vnewc;
        *_d_vnewc += vnewc * _d_ssTmp / rho0 * pnewc * bvc;
        *_d_bvc += vnewc * vnewc * _d_ssTmp / rho0 * pnewc;
        *_d_pnewc += vnewc * vnewc * bvc * _d_ssTmp / rho0;
        Real_t _r0 = _d_ssTmp * -((pbvc * enewc + vnewc * vnewc * bvc * pnewc) / (rho0 * rho0));
        *_d_rho0 += _r0;
    }
}
__attribute__((device)) __attribute__((always_inline)) static inline void UpdateVolumesForElems_device_pullback(Index_t numElem, Real_t v_cut, const Real_t *vnew, const Real_t *v, int i, Index_t *_d_numElem, Real_t *_d_v_cut, int *_d_i) {
    bool _cond0;
    Real_t _t0;
    Real_t _d_tmpV = 0.;
    Real_t tmpV;
    tmpV = vnew[i];
    {
        _cond0 = FABS(tmpV - Real_t(1.)) < v_cut;
        if (_cond0) {
            _t0 = tmpV;
            tmpV = Real_t(1.);
        }
    }
    const_cast<Real_t &>(v[i]) = tmpV;
    if (_cond0) {
        tmpV = _t0;
        Real_t _r_d1 = _d_tmpV;
        _d_tmpV = 0.;
    }
    {
        Real_t _r_d0 = _d_tmpV;
        _d_tmpV = 0.;
    }
}