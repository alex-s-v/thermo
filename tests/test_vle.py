# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Alexandr Vasilyev <alexandr.s.vasilyev@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from numpy.testing import assert_allclose
from thermo.vle import VLE
from thermo.eos_mix import PRMIX


def test_VLE_solve():
    # Two-phase nitrogen-methane
    eos = PRMIX(T=115, P=1E6, Tcs=[126.2, 190.564], Pcs=[3394387.5, 4599000.0],
                omegas=[0.04, 0.008], zs=[0.5, 0.5], kijs=[[0, 0], [0, 0]])
    vle = VLE(eos, phase='l', prop='T')
    # Test for constant liquid phase and optimized temperature
    T = vle.solve()[0]
    T_expected = 114.79355378739642
    assert_allclose(T, T_expected)
    # Test for constant liquid phase and optimized pressure
    vle = VLE(eos, phase='l', prop='P')
    P = vle.solve()[0]
    P_expected = 1010396.4497357836
    assert_allclose(P, P_expected)
    # Test for constant vapor phase and optimized pressure
    vle = VLE(eos, phase='g', prop='P')
    vle.eos.zs = [0.9, 0.1]  # for zs=[0.5, 0.5] currently not solvable
    P = vle.solve()[0]
    P_expected = 1232752.904810388
    assert_allclose(P, P_expected)
    # Test generating vle from IDs
    vle = VLE.from_IDs(["nitrogen", "methane"], zs=[0.5, 0.5], T=115, P=1E6)
    # Test for constant liquid phase and optimized temperature
    T = vle.solve()[0]
    assert_allclose(T, T_expected)


def test_VLE_fit():
    # Aceton - Carbon tetrachloride for isotermal data at 45 °C
    data = {
        "IDs": ["aceton", "carbon tetrachloride"],
        # liquid mole fractions
        "zs": [0.0556, 0.0903, 0.2152, 0.2929, 0.3970, 0.4769,
               0.5300, 0.6047, 0.7128, 0.8088, 0.9090, 0.9636],
        # vapor mole fractions
        "exp_zs": [0.2165, 0.2910, 0.4495, 0.5137, 0.5832, 0.6309,
                   0.6621, 0.7081, 0.7718, 0.8360, 0.9141, 0.9636],
        # pressures in Pa
        "exp_props": [42039.2092, 45289.6086, 53031.6385, 56323.3678,
                      59845.7447, 61850.9132, 63040.1487, 64682.6803,
                      66403.8720, 67579.7753, 68303.7158, 68421.0395],
        # 12 - number of datapoints
        "Ts": [318.15 for _ in range(12)],
        "Ps": [101325 for _ in range(12)],
        "prop": "P"
    }

    bip = VLE.fit_kijs_from_IDs(**data)
    assert_allclose(bip, 0.0641753392758043)
