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

__all__ = ['VLE']
from copy import deepcopy
import numpy as np
from scipy import optimize
from thermo.acentric import omega
from thermo.critical import Tc, Pc
from thermo.eos_mix import PRMIX
from thermo.identifiers import CAS_from_any


class VLE(object):
    r'''Class for solving generic vapor-liquid equilibrium problem.

    .. math::
        \phi_L x = \phi_V y

    Methods
    -------
    solve(**kwargs)
        Solve the specified VLE propblem, with kwargs being parameters
        for the ``scipy.optimize.brentq`` algorithm.
    from_IDs(IDs, zs, eos=PRMIX, T=298.15, P=101325, kijs=None,
             phase='l', prop='T')
        Creates VLE from IDs of inidividual components.
    '''

    def __init__(self, eos, phase='l', prop='T'):
        r'''Class for solving generic vapor-liquid equilibrium problem.

        Parameters
        ----------
        eos : GCEOSMIX
            An instance of the class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX``.
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.

        Examples
        --------
        >>> Tcs = [126.2, 190.564]
        >>> Pcs = [3394387.5, 4599000.0]
        >>> os = [0.04, 0.008]
        >>> zs = [0.5, 0.5]
        >>> eos = PRMIX(T=115, P=1e6, Tcs=Tcs, Pcs=Pcs, omegas=os, zs=zs)
        >>> vle = VLE(eos, phase='l', prop='T')
        >>> T = vle.solve()
        >>> print(f'T = {T}')
        T = 114.79355378739642
        '''
        assert phase in ['l', 'g']
        assert prop in ['T', 'P']
        self.eos = deepcopy(eos)
        self.phase = phase
        self.prop = prop

    def solve(self, full_output=False, **kwargs):
        r'''Solves specified VLE problem and returns an optimized parameter.

        Parameters
        ----------
        full_output : bool, optional
            If True, return optional outputs.
        kwargs : dict
            Keyword arguments for the ``scipy.optimize.brentq`` algorithm.

        Returns
        -------
        prop : float
            Equilibrium property (pressure of temperature).
        zs : array of float
            Equilibrium molar fractions in liquid or vapor state.
        s : float
            Deviation coefficient. The closer to 1 the better.
        '''
        def opt_func(value):
            setattr(self.eos, self.prop, value)
            s, _ = self._solve_eos()
            return s - 1

        # TODO: Try to improve the algorithm such that it
        # does not rely on the `step_size`
        if self.prop == 'T':
            step_size = 1
        else:
            step_size = 5e3

        a, b = self._initialize(
            opt_func,
            getattr(self.eos, self.prop),
            step_size
        )

        prop = optimize.brentq(opt_func, a, b, **kwargs)
        s, zs = self._solve_eos()
        if full_output:
            return prop, zs, s
        return prop, zs

    def _solve_eos(self):
        r'''Solve EoS and calculates new molar fractions.

        Returns
        -------
        s : float
            Sum of the unnormalized recalculated molar fractions.
        zs : list of float
            Normalized molar fractions.
        '''
        self.eos.solve()
        self.eos.fugacities()
        if self.phase == 'l':
            rzs = np.divide(self.eos.phis_l, self.eos.phis_g) * self.eos.zs
        else:
            rzs = np.divide(self.eos.phis_g, self.eos.phis_l) * self.eos.zs
        s = np.sum(rzs)
        return s, np.divide(rzs, s)

    def _initialize(self, f, x0, step_size, maxfun=500):
        r'''Find initial bracketing interval for VLE problem-solving.
        Returns previous and current value if their signs differ.

        Parameters
        ----------
        f : callable ``f(x)``
            Function for which the interval needs to be found.
        x0 : float
            Inital guess.
        step_size : float
            Increment or decrement which will be applied on each step
            until the interval was found.
        maxfun : int, optional
            Maximum number of function calling, defaults to 500.

        Returns
        -------
        a : float
            Left boundary of the interval.
        b : float
            Right boundary of the interval.
        '''
        s0 = f(x0)
        num = 1
        while True:
            x0 += step_size
            s = f(x0)
            num += 1
            if np.sign(s0) != np.sign(s):
                return sorted([x0 - step_size, x0])
            else:
                if np.abs(s) > np.abs(s0):
                    step_size *= -1
                else:
                    s0 = s
            if num >= maxfun:
                raise ValueError(["A maximum number of function "
                                  "calls has reached."][0])
        return []

    @classmethod
    def from_IDs(cls, IDs, zs, eos=PRMIX, T=298.15, P=101325,
                 kijs=None, lijs=None, mijs=None, phase='l', prop='T',
                 is_as=False):
        r'''Creates VLE from IDs of inidividual components.

        Parameters
        ----------
        IDs : list, optional
            List of chemical identifiers - names, CAS numbers, SMILES or
            InChi strings can all be recognized and may be mixed [-]
        zs : list of float
            Molar fractions of each individual component in the mixture.
        eos : type, optional
            Type of the EoS, class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX`. Defaults to
            ``thermo.eos_mix.PRMIX``.
        T : float, optional
            Temperature, [K]. Defaults to 298.15 K or 25 °C.
        P : float, optional
            Pressure, [Pa]. Defaults to 101325 Pa or 1 atm.
        kijs : list of list of float
            n*n size list of lists with binary interaction parameters for
            the Van der Waals mixing rules, default all 0.
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.

        Returns
        -------
        thermo.VLE

        Examples
        --------
        >>> IDs = ["nitrogen", "methane"]
        >>> vle = VLE.from_IDs(IDs, zs=[0.5, 0.5], T=115, P=1e6)
        '''
        cass = [CAS_from_any(x) for x in IDs]
        if not is_as:
            aeos = eos(
                Tcs=[Tc(x) for x in cass],
                Pcs=[Pc(x) for x in cass],
                omegas=[omega(x) for x in cass],
                T=T,
                P=P,
                zs=zs,
                kijs=kijs,
                is_as=is_as
            )
        else:
            aeos = eos(
                Tcs=[Tc(x) for x in cass],
                Pcs=[Pc(x) for x in cass],
                omegas=[omega(x) for x in cass],
                T=T,
                P=P,
                zs=zs,
                lijs=lijs,
                mijs=mijs,
                is_as=is_as
            )
        return cls(aeos, phase=phase, prop=prop)

    @classmethod
    def fit_kijs(cls, eoss, a=-0.2, b=0.2, exp_zs=[], exp_props=[], phase='l',
                 prop='T', **kwargs):
        r'''Fit binary interaction parameter of the specified EoS using
        expermental data.

        Parameters
        ----------
        eoss : list of GCEOSMIX
            An instances of the class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX``.
        a : float, optional
            Left boundary for the ``scipy.optimize.fminbound`` optimization
            algorithm. Defaults to -0.2.
        b : float, optional
            Right boundary for the ``scipy.optimize.fminbound`` optimization
            algorithm. Defaults to 0.2.
        exp_zs : list of float, optional
            Molar fractions of the low boiling component obtained by
            experiment.
        exp_props : list of float, optional
            Temperatures or pressures obtained by experiment.
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.
        kwargs : dict
            Keyword arguments for the ``scipy.optimize.fminbound`` algorithm.

        Returns
        -------
        float
            Fitted binary interaction parameter.

        Notes
        -----
        Either `exp_zs`, `exp_props` or both needs to be specified.
        '''
        assert exp_zs or exp_props
        if exp_zs:
            assert len(exp_zs) == len(eoss)
        else:
            assert len(exp_props) == len(eoss)

        def opt_func(k):
            kijs = [[0, k], [k, 0]]
            vals = []
            props = []
            for vle in [cls(eos, phase=phase, prop=prop) for eos in eoss]:
                vle.eos.kijs = kijs
                pr, zs = vle.solve()
                props.append(pr)
                vals.append(zs[0])
            err = 0
            if exp_zs:
                n0p = np.greater(exp_zs, 0)
                ezs = np.compress(n0p, exp_zs)
                vls = np.compress(n0p, vals)
                err += np.sum((2*np.subtract(ezs, vls) / ezs)**2)
            if exp_props:
                err += np.sum((2*np.subtract(exp_props, props) / exp_props)**2)
            return err

        res = optimize.fminbound(opt_func, a, b, **kwargs)
        return res

    @classmethod
    def fit_lmijs(cls, eoss, a=-0.2, b=0.2, exp_zs=[], exp_props=[], phase='l',
                  prop='T', **kwargs):
        r'''Fit binary interaction parameter of the specified EoS using
        expermental data.

        Parameters
        ----------
        eoss : list of GCEOSMIX
            An instances of the class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX``.
        a : float, optional
            Left boundary for the ``scipy.optimize.fminbound`` optimization
            algorithm. Defaults to -0.2.
        b : float, optional
            Right boundary for the ``scipy.optimize.fminbound`` optimization
            algorithm. Defaults to 0.2.
        exp_zs : list of float, optional
            Molar fractions of the low boiling component obtained by
            experiment.
        exp_props : list of float, optional
            Temperatures or pressures obtained by experiment.
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.
        kwargs : dict
            Keyword arguments for the ``scipy.optimize.fminbound`` algorithm.

        Returns
        -------
        float
            Fitted binary interaction parameter.

        Notes
        -----
        Either `exp_zs`, `exp_props` or both needs to be specified.
        '''
        assert exp_zs or exp_props
        if exp_zs:
            assert len(exp_zs) == len(eoss)
        else:
            assert len(exp_props) == len(eoss)

        def opt_func(lm):
            l, m = lm
            lijs = [[0, l], [l, 0]]
            mijs = [[0, m], [m, 0]]
            vals = []
            props = []
            for vle in [cls(eos, phase=phase, prop=prop) for eos in eoss]:
                vle.eos.lijs = lijs
                vle.eos.mijs = mijs
                pr, zs = vle.solve()
                props.append(pr)
                vals.append(zs[0])
            err = 0
            if exp_zs:
                n0p = np.greater(exp_zs, 0)
                ezs = np.compress(n0p, exp_zs)
                vls = np.compress(n0p, vals)
                err += np.sum((2*np.subtract(ezs, vls) / ezs)**2)
            if exp_props:
                err += np.sum((2*np.subtract(exp_props, props) / exp_props)**2)
            return err

        res = optimize.fmin(opt_func, [0.0, 0.0], **kwargs)
        # res = optimize.fminbound(opt_func, a, b, **kwargs)
        return res

    @classmethod
    def fit_lmijs_from_IDs(cls, IDs, zs, a=-0.2, b=0.2, exp_zs=[],
                           exp_props=[],
                           eos=PRMIX, Ts=[], Ps=[], phase='l', prop='T',
                           **kwargs):
        r'''Fit binary interaction parameter of the specified EoS using
        expermental data.

        Parameters
        ----------
        IDs : list, optional
            List of chemical identifiers - names, CAS numbers, SMILES or
            InChi strings can all be recognized and may be mixed [-]
        zs : list of float
            Molar fractions of each individual component in the mixture.
        a : float, optional
            Left boundary for the ``scipy.optimize.brentq`` optimization
            algorithm. Defaults to -0.2.
        b : float, optional
            Right boundary for the ``scipy.optimize.brentq`` optimization
            algorithm. Defaults to 0.2.
        exp_zs : list of float, optional
            Molar fractions of the low boiling component obtained by
            experiment.
        exp_props : list of float, optional
            Temperatures or pressures obtained by experiment.
        eos : type, optional
            Type of the EoS, class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX`. Defaults to
            ``thermo.eos_mix.PRMIX``.
        Ts : list of float
            Inital temperatures, [K].
        Ps : list of float
            Initial pressures, [Pa].
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.
        kwargs : dict
            Keyword arguments for the ``scipy.optimize.fminbound`` algorithm.

        Returns
        -------
        float
            Fitted binary interaction parameter.
        '''
        assert len(Ts) == len(Ps) == len(zs)
        if len(exp_zs) != 0:
            assert len(exp_zs) == len(Ts)
        if len(exp_props) != 0:
            assert len(exp_props) == len(Ts)
        eoss = []
        for t, p, z in zip(Ts, Ps, zs):
            eoss.append(
                VLE.from_IDs(
                    IDs,
                    [z, 1-z],
                    eos=eos,
                    T=t,
                    P=p,
                    phase=phase,
                    prop=prop,
                    is_as=True
                ).eos
            )
        return cls.fit_lmijs(eoss, a=a, b=b, exp_zs=exp_zs,
                             exp_props=exp_props,
                             phase=phase, prop=prop, **kwargs)

    @classmethod
    def fit_kijs_from_IDs(cls, IDs, zs, a=-0.2, b=0.2, exp_zs=[], exp_props=[],
                          eos=PRMIX, Ts=[], Ps=[], phase='l', prop='T',
                          **kwargs):
        r'''Fit binary interaction parameter of the specified EoS using
        expermental data.

        Parameters
        ----------
        IDs : list, optional
            List of chemical identifiers - names, CAS numbers, SMILES or
            InChi strings can all be recognized and may be mixed [-]
        zs : list of float
            Molar fractions of each individual component in the mixture.
        a : float, optional
            Left boundary for the ``scipy.optimize.brentq`` optimization
            algorithm. Defaults to -0.2.
        b : float, optional
            Right boundary for the ``scipy.optimize.brentq`` optimization
            algorithm. Defaults to 0.2.
        exp_zs : list of float, optional
            Molar fractions of the low boiling component obtained by
            experiment.
        exp_props : list of float, optional
            Temperatures or pressures obtained by experiment.
        eos : type, optional
            Type of the EoS, class that is directly inherited from
            or behave like ``thermo.eos_mix.GCEOSMIX`. Defaults to
            ``thermo.eos_mix.PRMIX``.
        Ts : list of float
            Inital temperatures, [K].
        Ps : list of float
            Initial pressures, [Pa].
        phase : {'l', 'g'}, optional
            The phase of the mixture which will be considered as constant.
        prop : {'T', 'P'}, optional
            Which of the two parameters (pressure or temperature) will be
            changing during the optimization process.
        kwargs : dict
            Keyword arguments for the ``scipy.optimize.fminbound`` algorithm.

        Returns
        -------
        float
            Fitted binary interaction parameter.
        '''
        assert len(Ts) == len(Ps) == len(zs)
        if len(exp_zs) != 0:
            assert len(exp_zs) == len(Ts)
        if len(exp_props) != 0:
            assert len(exp_props) == len(Ts)
        eoss = []
        for t, p, z in zip(Ts, Ps, zs):
            eoss.append(
                VLE.from_IDs(
                    IDs,
                    [z, 1-z],
                    eos=eos,
                    T=t,
                    P=p,
                    phase=phase,
                    prop=prop
                ).eos
            )
        return cls.fit_kijs(eoss, a=a, b=b, exp_zs=exp_zs, exp_props=exp_props,
                            phase=phase, prop=prop, **kwargs)
