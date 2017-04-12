# -*- coding: utf-8 -*-
"""
Python module to solve for and analyze full commitment and renegotiation-proof
contracts.  See https://github.com/jhconning/renegotiation
Note: code originally had delta and r... but some formulas assume delta=1/(1+r)

For a paper by Karna Basu and Jonathan Conning

@author: Jonathan Conning
"""
import numpy as np
from scipy.optimize import minimize, brentq, fsolve


class Contract(object):
    """ Base Class for defining contracts  """
    def __init__(self,beta,y=None):         # constructor method to set default params.
        self.beta  = beta                   # present bias in β-δ framework
        self.rho   = 0.95                   # 1/rho = elasticity of substitution
        if y is None:
            self.y = np.array([100,100,100])
        else:
            self.y = y

    def __repr__(self):
        return 'Contract(beta=%s, y=%s)' % (self.beta, self.y)

    def print_params(self):
        """ print out parameters """
        params = vars(self)
        for p in sorted(params):    # print attributes alphabetically
            print("{0:<7} : {1}".format(p,params[p]))

    def u(self,ct):
        """ utility function """
        if self.rho==1:
            return np.log(ct)
        else:
            return  (ct**(1-self.rho))/(1-self.rho)

    def PV(self,c):
        """discounted present value of any stream c"""
        return c[0] + np.sum(c[1:])

    def PVU(self,c, beta):
        """discounted present utility value of any stream c"""
        return  self.u(c[0]) + beta * sum([self.u(ct) for ct in c[1:]] )

    def profit(self,c,y):
        """ present value of lender profits when exchanges c for y"""
        return  self.PV(y)-self.PV(c)

    def indif(self, ubar, beta):
        """ returns u(c1, c2) for graphing indifference curves in c1-c2 space.
        if beta = 1, will describe self 0's preferences """
        if self.rho==1:
            def idc(c1):
                lnc2 = (ubar-np.log(c1))/beta
                return np.exp(lnc2)
        else:
            def idc(c1):
                return (((1-self.rho)/beta)*(ubar-self.u(c1)))**(1/(1-self.rho))

        return idc

    def isoprofit(self, prfbar, y):
        """ returns profit(c1, c2) for graphing isoprofit lines in c1-c2 space.
        """
        def isoprf(c1):
            """isoprofit function isoprf(c1) """
            return np.array(y[1] + y[2] - prfbar) - c1
        return isoprf

    def reneg(self,c):
        """ Renegotiated monopoly contract offered to period-1-self if
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        beta, rho = self.beta, self.rho
        btr = beta**(1/rho)
        if rho==1:
            lncr1 = (np.log(c[1])+beta*np.log(c[2])-beta*np.log(beta))/(1+beta)
            cr1 = np.exp(lncr1)
        else:
            pu =  self.u(c[1]) + beta*self.u(c[2])
            a  =  (pu *(1-rho) )**(1/(1-rho))
            b = (1 + btr)**(1/(rho-1))
            cr1 = a*b
        cr2 = cr1*btr
        return np.array([c[0],cr1,cr2])


class Monopoly(Contract):                    # build on contract class
    """ Class for solving Monopoly equilibrium contracts  """
    def __init__(self,beta):
        super(Monopoly,self).__init__(beta)    # inherit parent class properties
        self.kappa  = 0                        # cost of renegotiation
        self.guess  = self.y                   # initial guess for solver

    def __repr__(self):
        return 'Monopoly(beta=%s, y=%s)' % (self.beta, self.y)

    def fcommit(self):
        """monopolist optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        bt, rho = self.beta, self.rho
        btr = bt**(1/rho)
        pvu = self.PVU(self.y,bt)
        if rho==1:
            lnc0 = (pvu-2*bt*np.log(bt))/(1+2*bt)
            c0 = np.exp(lnc0)
        else:
            a = ((pvu*(1-rho))**(1/(1-rho)) )
            b = (1 + 2*btr)**(1/(rho-1))
            c0 = a*b
        return np.array([1,btr,btr])*c0

    def negprofit(self,c):
        """ Negative profits (for minimization)"""
        return  -(self.PV(self.y) - self.PV(c))

    def bestreneg(self, c0):
        '''return period 1 consumption of reneg-proof contract.
        Assumes '''
        beta, rho = self.beta, self.rho
        Y = np.sum(self.y)
        c1F = (Y - c0) / 2     # for lower bound
        btr = beta ** (1 / rho)
        c1P = (Y - c0 - self.kappa) / (1 + btr)   #for upper bound
        ub = (1 + btr) * self.u(c1P)

        def U1(c1):
            return self.u(c1) + beta * self.u(Y - c0 - c1)

        def f(c1):
            return U1(c1) - ub

        return brentq(f, c1F, c1P)

    def reneg_proof(self):
        """Find period 0 monopoly best renegotiation-proof contract by searching over
        subgame perfect responses """

        guess = self.fcommit()[0]
        Y = np.sum(self.y)

        def f(c0):
            c1 = self.bestreneg(c0)
            return -(self.u(c0) + self.beta * (self.u(c1) + self.u(Y - c0 - c1)))

        c0rp = minimize(f, guess, method='Nelder-Mead').x[0]
        c1rp = self.bestreneg(c0rp)
        c2rp = Y - c0rp - c1rp

        return np.array([c0rp, c1rp, c2rp])

    def reneg_proof_cons(self,c):
        """ the renegotiation-proof constraint gain from renegotiation
        cannot exceed its cost kappa"""
        return  -(self.profit(self.reneg(c),self.y)
                  -  self.profit(c,self.y) - self.kappa)

    def participation_cons(self,c):
        return (self.PVU(c,self.beta)  - self.PVU(self.y,self.beta))

    def reneg_proof2(self):
        """calculate renegotiation-proof contract
        supplies constraints to solver that bank can't profit too much
        and period 0 borrower participation"""

        cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_cons },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        res = minimize(self.negprofit, self.guess, method='COBYLA',
                     constraints = cons)
        return res


class Competitive(Contract):                    # build on contract class
    """ Class for solving competitive equilibrium contracts  """
    def __init__(self, beta):
        super(Competitive,self).__init__(beta)  # inherits parent class properties
        self.kappa  = 0                         # cost of renegotiation
        self.guess  = self.y                    # initial guess for solver

    def __repr__(self):
        return 'Competitive(beta=%s, y=%s)' % (self.beta, self.y)

    def fcommit(self):
        """competitive optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        btr = self.beta**(1/self.rho)
        return np.array([1,btr,btr])*self.PV(self.y)/(1 + 2*btr)

    def ownsmooth(self):
        """contract if have to smooth by oneself without contract
        from closed form solution for CRRA with kappa=0"""
        beta, rho = self.beta, self.rho
        btr = beta**(1/rho)
        pvy = self.PV(self.y)
        lam = (beta+btr)/(1+btr)
        ltr = lam**(1/rho)
        return np.array([1, ltr, btr*ltr])*pvy/(1+ltr*(1+btr))

    def negPVU(self,c):
        """0 self negative present utility value of
        any stream c for minimization call"""
        return  - self.PVU(c, self.beta)

    def kbar(self):
        '''Renegotiation cost necessaru to sustain full commitment competitive contract'''
        rho = self.rho
        if (rho == 1):
            rho = 0.999  # cheap trick to deal with special case
        btr = self.beta ** (1 / rho)
        c1F = np.sum(self.y) * btr / (1 + 2 * btr)
        A = (2 - (1 + btr) * ((1 + self.beta) / (1 + btr)) ** (1 / (1 - rho)))
        return A * c1F

    def bestreneg(self, c0):
        '''return period 1 consumption of reneg-proof contract.
        Assumes '''
        beta, rho = self.beta, self.rho
        Y = np.sum(self.y)
        c1F = (Y - c0) / 2     # for lower bound
        btr = beta ** (1 / rho)
        c1P = (Y - c0 - self.kappa) / (1 + btr)   #for upper bound
        ub = (1 + btr) * self.u(c1P)

        if self.kappa == 0:
            return c1P

        def U1(c1):
            return self.u(c1) + beta * self.u(Y - c0 - c1)

        def f(c1):
            return U1(c1) - ub

        guess = (c1F+c1P)/2
        return fsolve(f,guess )


    def renegC(self, c):
        """ Renegotiated Competitive contract offered to period-1-self
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)
        We distinguish from .reneg (ex-post monopoly) l"""
        beta, rho = self.beta, self.rho
        btr  =  beta**(1/rho)
        pv =  c[1] + c[2] - self.kappa
        cr1 = pv/(1+btr)
        cr2 = btr*cr1
        return np.array([c[0],cr1,cr2])

    def reneg_proof_cons(self,c):
        """ the renegotiation-proof constraint gain from renegotiation
        cannot exceed its cost kappa"""
        return  -(self.profit(self.reneg(c),self.y)
                  -  self.profit(c,self.y) - self.kappa)

    def reneg_proof_consC(self,c):
        """ renegotiation-proof constraint gain from renegotiation
        goes to customer """
        cr = self.renegC(c)[1:]   #last two periods
        return  -(self.PVU(cr,self.beta)
                  -  self.PVU(c[1:],self.beta))


    def participation_cons(self,c):
        return (self.PV(self.y) - self.PV(c))

    def reneg_proof2(self, monop_reg = False):
        """Alternative method: calculate renegotiation-proof contract that maxes 0-self's utility.
        supplies constraints to solver that bank can't profit too much and period 0 borrower participation
        the reneg_proof method incorporates the constraints into objective and is closer to the
        methods for finding contracts that are described in the paper"""
        if monop_reg:
            cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_cons },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        else:
            #print('reneg surplus to customer -- sensitive solns ')
            cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_consC },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        self.guess = self.ownsmooth()  # works best kappa=0
        res=minimize(self.negPVU, self.guess, method='COBYLA',
                     constraints = cons)

        return res.x

    def reneg_proof(self):
        """Find best renegotiation-proof contract by searching over
        subgame perfect responses """

        guess = self.reneg_proof2()[0]
        Y = np.sum(self.y)

        def f(c0):
            c1 = self.bestreneg(c0)
            return -(self.u(c0) + self.beta * (self.u(c1) + self.u(Y-c0-c1)))

        c0rp = minimize(f, guess, method='Nelder-Mead').x[0]
        c1rp = self.bestreneg(c0rp)
        c2rp = Y - c0rp - c1rp

        return np.array([c0rp, c1rp, c2rp])


if __name__ == "__main__":

    RHO   =  0.95  # for testing different values
    print("Base contract")
    c = Contract(beta = 0.5)
    c.rho = RHO
    c.y = [60, 120, 120]
    c.print_params()

    print("Monopoly contract")

    cM = Monopoly(beta = 0.5)
    cM.y = [80, 110, 110]
    cM.rho = RHO
    cM.print_params()

    print("Competitive contract")
    cC = Competitive(beta = 0.5)
    cC.rho = RHO
    cC.y = [80, 110, 110]
    cC.print_params()


    cCF = cC.fcommit()
    print("cCF: ",cCF)

    cCr = cC.reneg(cCF)
    print("cCF reneg: ",cCr)
    cC.guess = cCr
    cCRP = cC.reneg_proof()
    print("cCRP: ", cCRP)


    cMF = cM.fcommit()
    cMr = cM.reneg(cCF)
    cM.guess = cMr
    cMRP = cM.reneg_proof()

   # Analytic closed forms competitive
  #  A = cC.beta ** (1/cC.rho)
   # cA0 = (sum(cC.y) - cC.kappa)/(1+2*cA)
   # cCRPa = np.array([cA0, ])

    def ccrpa(C):
        B = C.beta**(1/C.rho)
        D = 1/(1+(1+B)*((C.beta+B)/(1+B))**(1/C.rho))
        print("D is equal to",D)
        c0 = sum(C.y)*D
        c1 = (sum(C.y)-c0)/(1+B)
        c2 = B* c1
        return np.array([c0, c1, c2])

    print("testing cCRP")
    print(cCRP.sum())
    print("reneg(cCRP):",cC.reneg(cCRP))
    print("PVU(cCRP) :",cC.PVU(cCRP,cC.beta))

    cCRPa = ccrpa(cC)
    print("PVU(cCRPa) :",cC.PVU(cCRPa,cC.beta))
    print("PVU(cCF) :",cC.PVU(cCF,cC.beta))
