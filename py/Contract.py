# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:26:19 2015

@author: Jonathan
"""
import numpy as np
from scipy.optimize import minimize


class Contract(object):
    """ Base Class for defining contracts  """                                                    
    def __init__(self,beta,y=None):         # constructor method to set default params. 
        self.beta  = beta                   # present bias in β-δ framework
        self.rho   = 0.95                   # 1/rho = elasticity of substitution
        if y is None:                     
            self.y = np.array([100,100,100])
        else:
            self.y = y     
        self.r     = 0.0                     # bank's opportunity cost of funds
        self.delta = 1/(1+self.r)            # agent's psychic discount factor     
    
    def print_params(self):    
        """ print out parameters """
        params = vars(self)
        for p in sorted(params):    # print attributes alphabetically  
            print("{0:<7} : {1}".format(p,params[p]))
        
    def u(self,ct):
        """ utility function """
        return  ( (1/(1-self.rho)) * ct**(1-self.rho)  )
    
    def PV(self,c):         
        """discounted present value of any stream c"""
        return  c[0] + sum((c[t]* (1/(1+self.r))**t for t in range(1,len(c))) )  
    
    def PVU(self,c,beta):         
        """discounted present utility value of any stream c"""        
        return  self.u(c[0]) + beta * sum((self.u(c[t])*self.delta**t  for t in range(1,len(c))) )
           
    def profit(self,c,y):   
        """ present value of lender profits when exchanges c for y""" 
        return  self.PV(y)-self.PV(c)
    
    def negprofit(self,c):      
        """ Negative profits (for minimization)"""
        return  -(self.PV(self.y) - self.PV(c))
       
    def indif(self, ubar, beta):
        """ returns function u(c1, c2) for graphing indifference curves in c1-c2 space.  
        if beta = 1, will describe self 0's preferences """
        def idc(c1):
            return np.array((((1-self.rho)/(beta*self.delta))
                  *(ubar-self.u(c1)))**(1/(1-self.rho)))
        return idc
    
    def isoprofit(self, prfbar, y):
        """ returns function profit(c1, c2) for graphing isoprofit lines in c1-c2 space.  
        """    
        def isoprf(c1):
            """isoprofit function isoprf(c1) """
            return np.array(y[1] + y[2] - prfbar) - c1       
        return isoprf
        
        
class Monopoly(Contract):                    # build on contract class
    """ Class for solving Monopoly equilibrium contracts  """                                                                                                        
    def __init__(self,beta, y=None):
        super(Monopoly,self).__init__(beta)    # make sure inherits parent class properties
        self.kappa  = 0                                # cost of renegotiation    
        self.guess  = self.y                           # initial guess for solver
        
    def fcommit(self):                                
        """monopolist optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        A = ((self.PVU(self.y,self.beta) *(1-self.rho) )**(1/(1-self.rho)) )
        B = (1 + self.beta**(1/self.rho) * (self.delta + self.delta**2))**(1/(self.rho-1))
        c0 = A*B
        c1 = c0 *(self.beta * self.delta *(1+self.r)  )**(1/self.rho)
        c2 = c0 *(self.beta * self.delta**2 *(1+self.r)**2  )**(1/self.rho) 
        return np.array([c0,c1,c2])
    
    def reneg(self,c):                 
        """ Renegotiated contract offered to period-1-self   
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        PU =  self.u(c[1]) + self.beta*self.delta*self.u(c[2])
        A  =  (PU *(1-self.rho) )**(1/(1-self.rho)) 
        B = (1 + self.beta**(1/self.rho) * self.delta)**(1/(self.rho-1))
        cr0 = c[0]
        cr1 = A*B
        cr2 = cr1 *(self.beta * self.delta *(1+self.r)  )**(1/self.rho)
        return np.array([cr0,cr1,cr2])
    
    def reneg_proof_cons(self,c):                 
        """ the renegotiation-proof constraint gain from renegotiation 
        cannot exceed its cost kappa"""                                           
        return  -(self.profit(self.reneg(c),self.y)
                  -  self.profit(c,self.y) - self.kappa)     
    
    def participation_cons(self,c):
        return self.PVU(c,self.beta)  - self.PVU(self.y,self.beta)
        
    def reneg_proof(self):                           
        """calculate renegotiation-proof contract 
        supplies constraints to solver that bank can't profit too much
        and period 0 borrower participation"""
        
        cons = ({'type': 'ineq',                          
                 'fun' : self.reneg_proof_cons },    
                {'type': 'ineq', 
                 'fun' : self.participation_cons })      
        res=minimize(self.negprofit, self.guess, method='COBYLA',                     constraints = cons)
        
        return res

class Competitive(Contract):                    # build on contract class
    """ Class for solving competitive equilibrium contracts  """                                                    
    def __init__(self, beta):
        super(Competitive,self).__init__(beta)    # make sure inherits parent class properties
        self.kappa  = 0                                # cost of renegotiation  
        self.guess  = self.y                           # initial guess for solver
        
    def fcommit(self):                                
        """competitive optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        A = self.beta**(1/self.rho)
        B = self.PV(self.y)
        C = 1 + 2*A
        c0 = B/C
        c1 = A * c0  
        c2 = A * c0
        return np.array([c0,c1,c2])
    
    def reneg(self,c):                 
        """ Renegotiated contract offered to period-1-self   
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        PV =  c[1] + c[2]
        A  =  self.beta**(1/self.rho) 
        cr0 = c[0]
        cr1 = PV/(1+A)
        cr2 = A*cr1
        return np.array([cr0,cr1,cr2])
    
    def reneg_proof_cons(self,c):                 
        """ the renegotiation-proof constraint gain from renegotiation 
        cannot exceed its cost kappa"""                                           
        return  -(self.profit(self.reneg(c),self.y) 
                  - self.profit(c,self.y) - self.kappa)     
    
    def participation_cons(self,c):
        return self.PVU(c,self.beta)  - self.PVU(self.y,self.beta)
        
    def reneg_proof(self):                           
        """calculate renegotiation-proof contract 
        supplies constraints to solver that bank can't profit too much
        and period 0 borrower participation"""
        
        cons = ({'type': 'ineq',                          
                 'fun' : self.reneg_proof_cons },    
                {'type': 'ineq', 
                 'fun' : self.participation_cons })      
        res=minimize(self.negprofit, self.guess, method='COBYLA',                     constraints = cons)
        
        return res

if __name__ == "__main__":
    print("Base contract")
    c = Contract(beta = 0.7)
    c.y = [60, 120, 120]
    c.print_params()
    
    print("Monopoly contract")
    cM = Monopoly(beta = 0.7)
    cM.y = [60, 120, 120]
    cM.print_params()
    
    print("Competitive contract")
    cC = Competitive(beta = 0.7)
    cC.print_params()
    