#dynamic values are values that can be changed (all except the two on the end)
#bulk values are values that are the centerpiece of a calculation (all except one on each end)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from scipy.interpolate import make_interp_spline
from scipy.signal import sawtooth
import sympy as smp
import time
import random as rd
from sympy import symbols
from sympy.polys.polyfuncs import interpolate
import mpmath
from mpmath import mp
import numbers

from fhBoundaryLayer import *

Dx = smp.symbols('Δx')

class basic_field():
    def __init__(self, symbol_list):
        self.s = symbol_list.copy()
        self.create_derivatives()
        return
    def create_derivatives(self):
        self.sd = [(self.s[1] - self.s[0])/Dx]
        self.sdd = [0] #This pattern is only used for .sdd[0]
        k = 1
        while k < len(self.s) - 1:
            self.sd.append((self.s[k+1] - self.s[k])/Dx)
            self.sdd.append((self.s[k-1] + self.s[k+1] - 2*self.s[k])/Dx**2)
            k += 1
        self.sd.append(self.sd[-1])
        self.sdd.append(0)

class field:
    def __init__(self, label, ansatz, dynamic = True, domain = [-100, 100, 201], fixes = ["free", "free", "free", "free"], positive = False, bounds = [None, None]):
        #Calculate Bounds Function
        if bounds[0] == None or bounds[1] == None:
            self.reflect_func = lambda x: x
        else:
            mn = bounds[0]
            mx = bounds[1]
            avg = (mx + mn)/2
            scale = mx - avg
            self.reflect_func = lambda x: scale * sawtooth((((x-mn)*np.pi)/(2*scale)), .5) + avg
        
        self.ever_dynamic = dynamic
        self.Dx = Dx
        
        if fixes[0] == "free" and fixes[2] != "free":
            self.NeumannL = True
        else:
            self.NeumannL = False
        if fixes[1] == "free" and fixes[3] != "free":
            self.NeumannR = True
        else:
            self.NeumannR = False
            
        self.domain = domain
        self.xm = domain[0]
        self.xp = domain[1]
        self.N = domain[2]
        self.dx = (self.xp - self.xm)/(self.N - 3)
        self.x_list = np.linspace(self.xm - self.dx, self.xp + self.dx, self.N)
        self.ansatz = ansatz
        self.dynamic = dynamic
        self.fixes = fixes
        self.label = label
        self.positive = positive

        self.create_symbol_lists()
        self.initialize_values()
        self.create_derivative_symbol_lists()
        self.create_dynamic_s()
    
    def updatefixes(self, fixes):
        self.fixes = fixes
        self.initialize_values()
        self.create_dynamic_s()
    
    def initialize_values(self, reset_v = True, from_ansatz = False):
        if reset_v:
            self.v = [self.ansatz(self.x_list[i]) for i in range(self.N)]
        #Fix Left B.Cs
        if self.NeumannL:
            self.s[0] = self.s[2] - 2*self.Dx * self.fixes[2]
            self.s[1] = self.s[2] - self.Dx * self.fixes[2]
        elif from_ansatz:
            if self.fixes[0] != "free":
                self.s[1] = self.v[1]
            if self.fixes[2] != "free":
                self.s[0] = self.v[0] 
        else:
            if self.fixes[0] != "free":
                self.v[1] = self.fixes[0]
                self.s[1] = self.fixes[0]
            if self.fixes[2] != "free":
                self.v[0] = self.v[1] - self.dx*self.fixes[2]
                self.s[0] = self.v[1] - self.dx*self.fixes[2] 

        #Fix Right B.Cs
        if self.NeumannR:
            self.s[self.N - 1] = self.s[self.N - 2] + self.Dx * self.fixes[3]
        elif from_ansatz:
            if self.fixes[1] != "free":
                self.s[self.N - 2] = self.v[self.N - 2]
            if self.fixes[3] != "free":
                self.s[self.N-1] = self.v[self.N-1]
        else:            
            if self.fixes[1] != "free":
                self.v[self.N - 2] = self.fixes[1]
                self.s[self.N - 2] = self.fixes[1]
            if self.fixes[3] != "free":
                self.v[self.N-1] = self.v[self.N-2] + self.dx*self.fixes[3]
                self.s[self.N-1] = self.v[self.N-2] + self.dx*self.fixes[3]
    
    def create_dynamic_s(self):
        self.dynamic_s = []
        if not self.dynamic:
            return
        for smb in self.s:
            if isinstance(smb, smp.Symbol):
                self.dynamic_s.append(smb)            
                
    def create_symbol_lists(self):
        self.s = [smp.symbols(f"{self.label}_{n}", positive = self.positive) for n in range(self.N)]
        
    def create_derivative_symbol_lists(self):        
        self.sd = [(self.s[1] - self.s[0])/Dx] #Will never evaluate this
        self.sdd = [0] #Will never evaluate this
        k = 1
        while k < len(self.v) - 1:
            self.sd.append((self.s[k+1] - self.s[k])/Dx)
            self.sdd.append((self.s[k-1] + self.s[k+1] - 2*self.s[k])/Dx**2)
            k += 1
        self.sd.append(0)
        self.sdd.append(0) #repeat last
        
        self.idx_dict = {}
        k = 0
        while k < len(self.v):
            self.idx_dict[self.s[k]] = [k, self]
            k += 1
    def perturb_field(self, n):
        k = 0
        while k < self.N:
            if self.s[k] in self.dynamic_s:
                r = (rd.random() - .5)*n
                self.v[k] += r
                if self.positive:
                    if fld.v[k] <=0:
                        fld.v[k] += -2*r
            k += 1
        y.update_args()
    def make_spline(self, order):
        self.spline = make_interp_spline(self.x_list, self.v, k=order)
        
class full_field:
    def __init__(self, fields):
        self.fields = fields
        self.domain = fields[0].domain
        for fld in self.fields:
            if fld.domain != self.domain:
                print(f"Domain of field {fld.label}: {fld.label.domain} did not match domain of field {fields[0]}: {fields[0].domain}.")
        self.xm = self.domain[0]
        self.xp = self.domain[1]
        self.N = self.domain[2]
        self.dx = (self.xp - self.xm)/(self.N - 3)
        
        #create self.s list
        self.s = []
        self.mutable_s = []
        for fld in self.fields:
            if fld.ever_dynamic:
                self.mutable_s = self.mutable_s + fld.s
            self.s = self.s + fld.s        
        self.s = [x for x in self.s if isinstance(x, smp.Symbol)]

        #create self.idx_dict
        self.idx_dict = {}
        for field in self.fields:
            self.idx_dict = self.idx_dict | field.idx_dict # | is like + for dictionaries
        self.create_dynamic_s()
        self.update_args()
        self.init_args = self.args.copy()
            
    def create_dynamic_s(self):
        self.dynamic_s = []
        for fld in self.fields:
            fld.create_dynamic_s() #create dynamic symbol list for each dynamic field            
            self.dynamic_s = self.dynamic_s + fld.dynamic_s
        self.update_dynamic_v()
            
    def update_dynamic_v(self):
        self.dynamic_v = []
        for smb in self.dynamic_s:
            idx, fld = self.idx_dict[smb]
            self.dynamic_v.append(fld.v[idx])  
            
    def update_args(self):
        self.args = {Dx : self.dx}        
        for field in self.fields:
            k = 0
            while k < len(field.s):
                if isinstance(field.s[k], smp.Symbol):
                    self.args[field.s[k]] = field.v[k]
                k += 1
        self.update_dynamic_v()
        
    def fixfields(self, fixed):        
        for fld in self.fields:
            if fld in fixed:
                fld.dynamic = False
            else:
                fld.dynamic = True            
        self.create_dynamic_s()
        self.update_dynamic_v()
        
    def reset_values(self):
        for fld in self.fields:
            fld.initialize_values()
        self.update_args()

class Action:
    def __init__(self, L, fields, calcderiv = True, trap = False, lambdify_deriv = True, lambdify_type = "numpy", params = {}, boundary_term = smp.Integer(0), make_poly = False):
        self.lambdify_type = lambdify_type
        self.params = params
        print("Initializing Action")
        self.fields = fields
        self.field_list = self.fields.fields
        self.boundary_term = boundary_term
        self.L = L
        self.trap = trap
        self.CreateL_list_general(self.fields.N)
        self.CreateL_list()
        self.CreateExpr()
        if calcderiv:
            print("Calculating First Derivatives")
            self.calc_first_deriv(make_poly = make_poly)
            print()
            print("Calculating Second Derivatives")
            self.calc_second_deriv()
            if lambdify_deriv:
                print("Lambdifying Derivatives")
                self.lambdify_dSdy()
        return
    def CreateL_list_general(self, N):
        self.L_list_general = [smp.S.Zero] + [self.L(i) for i in range(1, N-1)] + [smp.S.Zero]
        #Implement Trapezoid Rule
        if self.trap:
            self.L_list_general[1] = .5*self.L_list_general[1]
            self.L_list_general[-2] = .5*self.L_list_general[-2]
    def CreateL_list(self):
        self.L_list = []
        for Li in self.L_list_general:
            self.L_list.append(Li.xreplace(self.params))
    def CreateExpr(self):
        self.expr = Dx * sum(self.L_list) + self.boundary_term
        #self.expr_func = smp.lambdify(self.expr, self.fields.dynamic_s)
    def UpdateParameters(self, params):
        self.params = params
        self.CreateL_list()
        print("Calculating First Derivatives")
        self.calc_first_deriv()
        print("Calculating Second Derivatives")
        self.calc_second_deriv()
        print("Lambdifying Derivatives")
        self.lambdify_dSdy()
    def CalcFullAction(self):
        return self.expr.xreplace(self.fields.args)
        #return self.expr_func(*self.fields.dynamic_v)
    def PrintFullAction(self):
        print(self.CalcFullAction())
    def calc_first_deriv(self, make_poly = False):
        self.dSdy = {}
        for fld in self.field_list:
            if fld.ever_dynamic:
                print(f"Calculating First Derivatives for {fld.label}")
                for yd in fld.s:
                    if isinstance(yd, smp.Symbol):
                        if make_poly:
                            self.dSdy[yd] = smp.fraction(smp.together(smp.diff(self.expr, yd)))[0]
                        else:
                            self.dSdy[yd] = smp.diff(self.expr, yd)
    def calc_second_deriv(self):
        self.d2Sdydy = {}
        for fld in self.field_list:
            if fld.ever_dynamic:
                print(f"Calculating Second Derivatives for {fld.label}")
                for yi in fld.s:
                    if isinstance(yi, smp.Symbol):
                        self.d2Sdydy[yi] = {}
                        dep_symb = self.dSdy[yi].free_symbols
                        for yj in self.fields.s:
                            if yj in dep_symb:
                                try:
                                    self.d2Sdydy[yi][yj] = self.d2Sdydy[yj][yi]
                                except:
                                    self.d2Sdydy[yi][yj] = smp.diff(self.dSdy[yi], yj)
                            else:
                                self.d2Sdydy[yi][yj] = smp.Integer(0)
    def popEM(self):
        self.E = np.array([self.lambda_dSdy[yi](*self.fields.dynamic_v) for yi in self.fields.dynamic_s])        
        self.M = np.array([[self.lambda_d2Sdydy[yj][yi](*self.fields.dynamic_v) for yi in self.fields.dynamic_s] for yj in self.fields.dynamic_s])
        
    def calcdelta(self, method = "Newton"):
        self.popEM()
        if len(self.E) == 0:
            self.delta = 0
        elif method == "Newton":
            if self.lambdify_type == "mpmath":
                M_mp = mpmath.matrix(self.M)
                E_mp = mpmath.matrix(self.E)
                self.delta = mpmath.lu_solve(M_mp, -E_mp)
            else:
                self.delta = np.linalg.solve(self.M, -self.E)
        elif method == "Gradient":
            self.delta = -0.5*self.E
            norm = sum(self.delta**2)
            
    def FindMaxDeltaScale(self, calculatedelta = False, PrintBound = False):
        MaxScale = 2
        MinScale = -1
        if calculatedelta:
            self.calcdelta(method = calculatedelta)
        for i in range(len(self.fields.dynamic_s)):
            var = self.fields.dynamic_s[i]
            idx, field = self.fields.idx_dict[var]
            if field.s[idx].is_positive:
                NewBound = -field.v[idx]/self.delta[i]
                if NewBound > 0:
                    if NewBound < MaxScale:
                        MaxScale = NewBound
                elif NewBound < 0:
                    if NewBound > MinScale:
                        MinScale = NewBound
        return MinScale, MaxScale

    def Squares(self, var = "unspecified"):
        if var == "unspecified":
            var = self.fields.dynamic_v
        return_value = 0
        for dSdy in self.lambda_dSdy.values():
            return_value += (dSdy(*var))**2
        return return_value
    
    def CheckDelta(self, deltaMin, deltaMax, output = False):
        eig = np.linalg.eig(self.M)
        vals = eig.eigenvalues
        vecs = eig.eigenvectors.T
        
        deltaMin = float(deltaMin) #convert to floats in case they are mpmath types
        deltaMax = float(deltaMax)
        print(f"Calculating Starting Squares")
        Starting_Sq = self.Squares()

        if deltaMax > 1:
            max_step = 1
        else:
            max_step = .95*deltaMax
        step = max_step

        Plotx = [0]
        Ploty = [Starting_Sq]

        while True:
            if step < 1e-4:
                if False:
                    step = max_step
                    plt.scatter(Plotx, Ploty)
                    plt.show()
                    Plotx = [0]
                    Ploty = [Starting_Sq]
    
                    abs_vals = abs(vals)
                    idx = list(abs_vals).index(min(abs_vals))
                    plt.plot(self.field_list[0].x_list[2:self.fields.N - 2], self.delta)
                    
                    print(f"Removed {vals[idx]:.2f}")
    
                    vals = np.delete(vals, idx)
                    vecs = np.delete(vecs, idx, axis=0)

                    self.delta = -1*(self.E@vecs[0])*(vecs[0]/vals[0])
                    idx = 1
                    while idx < len(vecs):
                        self.delta += -1*(self.E@vecs[idx])*(vecs[idx]/vals[idx])
                        idx += 1

                    plt.plot(self.field_list[0].x_list[2:self.fields.N - 2], self.delta)
                    plt.show()

                    print(f"Removed eigen direction.  Length of vals now {len(vals)}")
                else:
                    print("Modified NR failed")
                    return 0                
                
            dyvar = []
            for i in range(len(self.fields.dynamic_s)):
                dyvar.append(self.fields.dynamic_v[i] + step*self.delta[i])
            newSq = self.Squares(var = dyvar)
            Plotx.append(step)
            Ploty.append(newSq)
            if newSq < Starting_Sq:
                min_idx = Ploty.index(min(Ploty))
                if output:
                    print(f"delta was scaled by {step}")
                    print(f"Old Squares Value was: {self.Squares():.2e}")
                    print(f"New Squares Value is: {newSq:.2e}")
                return step
            else:
                step = step/1.5            
        
        return 0
        
    def updatefields(self, deltaScale = 1, reflect = True):
        #Update Dynamic Variables
        for i in range(len(self.fields.dynamic_s)):
            var = self.fields.dynamic_s[i]
            idx, field = self.fields.idx_dict[var]
            if reflect:
                field.v[idx] = field.reflect_func(field.v[idx] + self.delta[i]*deltaScale)
            else:
                field.v[idx] = field.v[idx] + self.delta[i]*deltaScale
        self.fields.update_args()
        #Update Variables that are associated to Dynamic Symbols
        for fld in self.field_list:
            k = 0
            while k < len(fld.s):
                if isinstance(fld.s[k], numbers.Number): #For fixed variables update values to fixed value in symbol list
                    fld.v[k] = fld.s[k]
                else:
                    fld.v[k] = fld.s[k].xreplace(self.fields.args)
                k += 1
                
    def runNR(self, steps, dynamic = "default", PureNR = True, lambdify_first = True, revert_if_failed = True, output = True):
        self.fields.update_args()
        if dynamic == "default":
            self.fields.create_dynamic_s()
        else:
            self.fields.dynamic_s = dynamic.copy()
            self.fields.update_dynamic_v()
        if lambdify_first:
            #print("Lambdifying Derivatives")
            self.lambdify_dSdy()
        #print("Running Newton Raphson")

        v_old = []
        for smb in self.fields.dynamic_s:
            v_old.append(self.fields.args[smb])
        worked = False
        k = 0        
        while k < steps:
            self.calcdelta() #This is where NR is actually run           
            if PureNR:
                self.updatefields(deltaScale = 1)
            else:
                dmin, dmax = self.FindMaxDeltaScale(calculatedelta = True)
                scale = self.CheckDelta(dmin, dmax, output = output)
                if scale == 0:
                    break
                self.updatefields(deltaScale = scale)
            k += 1
            Sq = self.Squares()
            if Sq < 1e-12:
                if output:
                    print(f"Newton Raphson Converged to Square Value of {Sq:.2e} after {k} steps")
                worked = True
                return worked        
        if not worked:
            if output:
                print(f"Newton Raphson Only Converged to Square Value of {Sq:.2e}")
            if revert_if_failed:
                self.updatefields_LS(v_old)
            return worked
    
    def lambdify_dSdy(self, lambdify_type = "default"):
        if lambdify_type == "default":
            lambdify_type = self.lambdify_type

        #Make a dictionary to replace all the non-dynamic symbols with values

        self.fields.update_args()
        non_dynamic = self.fields.args.copy()
        for smb in self.fields.dynamic_s:
            non_dynamic.pop(smb, None)

        self.lambda_dSdy = {}
        for smb in self.fields.dynamic_s:
            dSdy = self.dSdy[smb].xreplace(non_dynamic)
            self.lambda_dSdy[smb] = smp.lambdify(self.fields.dynamic_s, dSdy, lambdify_type)

        
        zero_func = smp.lambdify(self.fields.dynamic_s, 0, "numpy")
        self.lambda_d2Sdydy = {}
        for y1 in self.fields.dynamic_s:
            self.lambda_d2Sdydy[y1] = {}
            for y2 in self.fields.dynamic_s:
                if self.d2Sdydy[y1][y2] == 0:
                    self.lambda_d2Sdydy[y1][y2] = zero_func
                else:
                    d2Sdydy = self.d2Sdydy[y1][y2].xreplace(non_dynamic)
                    self.lambda_d2Sdydy[y1][y2] = smp.lambdify(self.fields.dynamic_s, d2Sdydy, lambdify_type)     
    
    def variations(self, dynam_v):
        return_list = []
        for smb in self.fields.dynamic_s:
            return_list.append(self.lambda_dSdy[smb](*dynam_v))
        return np.array(return_list)
        
    def variations_sqrd(self, dynam_v):
        dSdy_list = self.variations(dynam_v)
        return np.dot(dSdy_list, dSdy_list)
        
    def updatefields_LS(self, x):
        x_idx = 0
        for smb in self.fields.dynamic_s:
            idx, field = self.fields.idx_dict[smb]
            field.v[idx] = x[x_idx]
            x_idx += 1
        self.fields.update_args()
        for fld in self.field_list:
            k = 0
            while k < len(fld.s):
                if isinstance(fld.s[k], numbers.Number):
                    fld.v[k] = fld.s[k]
                else:
                    fld.v[k] = fld.s[k].xreplace(self.fields.args)
                k += 1
    
    def runLS(self, fixed, remove_extra = "no"):
        self.fields.fixfields(fixed)
        print("Lambdifying dSdy")
        self.lambdify_dSdy(lambdify_type = "numpy")

        print("Defining Seed and Bounds")
        v_ansatz = []
        for smb in self.fields.dynamic_s:
            v_ansatz.append(self.fields.args[smb])
        
        lower_bounds = []
        upper_bounds = []
        for smb in self.fields.dynamic_s:
            if smb.is_positive:
                lower_bounds.append(0)
            else:
                lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)

        print("Running Least Squares")
        sol = least_squares(self.variations, v_ansatz, bounds=(lower_bounds, upper_bounds), max_nfev=100000, ftol=1e-15, gtol=1e-15, xtol=1e-15)
        self.sol = sol
        print(sol.message)
        self.updatefields_LS(sol.x)

    def Look_for_Global_Solution(self, fixed):
        self.fields.fixfields(fixed)
        self.lambdify_dSdy()
        
        # Set bounds
        bound_list = []
        for smb in self.fields.dynamic_s:
            if smb.is_positive:
                bound_list.append((0, 500))
            else:
                bound_list.append((-150, 200))

        self.sol_de = differential_evolution(self.variations_sqrd, bound_list, maxiter=2000, tol=1e-7)
        print(self.variations_sqrd(self.sol_de.x))
    
    def replace_functions(self, expr, idx):    
        if idx == 3:
            #Replace Ricci
            expr = expr.replace(r"\operatorname{R_{2}}{\left(f_{n-2},f_{n-1},f_{n},h_{n-1},h_{n} \right)}", r"R_{n-1}")
            expr = expr.replace(r"\operatorname{R_{3}}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"R_{n}")
            expr = expr.replace(r"\operatorname{R_{4}}{\left(f_{n},f_{n+1},f_{n+2},h_{n+1},h_{n+2} \right)}", r"R_{n+1}")
            expr = expr.replace(r"\operatorname{R_{3}}^{2}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"(R_{n})^2")
            #Replace D'Alembertian
            expr = expr.replace(r"\square\varphi_{2}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\varphi_{n-2},\varphi_{n-1},\varphi_{n} \right)}", r"\square^2 \varphi_{n-1}")   
            expr = expr.replace(r"\square\varphi_{3}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\varphi_{n-1},\varphi_{n},\varphi_{n+1} \right)}", r"\square^2 \varphi_{n}")
            expr = expr.replace(r"\square\varphi_{4}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\varphi_{n},\varphi_{n+1},\varphi_{n+2} \right)}", r"\square^2 \varphi_{n+1}")
            expr = expr.replace(r"\square\psi_{2}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\psi_{n-2},\psi_{n-1},\psi_{n} \right)}", r"\square^2 \psi_{n-1}")   
            expr = expr.replace(r"\square\psi_{3}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"\square^2 \psi_{n}")
            expr = expr.replace(r"\square\psi_{4}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\psi_{n},\psi_{n+1},\psi_{n+2} \right)}", r"\square^2 \psi_{n+1}")
            #Replace D'Alembertian Squared
            expr = expr.replace(r"\square\psi_{3}^{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"(\square^2 \psi_n)^2")
        elif idx == 2:
            #Replace Ricci
            expr = expr.replace(r"\operatorname{R_{1}}{\left(f_{n-2},f_{n-1},f_{n},h_{n-1},h_{n} \right)}", r"R_{n-1}")
            expr = expr.replace(r"\operatorname{R_{2}}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"R_{n}")
            expr = expr.replace(r"\operatorname{R_{3}}{\left(f_{n},f_{n+1},f_{n+2},h_{n+1},h_{n+2} \right)}", r"R_{n+1}")
            expr = expr.replace(r"\operatorname{R_{2}}^{2}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"(R_{n})^2")
            #Replace D'Alembertian
            expr = expr.replace(r"\square\varphi_{1}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\varphi_{n-2},\varphi_{n-1},\varphi_{n} \right)}", r"\square^2 \varphi_{n-1}")   
            expr = expr.replace(r"\square\varphi_{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\varphi_{n-1},\varphi_{n},\varphi_{n+1} \right)}", r"\square^2 \varphi_{n}")
            expr = expr.replace(r"\square\varphi_{3}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\varphi_{n},\varphi_{n+1},\varphi_{n+2} \right)}", r"\square^2 \varphi_{n+1}")
            expr = expr.replace(r"\square\psi_{1}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\psi_{n-2},\psi_{n-1},\psi_{n} \right)}", r"\square^2 \psi_{n-1}")   
            expr = expr.replace(r"\square\psi_{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"\square^2 \psi_{n}")
            expr = expr.replace(r"\square\psi_{3}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\psi_{n},\psi_{n+1},\psi_{n+2} \right)}", r"\square^2 \psi_{n+1}")
            #Replace D'Alembertian Squared
            expr = expr.replace(r"\square\varphi_{2}^{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\varphi_{n},\varphi_{n+1} \right)}", r"(\square^2 \varphi_n)^2")
            expr = expr.replace(r"\square\psi_{2}^{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"(\square^2 \psi_n)^2")
        elif idx==4:
            #Replace Ricci
            expr = expr.replace(r"\operatorname{R_{3}}{\left(f_{n-2},f_{n-1},f_{n},h_{n-1},h_{n} \right)}", r"R_{n-1}")
            expr = expr.replace(r"\operatorname{R_{4}}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"R_{n}")
            expr = expr.replace(r"\operatorname{R_{5}}{\left(f_{n},f_{n+1},f_{n+2},h_{n+1},h_{n+2} \right)}", r"R_{n+1}")
            expr = expr.replace(r"\operatorname{R_{4}}^{2}{\left(f_{n-1},f_{n},f_{n+1},h_{n},h_{n+1} \right)}", r"(R_{n})^2")
            #Replace D'Alembertian
            expr = expr.replace(r"\square\varphi_{3}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\varphi_{n-2},\varphi_{n-1},\varphi_{n} \right)}", r"\square^2 \varphi_{n-1}")   
            expr = expr.replace(r"\square\varphi_{4}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\varphi_{n-1},\varphi_{n},\varphi_{n+1} \right)}", r"\square^2 \varphi_{n}")
            expr = expr.replace(r"\square\varphi_{5}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\varphi_{n},\varphi_{n+1},\varphi_{n+2} \right)}", r"\square^2 \varphi_{n+1}")
            expr = expr.replace(r"\square\psi_{3}{\left(f_{n-1},f_{n},h_{n-1},h_{n},\psi_{n-2},\psi_{n-1},\psi_{n} \right)}", r"\square^2 \psi_{n-1}")   
            expr = expr.replace(r"\square\psi_{4}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"\square^2 \psi_{n}")
            expr = expr.replace(r"\square\psi_{5}{\left(f_{n+1},f_{n+2},h_{n+1},h_{n+2},\psi_{n},\psi_{n+1},\psi_{n+2} \right)}", r"\square^2 \psi_{n+1}")
            #Replace D'Alembertian Squared
            expr = expr.replace(r"\square\varphi_{4}^{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\varphi_{n},\varphi_{n+1} \right)}", r"(\square^2 \varphi_n)^2")
            expr = expr.replace(r"\square\psi_{4}^{2}{\left(f_{n},f_{n+1},h_{n},h_{n+1},\psi_{n-1},\psi_{n},\psi_{n+1} \right)}", r"(\square^2 \psi_n)^2")

        #Replace f derivatives
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} R_{n-1}", r"\frac{\partial R_{n-1}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} R_{n}", r"\frac{\partial R_{n}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} R_{n+1}", r"\frac{\partial R_{n+1}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \varphi_{n-1}", r"\frac{\partial \square^2 \varphi_{n-1}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \varphi_{n}", r"\frac{\partial \square^2 \varphi_{n}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \varphi_{n+1}", r"\frac{\partial \square^2 \varphi_{n+1}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \psi_{n-1}", r"\frac{\partial \square^2 \psi_{n-1}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \psi_{n}", r"\frac{\partial \square^2 \psi_{n}}{\partial f_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial f_{n}} \square^2 \psi_{n+1}", r"\frac{\partial \square^2 \psi_{n+1}}{\partial f_{n}} ")

        #Replace h derivatives
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} R_{n-1}", r"\frac{\partial R_{n-1}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} R_{n}", r"\frac{\partial R_{n}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} R_{n+1}", r"\frac{\partial R_{n+1}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \varphi_{n-1}", r"\frac{\partial \square^2 \varphi_{n-1}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \varphi_{n}", r"\frac{\partial \square^2 \varphi_{n}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \varphi_{n+1}", r"\frac{\partial \square^2 \varphi_{n+1}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \psi_{n-1}", r"\frac{\partial \square^2 \psi_{n-1}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \psi_{n}", r"\frac{\partial \square^2 \psi_{n}}{\partial h_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial h_{n}} \square^2 \psi_{n+1}", r"\frac{\partial \square^2 \psi_{n+1}}{\partial h_{n}} ")

        #Replace phi derivatives
        expr = expr.replace(r"\frac{\partial}{\partial \varphi_{n}} \square^2 \varphi_{n-1}", r"\frac{\partial \square^2 \varphi_{n-1}}{\partial \varphi_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial \varphi_{n}} \square^2 \varphi_{n}", r"\frac{\partial \square^2 \varphi_{n}}{\partial \varphi_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial \varphi_{n}} \square^2 \varphi_{n+1}", r"\frac{\partial \square^2 \varphi_{n+1}}{\partial \varphi_{n}} ")

        #Replace psi derivatives
        expr = expr.replace(r"\frac{\partial}{\partial \psi_{n}} \square^2 \psi_{n-1}", r"\frac{\partial \square^2 \psi_{n-1}}{\partial \psi_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial \psi_{n}} \square^2 \psi_{n}", r"\frac{\partial \square^2 \psi_{n}}{\partial \psi_{n}} ")
        expr = expr.replace(r"\frac{\partial}{\partial \psi_{n}} \square^2 \psi_{n+1}", r"\frac{\partial \square^2 \psi_{n+1}}{\partial \psi_{n}} ")

        expr = expr.replace(r"2.0", r"2")
        expr = expr.replace(r"4.0", r"4")
        expr = expr.replace("Δ", r"\Delta ")
        return expr

    def break_into_terms(self, expr):
        num, den = smp.fraction(expr)
        num_terms = num.as_ordered_terms()
        return_list = []
        for term in num_terms:
            return_list.append(smp.simplify(term/den))
        return return_list

    def Latexify(self, expr):
        term_list = self.break_into_terms(expr)
        latex_list = []
        for term in term_list:
            tex_term = smp.latex(term)
            tex_term = self.replace_functions(tex_term)
            latex_list.append(tex_term)
        return latex_list

    def get_term_string(self, expr):
        flist = self.Latexify(expr)
        output_string = ""
        for term in flist:
            output_string += term + r"\nonumber \\ &+"

        print(output_string)
        print()
    
    def make_print_args(self, idx):
        fnmm, fnm, fn, fnp, fnpp, hnmm, hnm, hn, hnp, hnpp = smp.symbols(
            "f_{n-2} f_{n-1} f_{n} f_{n+1} f_{n+2} h_{n-2} h_{n-1} h_{n} h_{n+1} h_{n+2}", positive=True)
        metric_args = {
            f.s[idx-2] : fnmm, f.s[idx-1] : fnm, f.s[idx] : fn, f.s[idx+1] : fnp, f.s[idx+2] : fnpp, 
            h.s[idx-2] : hnmm, h.s[idx-1] : hnm, h.s[idx] : hn, h.s[idx+1] : hnp, h.s[idx+2] : hnpp
        }

        wnm, wn, wnp = smp.symbols("w_{n-1} w_n w_{n+1}")
        w_args = {w.s[idx-1] : wnm, w.s[idx] : wn, w.s[idx+1] : wnp}

        phnmm, phnm, phn, phnp, phnpp, psinmm, psinm, psin, psinp, psinpp = smp.symbols(
            r"\varphi_{n-2} \varphi_{n-1} \varphi_{n} \varphi_{n+1} \varphi_{n+2} \psi_{n-2} \psi_{n-1} \psi_{n} \psi_{n+1} \psi_{n+2}", positive=True)
        con_args = {
            ph.s[idx-2] : phnmm, ph.s[idx-1] : phnm, ph.s[idx] : phn, ph.s[idx+1] : phnp, ph.s[idx+2] : phnpp,
            psi.s[idx-2] : psinmm, psi.s[idx-1] : psinm, psi.s[idx] : psin, psi.s[idx+1] : psinp, psi.s[idx+2] : psinpp}

        self.print_args = metric_args | w_args | con_args
        
    
    def printActionDerivatives(self, idx):
        self.make_print_args(idx)

        print("f derivative:")
        fexpr = smp.simplify(S_part.dSdy[f.s[idx]].xreplace(args))
        self.get_term_string(fexpr)
        print("h derivative:")
        hexpr = smp.simplify(S_part.dSdy[h.s[idx]].xreplace(args))
        self.get_term_string(hexpr)
        print()
        print("w derivative:")
        wexpr = smp.simplify(S_part.dSdy[w.s[idx]].xreplace(args))
        self.get_term_string(wexpr)
        print()
        print("phi derivative:")
        phexpr = smp.simplify(S_part.dSdy[ph.s[idx]].xreplace(args))
        self.get_term_string(phexpr)
        print()
        print("psi derivative:")
        psexpr = smp.simplify(S_part.dSdy[psi.s[idx]].xreplace(args))
        self.get_term_string(psexpr)
        print()
        
