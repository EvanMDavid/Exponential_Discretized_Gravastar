from GravastarClass import *

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import BPoly
import ipywidgets as widgets
import sympy as smp

ln = np.log

alpha_s, beta_s, gamma_s = smp.symbols("alpha, beta, gamma")

def make_parameters_global(a, b, g, e):
    global epsilon
    alpha = a
    beta = b
    gamma = g
    epsilon = e

def define_peculiar_variables(xm, xp, alpha, beta, gamma, epsilon):
    def aF(x):
        if x == 0:
            return 1
        z = np.sqrt(1 + abs(x/100))
        return (3*z**2 - 1)/2 - 3/8 * z**2 * (z**2 - 1) * (ln((z + 1)/(z - 1)))**2
    
    make_parameters_global(alpha, beta, gamma, epsilon)

    #Define Boundary Condition Parameters
    global c0, cinf, cH, cF, cS
    c0 = 1
    cH = 1
    cS = 1
    cinf = 3/(gamma * aF(xm)) + c0 + (cH - cS)*ln(epsilon * xp)
    cF = 0
    
    global phxm, phxp, psixm, psixp, dphixm, dphixp, dpsixm, dpsixp    

    original = True
    if original:    
        phxm = c0 + cH * ln(-epsilon * xm / 2)
        phxp = cinf + cS * ln(epsilon * xp)
        psixm = 0
        psixp = 0
        
        dphixm = cH / xm
        dphixp = cS / xp
        dpsixm = 0
        dpsixp = 0
    else:
        phxm = c0
        phxp = cinf
        psixm = -1/2*(phxm + cF)
        psixp = -1/2*(phxp + cF)
        
        dphixm = 0
        dphixp = 0
        dpsixm = 0
        dpsixp = 0

def create_ansatz(xm, xp, T_array = 30):
    global f_ansatz, b_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz

    def f_ansatz(x, T = T_array):
        if x <= -T:
            return -.5*x
        if x >= T:
            return x
        else:
            return float(CubicHermiteSpline([-T, T], [.5*T, T], [-.5, 1])(x))

    def h_ansatz(x, T = T_array):
        if x <= -T:
            return -2*x
        if x >= T:
            return x
        else:
            return float(CubicHermiteSpline([-T, T], [2*T, T], [-2, 1])(x))
            
    def b_ansatz(x):
        return np.sqrt(h_ansatz(x)/f_ansatz(x))
  
    def b_ansatz_complex(x, array = T_array):
        x1, db0, ddb0, b1, db1, ddb1, ddb2 = array
        y0 = [2, db0, ddb0]
        y1 = [b1, db1, ddb1]
        y2 = [1, 0, ddb2]
        if x <= -100:
            return 2
        elif x >= 100:
            return 1
        elif x <= x1:
            return float(BPoly.from_derivatives([-100, x1], [y0, y1], orders=5)(x))
        else:
            return float(BPoly.from_derivatives([x1, 100], [y1, y2], orders=5)(x))            

    def phi_ansatz(x):
        if x == 0:
            return -100
        elif x < 0:
            return c0 + cH * ln(-epsilon * x / 2)
        elif x > 0:
            return cinf + cS * ln(epsilon * x)
            
    def psi_ansatz(x):
        return 0
        
    def w_ansatz(x, T = 0):
        if abs(x) < T:
            return CubicHermiteSpline([-T, T], [-T, 0], [1, 0])(x)
        if x < 0:
            return x
        else:
            return 0
            
    def aF_ansatz(x):
        if x == 0:
            return 1
        z = np.sqrt(1 + abs(x/100))
        return (3*z**2 - 1)/2 - 3/8 * z**2 * (z**2 - 1) * (ln((z + 1)/(z - 1)))**2
    
    return [f_ansatz, b_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz]

def create_fields(ansatz, field_domain):
    xm = field_domain[0]
    xp = field_domain[1]
    global f, b, ph, psi, w, aF, y
    f = field("f", ansatz[0], domain = field_domain, fixes = [abs(xm)/2, xp, -1/2, 1], positive = False, dynamic = True)
    b = field("b", ansatz[1], domain = field_domain, fixes = [2, 1, 0, 0], positive = False, dynamic = True)
    ph = field("varphi", ansatz[2], domain = field_domain, fixes = [phxm, phxp, dphixm, dphixp])
    psi = field("psi", ansatz[3], domain = field_domain, fixes = [psixm, psixp, dpsixm, dpsixp])
    w = field("w", ansatz[4], domain = field_domain, fixes = ["free", 0, 1, 0])

    if False:
        f = field("f", ansatz[0], domain = field_domain, fixes = ["free", "free", "free", "free"], positive = False, dynamic = True)
        b = field("b", ansatz[1], domain = field_domain, fixes = ["free", "free", "free", "free"], positive = False, dynamic = True)
        ph = field("varphi", ansatz[2], domain = field_domain, fixes = ["free", "free", "free", "free"])
        psi = field("psi", ansatz[3], domain = field_domain, fixes = ["free", "free", "free", "free"])
        w = field("w", ansatz[4], domain = field_domain, fixes = ["free", "free", "free", "free"])
    
    aF = field("aF", ansatz[5], domain = field_domain, dynamic = False)
    
    y = full_field([f, b, ph, psi, w, aF])

def fields_from_scratch(xm, xp, N, T_array = [84, 46, 0, 0]):
    ansatz = create_ansatz(xm, xp, T_array = T_array)
    create_fields(ansatz, [xm, xp, N])

#Plot functions
def PlotMetric():
    plt.plot(x_list, f.v, color = "red", label = f"$f$ value")
    plt.plot(x_list, [f_ansatz(x) for x in x_list], color = "orange", label = f"$f$ Ansatz")
    plt.plot(x_list, b.v, color = "blue", label = f"$b$ value")
    plt.plot(x_list, [b_ansatz(x) for x in x_list], color = "purple", label = f"$b$ Ansatz")

    plt.legend(loc = "upper right")
    plt.title(f"Ansatz vs Minimum Value metric functions $f$ and $h$");
    plt.xlabel("Rescaled Position $x$");

    plt.show()

def plot_all_fields():
    # Create individual plots and capture them in Output widgets
    metric = widgets.Output()
    with metric:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(f.x_list, f.v, color = "red", label = f"$f$ value")
        ax1.plot(f.x_list, [f.ansatz(x) for x in f.x_list], color = "orange", label = f"$f$ Seed")
        ax1.plot(f.x_list, [100 * i for i in b.v], color = "blue", label = f"$b$ value")
        ax1.plot(f.x_list, [100*b.ansatz(x) for x in b.x_list], color = "purple", label = f"$h$ Seed")

        ax1.legend(loc = "upper right", fontsize="small")
        ax1.set_title(f"Ansatz vs. Solution: $f(x)$ & $h(x)$");
        ax1.set_xlabel("Rescaled Position $x$");
        
        display(fig1)
        plt.close(fig1) # Close the figure to prevent immediate display
    
    conformalon = widgets.Output()
    with conformalon:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(ph.x_list, ph.v, color = "red", label = f"$\\phi$ value")
        ax2.plot(ph.x_list, [ph.ansatz(x) for x in ph.x_list], color = "orange", label = f"$\\phi$ Ansatz")
        ax2.plot(psi.x_list, psi.v, color = "blue", label = f"$\\psi$ value")
        ax2.plot(psi.x_list, [psi.ansatz(x) for x in psi.x_list], color = "purple", label = f"$\\psi$ Ansatz")
        
        ax2.legend(loc = "upper right", fontsize="small")
        ax2.set_title(f"Ansatz vs. Solution for $\\phi(x)$ & $\\psi(x)$");
        ax2.set_xlabel("Rescaled Position $x$");

        display(fig2)
        plt.close(fig2) # Close the figure to prevent immediate display
    
    fourform = widgets.Output()
    with fourform:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(w.x_list, w.v, color = "red", label = f"$w$ value")
        ax3.plot(w.x_list, [w.ansatz(x) for x in w.x_list], color = "orange", label = f"$w$ Ansatz")
        
        #plt.legend(loc = "upper right")
        ax3.set_title(f"Ansatz vs. Solution for $w(x)$");
        ax3.set_xlabel("Rescaled Position $x$");
        
        ax3.legend(loc = "lower right", fontsize="small")
        display(fig3)
        plt.close(fig3)
    
    # Display them horizontally using HBox
    box = widgets.HBox([metric, conformalon, fourform], layout=widgets.Layout(justify_content="space-around"))
    
    display(box)

# Create Lagrangian

def dAlembert(field, i):
    return b.s[i]*(b.sd[i]*field.sd[i]*f.s[i] + b.s[i]*f.sd[i]*field.sd[i] + b.s[i]*f.s[i]*field.sdd[i])

def R_func(i):
    return -1*b.s[i]*(b.sd[i]*f.sd[i] + b.s[i]*f.sdd[i])

def LEH(i):
    return_val = 1/b.s[i] - 2*f.s[i]*b.sd[i] - b.s[i]*f.sd[i]
    return return_val

def L2(i):
    return_val = smp.Rational(3, 4) * b.s[i] * w.sd[i]**2   
    return return_val

def LInt(i):
    return_val = -gamma_s * w.s[i] * ( aF.sd[i]*(ph.s[i] + psi.s[i]) + aF.s[i]*(ph.sd[i] + psi.sd[i]) )
    return return_val

def L4(i):
    dAphi = dAlembert(ph, i)
    Ricci = R_func(i)
    body = dAphi**2 - ((f.s[i]*b.s[i]**2*Ricci)/3)*(ph.sd[i])**2 + smp.Rational(2,3)*Ricci*dAphi
    return_val = alpha_s*body/b.s[i]
    return return_val

def L4a(i):
    dAphi = dAlembert(ph, i)
    Ricci = R_func(i)
    body = dAphi**2
    return_val = alpha_s*body/b.s[i]
    return return_val

def L4b(i):
    dAphi = dAlembert(ph, i)
    Ricci = R_func(i)
    body = - ((f.s[i]*b.s[i]**2*Ricci)/3)*(ph.sd[i])**2
    return_val = alpha_s*body/b.s[i]
    return return_val

def L4c(i):
    dAphi = dAlembert(ph, i)
    Ricci = R_func(i)
    body = smp.Rational(2,3)*Ricci*dAphi
    return_val = alpha_s*body/b.s[i]
    return return_val

def L5(i):
    Ricci = R_func(i)
    return (beta_s/b.s[i])*Ricci**2*ph.s[i]

def L6(i):
    dApsi = dAlembert(psi, i)
    Ricci = R_func(i)
    body = dApsi**2 - (f.s[i]*b.s[i]**2*Ricci/3)*(psi.sd[i])**2
    return_val = -1*alpha_s*body/b.s[i]
    return return_val
    
def Ltot(i):
    return LEH(i) + L2(i) + LInt(i) + L4(i) + L5(i) + L6(i)

def SPart(L):
    return dx*sum([L(i).xreplace(y.args) for i in range(N)][1:-1])

def initialize_fds(xm, xp, N, alpha, beta, gamma, epsilon, T_array = 30):
    define_peculiar_variables(xm, xp, alpha, beta, gamma, epsilon)
    fields_from_scratch(xm, xp, N, T_array = T_array)