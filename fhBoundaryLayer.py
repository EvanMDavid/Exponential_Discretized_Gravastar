from GravastarClass import *

import numpy as np
from scipy.interpolate import CubicHermiteSpline
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
    make_parameters_global(alpha, beta, gamma, epsilon)

    #Define Boundary Condition Parameters
    global c0, cinf, cH, cF, cS
    c0 = 1
    cH = 1
    cS = 1
    cinf = 3/(gamma * .169542) + c0 + (cH - cS)*ln(epsilon * xp)
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

    #Define Ansatz Parameters
    global p, q, eta, lamb
    eta = 6.5
    lamb = .04
    p = 0
    q = 1

#xp and xm for purposes of calculating the ansatz

def Theta(sgn, x):
    return (1/2)*(1 + sgn*np.tanh(x))

def create_ansatz(xm, xp, transition = 0):
    global f_ansatz, h_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz

    def f_ansatz(x, T = transition):
        if abs(x) < T:
            return CubicHermiteSpline([-T, T], [.5 * T, T], [-.5, 1])(x)
        if x == 0:
            return .01
        if x < 0:
            return -x/2
        else:
            return x
        #return (1/2)*np.sqrt(x**2 + eta**2)*Theta(-1,lamb*x) + np.sqrt(x**2 + eta**2)*Theta(1,lamb*x)    
    def h_ansatz(x, T = transition):
        if abs(x) < T:
            return CubicHermiteSpline([-T, T], [2 * T, T], [-2, 1])(x)
        if x == 0:
            return .01
        if x < 0:
            return -2*x
        else:
            return x
        #return 2*np.sqrt(x**2 + eta**2)*Theta(-1,lamb*x) + np.sqrt(x**2 + eta**2)*Theta(1,lamb*x)    
    def old_phi_ansatz(x):
        left =  cH*(x/xm - 1) - (p*x + q)*ln((x**2 + eta**2)/(xm**2 + eta**2)) + 2*p*(x - xm) + 2*q*(x/xm - 1) + phxm
        right = -1*(p*x + q)*ln((x**2 + eta**2)/(xp**2 + eta**2)) + 2*p*(x - xp) + 2*q*(x/xp - 1) + phxp    
        return left*Theta(-1, x*lamb) + right*Theta(1, x*lamb)  
    def phi_ansatz(x):
        if x == 0:
            return -100
        elif x < 0:
            return c0 + cH * ln(-epsilon * x / 2)
        elif x > 0:
            return cinf + cS * ln(epsilon * xp)
    def psi_ansatz(x):
        left =  .5*(p*x + q)*ln((x**2 + eta**2)/(xm**2 + eta**2)) - p*(x - xm) - q*(x/xm - 1) + psixm
        right = .5*(p*x + q)*ln((x**2 + eta**2)/(xp**2 + eta**2)) - p*(x - xp) - q*(x/xp - 1) + psixp
        return 0 #left*Theta(-1, x*lamb) + right*Theta(1, x*lamb)    
    def w_ansatz(x):
        if x < 0:
            return x
        else:
            return 0
    def aF_ansatz(x):
        if x == 0:
            return 1
        z = np.sqrt(1 + abs(x/xp))
        return (3*z**2 - 1)/2 - 3/8 * z**2 * (z**2 - 1) * (ln((z + 1)/(z - 1)))**2
    
    return [f_ansatz, h_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz]

def create_fields(ansatz, field_domain):
    xm = field_domain[0]
    xp = field_domain[1]
    global f, h, ph, psi, w, aF, y
    f = field("f", ansatz[0], domain = field_domain, fixes = [abs(xm)/2, xp, -1/2, 1], positive = True, dynamic = True)
    h = field("h", ansatz[1], domain = field_domain, fixes = [2*abs(xm), xp, -2, 1], positive = True, dynamic = True)
    ph = field("varphi", ansatz[2], domain = field_domain, fixes = [phxm, phxp, dphixm, dphixp])
    psi = field("psi", ansatz[3], domain = field_domain, fixes = [psixm, psixp, dpsixm, dpsixp])
    w = field("w", ansatz[4], domain = field_domain, fixes = ["free", 0, 1, 0])
    aF = field("aF", ansatz[5], domain = field_domain, dynamic = False)
    
    y = full_field([f, h, ph, psi, w, aF])

def fields_from_scratch(xm, xp, N, transition = 0):
    ansatz = create_ansatz(xm, xp, transition = transition)
    create_fields(ansatz, [xm, xp, N])

#Plot functions
def PlotMetric():
    plt.plot(x_list, f.v, color = "red", label = f"$f$ value")
    plt.plot(x_list, [f_ansatz(x) for x in x_list], color = "orange", label = f"$f$ Ansatz")
    plt.plot(x_list, h.v, color = "blue", label = f"$h$ value")
    plt.plot(x_list, [h_ansatz(x) for x in x_list], color = "purple", label = f"$h$ Ansatz")

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
        ax1.plot(f.x_list, h.v, color = "blue", label = f"$h$ value")
        ax1.plot(f.x_list, [h.ansatz(x) for x in h.x_list], color = "purple", label = f"$h$ Seed")

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
    return h.s[i]*field.sdd[i] + smp.Rational(1,2)*(h.sd[i] + (h.s[i]/f.s[i])*f.sd[i])*field.sd[i]

def R_func(i):
    return -1*smp.Rational(1,2)*(f.sd[i]*h.sd[i]/f.s[i]) + smp.Rational(1,2)*(h.s[i]/f.s[i]**2)*(f.sd[i])**2 - (h.s[i]/f.s[i])*(f.sdd[i])

def LEH(i):
    return_val = smp.sqrt(f.s[i]/h.s[i])*(1 - h.sd[i])
    return return_val

def L2(i):
    #return_val = smp.Rational(3,4)*smp.sqrt(h.s[i]/f.s[i])*(w.sd[i])**2
    root_hf = smp.sqrt(h.s[i] / f.s[i])
    return_val = smp.Rational(3, 4) * root_hf * w.sd[i]**2   
    return return_val

def LInt(i):
    return_val = -gamma_s * w.s[i] * (aF.sd[i]*(ph.s[i] + psi.s[i]) + aF.s[i]*(ph.sd[i] + psi.sd[i]))
    return return_val

def L4(i):
    dAphi = dAlembert(ph, i)
    Ricci = R_func(i)
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    body = dAphi**2 - ((h.s[i]*Ricci)/3)*(ph.sd[i])**2 + smp.Rational(2,3)*Ricci*dAphi
    return_val = alpha_s*rootmg*body
    return return_val

def L5(i):
    Ricci = R_func(i)
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    return (beta_s*rootmg*Ricci**2*ph.s[i])

def L6(i):
    dApsi = dAlembert(psi, i)
    Ricci = R_func(i)
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    body = dApsi**2 - (h.s[i]*Ricci/3)*(psi.sd[i])**2
    return_val = -1*alpha_s*rootmg*body
    return return_val

def LFake(i):
    return (h.sdd[i])**2

def L4print(i):
    dAphs = smp.Function(fr"\square\\varphi_{i}")(f.s[i], f.s[i+1], h.s[i], h.s[i+1], ph.s[i-1], ph.s[i], ph.s[i+1])
    Rs = smp.Function(f"R{i}")(f.s[i-1], f.s[i], f.s[i+1], h.s[i], h.s[i+1])
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    body = dAphs**2 - (h.s[i]*Rs/3)*(ph.sd[i])**2 + (2/3)*Rs*dAphs
    return alpha*rootmg*body

def L5print(i):
    Rs = smp.Function(f"R{i}")(f.s[i-1], f.s[i], f.s[i+1], h.s[i], h.s[i+1])
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    return (beta*rootmg*Rs**2*ph.s[i])

def L6print(i):
    dApsf = smp.Function(fr"\square\psi_{i}")(f.s[i], f.s[i+1], h.s[i], h.s[i+1], psi.s[i-1], psi.s[i], psi.s[i+1])
    Rf = smp.Function(f"R{i}")(f.s[i-1], f.s[i], f.s[i+1], h.s[i], h.s[i+1])
    rootmg = smp.sqrt(f.s[i]/h.s[i])
    body = dApsf**2 - (h.s[i]*Rf/3)*(psi.sd[i])**2
    return -2*alpha*rootmg*body
    
def Ltot(i):
    return LEH(i) + L2(i) + LInt(i) + L4(i) + L5(i) + L6(i)

def SPart(L, N):
    return sum([(Dx*L(i)).xreplace(y.args) for i in range(N)][1:-1])

def initialize_fds(xm, xp, N, alpha, beta, gamma, epsilon, T = 0):
    define_peculiar_variables(xm, xp, alpha, beta, gamma, epsilon)
    fields_from_scratch(xm, xp, N, transition = T)