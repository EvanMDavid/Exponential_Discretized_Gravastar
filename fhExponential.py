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

def create_ansatz(xm, xp, transition = 0):
    global nu_ansatz, l_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz

    def nu_ansatz(x, T = transition):
        if abs(x) < T:
            return ln(CubicHermiteSpline([-T, T], [.5 * T, T], [-.5, 1])(x))
        if x == 0:
            return ln(.01)
        if x < 0:
            return ln(-x/2)
        else:
            return ln(x)
  
    def l_ansatz(x, T = transition):
        if abs(x) < T:
            return ln(CubicHermiteSpline([-T, T], [2 * T, T], [-2, 1])(x))
        if x == 0:
            return ln(.01)
        if x < 0:
            return ln(-2*x)
        else:
            return ln(x)

    def phi_ansatz(x, T = transition):
        if x == 0:
            return -100
        elif x < 0:
            return c0 + cH * ln(-epsilon * x / 2)
        elif x > 0:
            return cinf + cS * ln(epsilon * x)
    def psi_ansatz(x, T = transition):
        return 0
    def w_ansatz(x, T = transition):
        if abs(x) < -1:
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
    
    return [nu_ansatz, l_ansatz, phi_ansatz, psi_ansatz, w_ansatz, aF_ansatz]

def create_fields(ansatz, field_domain):
    xm = field_domain[0]
    xp = field_domain[1]

    num = ln(abs(xm)/2)
    nup = ln(xp)
    lm = ln(2*abs(xm))
    lp = ln(xp)
    
    global nu, l, ph, psi, w, aF, y
    nu = field("nu", ansatz[0], domain = field_domain, fixes = [num, nup, 1/xm, 1/xp], positive = False, dynamic = True, bounds = [-20, 20])
    l = field("lambda", ansatz[1], domain = field_domain, fixes = [lm, lp, 1/xm, 1/xp], positive = False, dynamic = True, bounds = [-20, 20])
    ph = field("varphi", ansatz[2], domain = field_domain, fixes = [phxm, phxp, dphixm, dphixp], bounds = [-1000, 1000])
    psi = field("psi", ansatz[3], domain = field_domain, fixes = [psixm, psixp, dpsixm, dpsixp], bounds = [-1000, 1000])
    w = field("w", ansatz[4], domain = field_domain, fixes = ["free", 0, 1, 0])
    aF = field("aF", ansatz[5], domain = field_domain, dynamic = False)
    
    y = full_field([nu, l, ph, psi, w, aF])

def fields_from_scratch(xm, xp, N, transition = 0):
    ansatz = create_ansatz(xm, xp, transition = transition)
    create_fields(ansatz, [xm, xp, N])

#Plot functions
def PlotMetric():
    plt.plot(x_list, f.v, color = "red", label = f"$f$ value")
    plt.plot(x_list, [nu_ansatz(x) for x in x_list], color = "orange", label = f"$f$ Ansatz")
    plt.plot(x_list, b.v, color = "blue", label = f"$b$ value")
    plt.plot(x_list, [l_ansatz(x) for x in x_list], color = "purple", label = f"$b$ Ansatz")

    plt.legend(loc = "upper right")
    plt.title(f"Ansatz vs Minimum Value metric functions $f$ and $h$");
    plt.xlabel("Rescaled Position $x$");

    plt.show()

def plot_all_fields():
    # Create individual plots and capture them in Output widgets
    metric = widgets.Output()
    with metric:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(nu.x_list, nu.v, color = "red", label = f"$ln(f)$ value")
        ax1.plot(nu.x_list, [nu.ansatz(x) for x in nu.x_list], color = "orange", label = f"$ln(f)$ Seed")
        ax1.plot(nu.x_list, l.v, color = "blue", label = f"$ln(h)$ value")
        ax1.plot(nu.x_list, [l.ansatz(x) for x in nu.x_list], color = "purple", label = f"$ln(h)$ Seed")

        ax1.legend(loc = "upper right", fontsize="small")
        ax1.set_title(f"Seed vs. Solution: $f(x)$ & $h(x)$");
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

def foh(i):
    return smp.exp(smp.Rational(1,2)*(nu.s[i] - l.s[i]))

def hof(i):
    return smp.exp(smp.Rational(1,2)*(l.s[i] - nu.s[i]))

def dAlembert(field, i):
    return smp.exp(l.s[i])*(field.sdd[i] + smp.Rational(1,2)*l.sd[i]*field.sd[i] + smp.Rational(1,2)*nu.sd[i]*field.sd[i])

def LEH(i):
    return_val = foh(i)*(1 - l.sd[i]*smp.exp(l.s[i]))
    return return_val

def L2(i):
    return_val = smp.Rational(3, 4) * hof(i) * w.sd[i]**2   
    return return_val

def LInt(i):
    return_val = - gamma_s * w.s[i] * (aF.sd[i]*(ph.s[i] + psi.s[i]) + aF.s[i]*(ph.sd[i] + psi.sd[i]))
    return return_val

def L4(i):
    dAphi = dAlembert(ph, i)
    Ricci = -1*dAlembert(nu, i)
    body = dAphi**2 - smp.Rational(1,3)*smp.exp(l.s[i])*Ricci*ph.sd[i]**2 + smp.Rational(2,3)*Ricci*dAphi
    return_val = alpha_s*foh(i)*body
    return return_val

def L5(i):
    Ricci = -1*dAlembert(nu, i)
    return beta_s*foh(i)*Ricci**2*ph.s[i]

def L6(i):
    dApsi = dAlembert(psi, i)
    Ricci = -1*dAlembert(nu, i)
    body = dApsi**2 - smp.Rational(1,3)*smp.exp(l.s[i])*Ricci*psi.sd[i]**2
    return_val = -1*alpha_s*foh(i)*body
    return return_val
    
def Ltot(i):
    return LEH(i) + L2(i) + LInt(i) + L4(i) + L5(i) + L6(i)

def SPart(L, N):
    return (sum([(Dx*L(i)).xreplace(y.args) for i in range(N)][1:-1]))

def initialize_fds(xm, xp, N, alpha, beta, gamma, epsilon, T = 0):
    define_peculiar_variables(xm, xp, alpha, beta, gamma, epsilon)
    fields_from_scratch(xm, xp, N, transition = T)