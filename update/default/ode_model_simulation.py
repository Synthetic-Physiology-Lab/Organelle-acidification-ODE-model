##########################################################################################################
#                                                                                                        #
# This code "ODE model simulation" is written by Ahmed Khalil for "a model of lysosomal acidification".  #
# The code is created based on the original ODE codes of Berkeley Madonna and Francesco Pasqualini.      # 
#                                                                                                        #
##########################################################################################################

import sys
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pumpflux import pumpvalues



class ODEMODEL:    
    def __init__(self,STOPTIME=True, DT=True, DTMAX=True, TOLERANCE=True, Pump_flux=True, B=True, Q=True):
        super(ODEMODEL, self).__init__()
        
        # ODE input file        
        inputfile = sys.argv[1]        
        def get_params(inputfile):
            param = []
            val = []
            #file = open(inputfile, "r")
            with open('input.txt', 'r') as f:
                for line in f:	
                    if not line.startswith(("#")):
                        sep = line.strip().split()
                        param.append(str(sep[0]))
                        val.append(float(sep[1]))
            self.ode_dict = {k:v for k,v in zip(param, val)}
            return self.ode_dict
        
        # call ode dictionary (keys and values)
        self.ode_dict = get_params(inputfile)
        # print to check the input file contents
        print("inputfile:", inputfile, self.ode_dict, "\n")
        
        
        # Standard Constants:
        self.mole=self.ode_dict["mole"]         # Avogadro's Number 
        self.RTF=self.ode_dict["RTF"]           # RT/F [mV]
        self.F=self.ode_dict["F"]               # Faraday's Constant
        self.cap_0=self.ode_dict["cap_0"]       # capacitance per unit area [Farad/cm^2]


        # METHOD STIFF
        self.STARTTIME=self.ode_dict["STARTTIME"]
        self.STOPTIME=self.ode_dict["STOPTIME"]
        self.DT=self.ode_dict["DT"]
        self.DTMAX=self.ode_dict["DTMAX"]
        self.TOLERANCE=self.ode_dict["TOLERANCE"]
        
        # Donnan Particles [M]
        self.B=B  
        # enforce initial osmotic balance                                         
        self.Q=Q            
         
        # parameter list ()
        # vesicle shape variables (assumes spherical compartment)
        self.R=self.ode_dict["R"]               # organelle radius [microns]
        # initial volume [liters]
        self.initV=(4/3*np.pi*self.R**3*1e-12)/1000. 
        # surface area [cm^2]
        self.SA=4*np.pi*self.R**2*1e-8
        # total capacitance [Farad]
        self.cap=((self.cap_0*self.SA)/1000.)  
        
        # surface potentials parameters in mV:        
        self.psi_out=self.ode_dict["psi_out"]   # outside leaflet [mV]
        self.psi_in=self.ode_dict["psi_in"]     # inside leaflet [mV]
        
        # cytosolic concentrations parameters
        self.pHbulk=self.ode_dict["pHbulk"]     # for pH [pH units]
        self.Kbulk=self.ode_dict["Kbulk"]       # for K+ [M]
        self.Clbulk=self.ode_dict["Clbulk"]     # for Cl- [M]
        self.Nabulk=self.ode_dict["Nabulk"]     # for Na+ [M]

        # luminal concentrations parameters
        self.initpH=self.ode_dict["initpH"]     # pH [pH units]
        self.initk= self.ode_dict["initk"]      # K+ [M]
        self.initcl=self.ode_dict["initcl"]     # Cl- [M]
        self.initna=self.ode_dict["initna"]     # Na+ [M]
        self.initH=self.ode_dict["initH"]       # total H+ [M]

        # set initial total membrane potential [mV]
        self.initpsi_total=self.ode_dict["initpsi_total"]

        # kinetic parameters
        self.P=self.ode_dict["P"]               # H+ permeability
        self.Pcl=self.ode_dict["Pcl"]           # Cl+ permeability
        self.Pk=self.ode_dict["Pk"]             # K+ permeability
        self.Pna=self.ode_dict["Pna"]           # Na+ permeability
        self.Pw=self.ode_dict["Pw"]             # Water permeability
        self.N_VATP=self.ode_dict["N_VATP"]     # Number of V-ATPases
        self.N_CLC=self.ode_dict["N_CLC"]       # Number of ClC-7 antiporters
        self.CLC_Cl=self.ode_dict["CLC_Cl"]     # ClC-7 Cl- Stoichiometry
        self.CLC_H=self.ode_dict["CLC_H"]       # ClC-7 H+ Stoichiometry
        self.Oc=self.ode_dict["Oc"]             # Cytoplasmic Osmolarity [M]
        self.oh=self.ode_dict["oh"]             # H+ Osmotic coefficient
        self.ok=self.ode_dict["ok"]             # K+ Osmotic coefficient
        self.ona=self.ode_dict["ona"]           # Na+ Osmotic coefficient
        self.ocl=self.ode_dict["ocl"]           # Cl- Osmotic coefficient    

        # buffering capacity parameters in [mM/pH unit]
        self.beta=self.ode_dict["beta"]
        # Flux values
        self.Pump_flux=Pump_flux


    # aux. fun
    @property
    def GetDIMsValues(self):

        grid_x = np.linspace(-300,300,31)   
        grid_y = np.linspace(0,9,46)        
        # get original flux values
        values = pumpvalues
        return grid_x, grid_y, values

    # aux. fun
    def GetMeshGrid(self, grid_x, grid_y):

        Gx, Gy = np.meshgrid(grid_x, grid_y)
        X_Interp = Gx.flatten()
        Y_Interp = Gy.flatten()
        return X_Interp, Y_Interp

    # aux. fun
    def ManageValues(self, values):

        # do slicing of sublists over values
        vsublists = [values[i:i+46] for i in range(0,len(values),46)]
        # convert it to numpy array
        vsublists = np.array(vsublists)
        # do transposing with reshaping to get the final values
        fvalues = vsublists.T.flatten()
        return fvalues
    
    @property
    def Interpolate(self):

        grid_x, grid_y, values = self.GetDIMsValues
        X_Interp, Y_Interp = self.GetMeshGrid(grid_x, grid_y)
        fvalues = self.ManageValues(values)
        self.Pump_flux = interpolate.interp2d(X_Interp, Y_Interp, fvalues, kind ='linear')
        return self.Pump_flux

    ##==========================================================================================================================
    ##==========================================================================================================================
    
    @property
    def InitConditions(self):

        pH  = self.initpH
        Ncl = self.initcl*self.initV*self.mole
        NK  = self.initk*self.initV*self.mole
        Nna = self.initna*self.initV*self.mole
        NH  = self.initH*self.initV*self.mole
        V   = self.initV
        return pH, Ncl, NK, Nna, NH, V


    def LuminalConcs(self, Ncl, V, NK, NH, Nna):
        #pH, Ncl, NK, Nna, NH, V
        Cl = Ncl/V/self.mole
        K = NK/V/self.mole
        H = NH/V/self.mole
        Na = Nna/V/self.mole

        return Cl, K, H, Na

 
    @property
    def GetMCytoSurfConcs(self):

        #Modified Cytoplasmic Surface Concentrations
        Cle = self.Clbulk*np.exp(self.psi_out/self.RTF)
        Ke  = self.Kbulk*np.exp(-self.psi_out/self.RTF)
        Nae = self.Nabulk*np.exp(-self.psi_out/self.RTF)
        pHe = (self.pHbulk+self.psi_out/(self.RTF*2.3))
        return Cle, Ke, Nae, pHe


    def GetMLuminalSurfConcs(self, Cl, K, Na, pH):
    
        #Modified Luminal Surface Concentrations
        Cli = Cl*np.exp((self.psi_in/self.RTF))
        Ki  = K*np.exp((-self.psi_in/self.RTF))
        Nai = Na*np.exp((-self.psi_in/self.RTF))
        pHi = (pH+self.psi_in/(self.RTF*2.3))
        return Cli, Ki, Nai, pHi


    @property
    def GetDonnanParticles(self):

        """
        it returns Donnan Particles [M] 
        """
        self.B = self.initk+self.initna+self.initH-self.initcl - self.cap/self.F/self.initV*(self.psi_in - self.psi_out) - self.initpsi_total*self.cap/self.F/self.initV
        return self.B


    @property
    def SetOsmoticBalance(self):
        
        """
        enforce initial osmotic balance
        """
        self.Q = self.initV*(self.Oc - (self.oh*10**(-self.initpH) + self.ok*self.initk + self.ona*self.initna + self.ocl*self.initcl))
        return self.Q


    def GetHpump(self,psi, pH):

        Hpump = self.N_VATP*self.Pump_flux(psi,pH)
        return Hpump


    def Getgg(self, psi):

        """
        it returns gg that allows treatment of singular terms for passive ion flux
        """  
    	
        if abs(psi) > .01:
            gg = psi/(1-np.exp(-psi/self.RTF))/self.RTF
            return gg
        else:
            gg = 1/(1 - (psi/self.RTF)/2 + (psi/self.RTF)**2/6. - (psi/self.RTF)**3/24. +(psi/self.RTF)**4/120.)
            return gg



    def GetggL(self, psi):

        """
        it returns gg that allows treatment of singular terms for passive ion flux
        """ 

        for p in psi: 
    	
            if abs(p) > .01:
                gg = p/(1-np.exp(-p/self.RTF))/self.RTF
                return gg
            else:
                gg = 1/(1 - (p/self.RTF)/2 + (p/self.RTF)**2/6. - (p/self.RTF)**3/24. +(p/self.RTF)**4/120.)
                return gg


 
    def GetCLC7(self, Cle, Cli, pHe, pHi, psi): 

        clc7f = (self.CLC_Cl+self.CLC_H)*psi + self.RTF*(2.3*(pHe - pHi) + self.CLC_Cl*np.log(Cle/Cli))
        #clc7f = (self.CLC_Cl+self.CLC_H)*psi + self.RTF*(2.3*(pHe - pHi) + self.CLC_Cl*np.log(max(Cle/Cli, 1e-200)))
    
        G1 = -0.3*clc7f
        G3 = -1.5e-5*clc7f**3
        S = 0.5+0.5*np.tanh((clc7f+250)/75.)
        CLC7_rate = S*G1+(1-S)*G3
        CLC7 = self.N_CLC*CLC7_rate
        return CLC7 


    # passive flow functions:
    # 1.  
    def GetHflow(self, pHe, pHi, pH, psi, gg): 

        Hflow = self.P*self.SA*(10**(-pHe)*np.exp(-psi/self.RTF)-10**(-pHi))*gg*self.mole/1000.
        return Hflow

    # 2.
    def GetClflow(self, Cle, Cli, psi, gg):


        Clflow = self.Pcl*self.SA*( Cle-Cli*np.exp(-psi/self.RTF))*gg*self.mole/1000.
        return Clflow

    # 3.
    def GetKflow(self, Ke, psi, Ki, gg): ### i am here

        Kflow  = self.Pk*self.SA*(Ke*np.exp(-psi/self.RTF)-Ki)*gg*self.mole/1000.
        return Kflow

    # 4.
    def GetNaflow(self, Nae, psi, Nai, gg):

        Naflow = self.Pna*self.SA*(Nae*np.exp(-psi/self.RTF)-Nai)*gg*self.mole/1000. 
        return Naflow

    # 5.
    def GetJW(self, pH, K, Na, Cl, V):
    
        Jw = self.Pw*self.SA*(self.oh*10**(-pH) + self.ok*K + self.ona*Na + self.ocl*Cl + self.Q/V - self.Oc)
        return Jw



    # Time Dependent Quantitie (TDQ) functions:
    # 1.
    def Get_dpH_dt(self, Hpump, Hflow, CLC7, V):
        dpH_dt = ((-Hpump - Hflow -(self.CLC_H)*CLC7)/V/self.mole)/self.beta
        return dpH_dt

    # 2.
    def Get_dNcl_dt(self, Clflow, CLC7):

        dNcl_dt = Clflow - self.CLC_Cl*CLC7
        return dNcl_dt

    # 3.
    def Get_dNK_dt(self, Kflow):
        dNK_dt = Kflow
        return dNK_dt

    # 4.
    def Get_dNna_dt(self, Naflow):

        dNna_dt = Naflow
        return dNna_dt


    # 5.
    def Get_dNH_dt(self,Hpump, Hflow, CLC7):

        dNH_dt = Hpump + Hflow + (self.CLC_H)*CLC7
        return dNH_dt   

    # 6.
    def Get_dV_dt(self, Jw):
        dV_dt = Jw/(1000*55.)
        return dV_dt
    

    def GetTDP(self, list):

        """
        it returns time dependent parameters (TDP)
        """

        pH, Ncl=list[0],list[1]
        NK, Nna=list[2],list[3]
        NH, V=list[4],list[5]
        return pH,Ncl,NK,Nna,NH,V
        

    def Getpsi(self, V, H, K, Na, Cl):
    
        psi = self.F*(V*(H+(K+Na)-Cl)-(self.B*self.initV))/self.cap
        return psi 
        
        
    def GetSolverConcs(self, MAT=np.array([])):
        
        """
        it returns concentrations matrices from the solver
        """  
        pH = MAT[0, :]      
        Ncl = MAT[1, :]
        NK = MAT[2, :]
        Nna = MAT[3, :]
        NH = MAT[4, :]
        V = MAT[5, :] 
        
        return pH, Ncl, NK, Nna, NH, V
        
       

    
    def TDQ(self, t, y):

    	# get time dependent parameters
        pH, Ncl, NK, Nna, NH, V = self.GetTDP(y)
        ##print("main parameters", pH, Ncl, NK, Nna, NH, V)
                     
        Cl, K, H, Na = self.LuminalConcs(Ncl, V, NK, NH, Nna)
        ## Modified Cytoplasmic Surface Concentrations
        Cle, Ke, Nae, pHe = self.GetMCytoSurfConcs 

        ###print("Cle, Ke, Nae, pHe", Cle, Ke, Nae, pHe)       
        ## Modified Luminal Surface Concentrations
        Cli, Ki, Nai, pHi = self.GetMLuminalSurfConcs(Cl, K, Na, pH) ###  

        ## get psi
        psi = self.Getpsi(V, H, K, Na, Cl)
        # get Hpump
        Hpump = self.GetHpump(psi, pH) 
        # get CLC7      
        CLC7 = self.GetCLC7(Cle, Cli, pHe, pHi, psi)
        # treatment of singular terms for passive ion flux
        gg = self.Getgg(psi)
        # get Hflow        
        Hflow = self.GetHflow(pHe, pHi, pH, psi, gg)

        # get Clflow
        Clflow = self.GetClflow(Cle, Cli, psi, gg)
        Kflow = self.GetKflow(Ke, psi, Ki, gg)
        Naflow = self.GetNaflow(Nae, psi, Nai, gg)
        Jw = self.GetJW(pH, K, Na, Cl, V)      
 
        dpH_dt = self.Get_dpH_dt(Hpump, Hflow, CLC7, V) 
        dNcl_dt = self.Get_dNcl_dt(Clflow, CLC7)
        dNK_dt = self.Get_dNK_dt(Kflow)
        dNna_dt = self.Get_dNna_dt(Naflow)
        dNH_dt = self.Get_dNH_dt(Hpump, Hflow, CLC7)
        dV_dt = self.Get_dV_dt(Jw)
         
        return [dpH_dt, dNcl_dt, dNK_dt, dNna_dt, dNH_dt, dV_dt]
        
        
    @property    
    def GetOriginalData(self):
    
        data=np.loadtxt("DATA_SimResults.dat", skiprows=1)
        time=data[:,0]
        pH=data[:,1]
        return time, pH


    def GetPlot(self, time, tdq, color=None, label=None, xlabel=True, ylabel=True, figname=None):
    
        original_time, original_pH = self.GetOriginalData
        plt.plot(original_time, original_pH, linestyle='-', linewidth=2, color="slateblue", label="pH - Berkeley Madonna")
        plt.plot(time, tdq, linestyle='-', linewidth=2, color=color, label=label)
        
        print("length:", len(original_pH), len(tdq))

        plt.title(label="Default case", fontsize=18, color="black")
        plt.grid(linestyle='--',alpha=2)
        plt.legend(loc="best", prop={'size':12}, frameon=False)        
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()
        return 




    def GetPlotF(self, time, flow, color=None, label=None, xlabel=True, ylabel=True, figname=None):
    
        plt.plot(time, flow, linestyle='-', linewidth=2, color=color, label=label)
        
        ##plt.title(label="Default case", fontsize=18, color="black")
        plt.grid(linestyle='--',alpha=2)
        plt.legend(loc="best", prop={'size':12}, frameon=False)        
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()
        return 
 



    def main(self, object):
 
        self.Pump_flux = object.Interpolate
        self.B = self.GetDonnanParticles
        self.Q=self.SetOsmoticBalance
        # set inital conditions
        # [dpH_dt, dNcl_dt, dNK_dt, dNna_dt, dNH_dt, dV_dt]
        pH, Ncl, NK, Nna, NH, V = object.InitConditions  #
        print("initial conditions:", pH, Ncl, NK, Nna, NH, V)

        TIMEINTERVAL = [self.STARTTIME, self.STOPTIME] 
        time = np.arange(self.STARTTIME, self.STOPTIME,self.DT)
        
        # solve ode using solve_ivp
        SOL = solve_ivp(fun=object.TDQ, t_span=TIMEINTERVAL, y0=[pH, Ncl, NK, Nna, NH, V], t_eval=time, method='BDF') 
 
        pH, Ncl, NK, Nna, NH, V = object.GetSolverConcs(SOL.y)
    
        Cl, K, H, Na = object.LuminalConcs(Ncl, V, NK, NH, Nna) 

        Cle, Ke, Nae, pHe = object.GetMCytoSurfConcs
        ##print("TEST", Cle, Ke, Nae, pHe)

        Cli, Ki, Nai, pHi = object.GetMLuminalSurfConcs(Cl, K, Na, pH)
        ##print("Cli, Ki, Nai, pHi", Cli, Ki, Nai, pHi)

        # get psi and psi total
        psi = object.Getpsi(V, H, K, Na, Cl) 
        psi_total = psi + self.psi_out-self.psi_in

        # get CLC7      
        CLC7 = object.GetCLC7(Cle, Cli, pHe, pHi, psi)

        # treatment of singular terms for passive ion flux
        gg = object.GetggL(psi)
        # get flows:
        # 1. H-flow        
        Hflow = object.GetHflow(pHe, pHi, pH, psi, gg)

        # 2. Cl-flow 
        Clflow = object.GetClflow(Cle, Cli, psi, gg)
        # 3. K-flow
        Kflow = object.GetKflow(Ke, psi, Ki, gg)
        # 4. Na-flow
        Naflow = object.GetNaflow(Nae, psi, Nai, gg)
        # 5. Jw-flow
        Jw = object.GetJW(pH, K, Na, Cl, V)      
 

        ## plot-> psi for validation
        origdata=np.loadtxt("DATA_SimResults.dat", skiprows=1)
        origtime=origdata[:,0]
        origpsi=origdata[:,2]


        # get TDQ plots for validation
        object.GetPlot(SOL.t, pH, color="darkorange", label="pH - Python", xlabel="Time [s]", ylabel="pH", figname="validation_default_case")

        
        plt.plot(origtime, origpsi, linestyle='-', linewidth=2, color="blue", label="psi - Berkeley Madonna")
        plt.plot(SOL.t, psi_total, linestyle='-', linewidth=2, color="red", label="psi - Python")        
        plt.title(label="psi - Default case", fontsize=18, color="black")
        plt.grid(linestyle='--',alpha=2)
        plt.legend(loc="best", prop={'size':12}, frameon=False)        
        plt.xlabel("Time [s]", fontsize=15)
        plt.ylabel("psi [mV]", fontsize=15)
        plt.tight_layout()
        plt.savefig("psi-default.png")
        plt.show()



        ## get flow plots
        object.GetPlotF(SOL.t, Hflow, color="darkorange", label="Hflow", xlabel="Time [s]", ylabel="Hflow", figname="Hflow")

        object.GetPlotF(SOL.t, Clflow, color="darkorange", label="Clflow", xlabel="Time [s]", ylabel="Clflow", figname="Clflow")
        object.GetPlotF(SOL.t, Kflow, color="darkorange", label="Kflow", xlabel="Time [s]", ylabel="Kflow", figname="Kflow")

        object.GetPlotF(SOL.t, Naflow, color="darkorange", label="Naflow", xlabel="Time [s]", ylabel="Naflow", figname="Naflow")
       
        


# run ODEMODEL simulation   
if __name__ == "__main__":

    ODEobject = ODEMODEL()
    ODEobject.main(ODEobject) 
