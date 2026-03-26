#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:21:45 2026

@author: isaac
"""

'''
Drop oscillations in to determine which mode each peak is
'''
from astropy.table import Table
import numpy as np
# from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.dpi'] = 600
            

class power_spectrum:
    
    
    def __init__(self,star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma,  star_name ="", sun_facule_tau = 1, sun_facule_sigma = 1):
        """
        

        Parameters
        ----------
        star_mass : float
            Mass of star in solar masses.
        star_radius : float
            Radius of star  in solar radii .
        star_teff : float
            Teff of star in kelvin.
        sun_nu_max : float
            nu max of sun.
        sun_teff : float
            Teff of star in kelvin.
        sun_granulation_tau : float
            Granulation timescale of sun .
        sun_granulation_sigma : float
            Granulation amplitude of sun.
            
        star_name : String, optional
            Name of star. The default is "".
        sun_facule_tau : TYPE, optional
            Facule timescale of sun. The default is 1.
        sun_facule_sigma : TYPE, optional
            Facule amplitude of sun . The default is 1.

        
        """
        
        
        
        #sun constants
        
        self.sun_nu_max = float(sun_nu_max)
        self.sun_teff = float(sun_teff)
        
        self.sun_granulation_tau = float(sun_granulation_tau)
        self.sun_granulation_sigma = float(sun_granulation_sigma)
        
        self.sun_facule_tau = float(sun_facule_tau)
        self.sun_facule_sigma = float(sun_facule_sigma)
        
        self.sun_max_rms_amplitude = 2.1  
        
        
        
        #line space constants
        
        self.freq_powerspectrum_uHz = np.linspace(0.1, 10000.0, 50000) #range of frequencies where power spectrum is evaluated
        self.angular_degree = np.linspace(0,3,4) #List of angular degrees l for oscillations 
        self.radial_mode = np.linspace(0,30,31) #list of radial modes n for oscillations
        
        self.visibilites = [1, 1.505, 0.620, 0.075]
        
        
        
        #star properties 
        
        #change to be input from catalogue when class is called
        self.star_name = star_name
        self.star_mass = float(star_mass)
        self.star_radius =  float(star_radius)
        self.star_teff = float(star_teff)
        
        
        #calculatuting star properties
        
        self.nu_max_ratio = self.calc_nu_max_ratio() #vmax ratio 
        self.star_nu_max = self.nu_max_ratio * self.sun_nu_max #vmax of star 
        
        self.luminosity_ratio = self.calc_luminosity_ratio() #ratio of star luminosity to sun
        
        self.star_tau_ratio = self.calc_tau_ratio() #granulation time scale 
        self.star_granulation_tau = self.star_tau_ratio * self.sun_granulation_tau
        
        self.sigma_ratio = self.calc_sigma_ratio()
        self.star_granulation_sigma = self.sigma_ratio * self.sun_granulation_sigma
        
        #oscillation stuff
        self.sun_delta_nu = 135.1
        self.epsilon= 1.5 #default: 1.55
        
        self.star_delta_nu = self.sun_delta_nu * (self.star_mass**0.5) * (self.star_radius**(-1.5))
        
        self.max_rms_amplitude_radial_ratio = self.calc_maximum_rms_amplitude_radial_ratio()
        self.star_max_rms_amplitude_radial = self.max_rms_amplitude_radial_ratio *  self.sun_max_rms_amplitude
        
        self.star_standev_envelop = self.calc_envelope_width()
        
        
        #line width constants from paper
        # self.alpha = self.calc_width_parameter(2.95, 0.39)
        # self.width_alpha = self.calc_width_parameter(3.08, 3.32)
        # self.delta_width_dip = self.calc_width_parameter(-0.47, 0.62)
        # self.W_dip = self.calc_width_parameter(4637, -141)
        # self.nu_dip = self.calc_width_parameter(2984, 60)
        self.alpha = self.calc_width_parameter(-3.71, 1.073e-3, 1.883e-4)
        self.width_alpha = self.calc_width_parameter(-7.209e1, 1.543e-2, 9.101e-4)
        self.delta_width_dip = self.calc_width_parameter(-2.266e-1, 5.083e-5, 2.715e-6)
        self.W_dip = self.calc_width_parameter(-5.639e-1, 1.138e-4, 1.312e-4)
        self.nu_dip = self.calc_width_parameter(-2.19e3, 4.302e-1, 8.427e-1)
        
        
    
    def calc_nu_max_ratio(self):
        """
        Calculate nu_max ratio using scaling relationship

        
        nu_max / nu_max_sun = (M/Msun) * (R/Rsun)^(-2) * (Teff/Teff_sun)^(-1/2)

        Parameters
        ----------
        mass_star : float
            Stellar mass in solar masses
        r_star : float
            Stellar radius in solar radii
        teff_star : float
            Effective temperature in Kelvin

        Returns
        -------
        nu_max : float
            nu_max in microHz
        """
                
        nu_max_ratio =  (self.star_mass / (self.star_radius**2)) * (self.star_teff / self.sun_teff) ** (-0.5)

        return nu_max_ratio
        
    def calc_luminosity_ratio(self):
        """
        Compute luminosity ratio L/Lsun using 
       
            L/Lsun = (R/Rsun)^2 * (Teff/Teff_sun)^4

        
        Parameters
        ----------
        
        star_radius : float
            Stellar radius in solar radii
        star_teff : float
            Effective temperature in Kelvin

        Returns
        -------
        luminosity_ratio :float
            Luminosity ratio L/Lsun
        """
        
        luminosity_ratio = (self.star_radius**2) * (self.star_teff / self.sun_teff) ** 4
        
        return luminosity_ratio
    
    def calc_tau_ratio(self):
        """
        Calculate timescale of a background component 
            

    

        Parameters
        ----------
        nu_max_star_uHz : float
            nu_max for the star in microHz

        Returns
        -------
        tau_ratio : float
            time scale ratio of background component
        """
        
        tau_ratio =  (self.nu_max_ratio) ** (-1) #changed ratio

        return tau_ratio
        
    def calc_sigma_ratio(self):
        """
        Calculates amplitude ratio using scaling relationship to remove constants from proportionallity constant. Ball et al equation 22 for proportional relationship for granulation
        

        Scaling relationship of granulation amplitude,
            sigma_star/ sigma_sun =
                (L/Lsun)^2 * (M/Msun)^(-3) * (Teff/Teff_sun)^(-5.5) * (nu_max/nu_max_sun)
                
        

       

        Parameters
        ----------
        nu_max_ratio : float
            Ratio fo nu max of star to sun 
        luminosity_ratio : float
            L/Lsun
        star_mass : float
            Stellar mass in solar masses
        star_teff : float
            Effective temperature in Kelvin
        

        Returns
        -------
        sigma_ratio : float
            ratio of background amplitude in star to sun
        """
        
        

        sigma_ratio = (self.luminosity_ratio ** 2) * (self.star_mass ** -3) * (self.star_teff / self.sun_teff) ** (-5.5) * (self.nu_max_ratio) #changed ratio
        
        

        return sigma_ratio   

    
    def calc_nu_nl(self, n,l):
        """
        

        Parameters
        ----------
        n : int
            radial mode
        l : int
            angular degree

        Returns
        -------
        nu_nl : float
            frequency of the p-mode oscillations

        """

        #In the asymptotic limit, dnu03 = 5/3 dnu02, which i have reflected here (Chaplin and Miglio 2013).
        
        # linear fit is from the Kepler LEGACY stars scatter plot, with some random Gaussian noise added in (parameters from star data).
        # Keen et al 2014 gives the second term (the frequency splitting term) to be simply the small frequency separation, hence what i have attempted here,
        # though they add it instead of subtracting it, i tested this and it made everything worse, though that may be due to the random element in fit
        
        
        # fit = (0.05200 * self.star_delta_nu) + np.random.normal(1.42683, 1.41997)
        # fit = (0.05200 * self.star_delta_nu) + 1.42683
        # fit = (0.05200 * self.star_delta_nu)
        fit = 1
        
        if l % 2 == 0:
        
             nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * l * (l+1)
            
        elif l % 2 == 1:
            nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * (5/3) * l * (l+1) 
            # nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * l * (l+1)
        
        return nu_nl, l, fit
    
    def calc_width_nl(self, n, l):
        
        nu_nl, ang_deg, fit = self.calc_nu_nl(n,l)
        
        line_width = ((self.alpha * np.log(nu_nl / self.star_nu_max)) + np.log(self.width_alpha)) + (np.log(self.delta_width_dip)/(1 + (((2 * np.log(nu_nl/self.nu_dip))/(np.log(self.W_dip/self.star_nu_max)))**2)))
        
        width_nl = np.exp(line_width)
        
        return width_nl
    
    def calc_envelope_width(self):
        """
        Calculates the envelope gaussian width using equation 19 and the line below equation 20 from Ball et al
        
        Envelop = 0.66 * (Vmax)^0.88
        
        Parameters input are in units of microHz hence the conversion terms are removed from the Ball et al equations. 
        

        Returns
        -------
        None.

        """
        
        envelope_FWHM = 0.66 * (self.star_nu_max**0.88)
        
        if self.star_teff > self.sun_teff:
            envelope_FWHM = envelope_FWHM * (1+6e-4 * (self.star_teff-self.sun_teff))
            
        width_envelope = envelope_FWHM / (2 * np.sqrt(2*np.log(2)))
 
        return width_envelope

    def calc_powder_density(self):
        
        n_min, n_max =  self.calc_n_min_max()
        
        power_density_oscillations = np.zeros(len(self.freq_powerspectrum_uHz)) #creates base array same size as frequency arraay 
        n_min_max_linspace = np.linspace(min(n_min), max(n_max), (max(n_max)-min(n_min))+1) #lin space of n values  to itterate  over
        
        modes = []
        lorentz_funcs = []
        
        for l in self.angular_degree: #iterates over all possible l angular degrees
            
            for n in n_min_max_linspace: #iterates over all possible n radial degrees 
                
                nu_nl, ang_deg, fit = self.calc_nu_nl(n,l) 
                width_nl = self.calc_width_nl(n, l)
                
                nl_rms_amplitude_radial = self.calc_nl_rms_amplitude_radial(nu_nl)
                
                amplitude_nl = self.visibilites[int(l)] * nl_rms_amplitude_radial
                
                power_amplitude = calc_power_amplitude(amplitude_nl, width_nl)
                lorentz_value = calc_lorentz(self.freq_powerspectrum_uHz, nu_nl, width_nl)
                
                # power_density_oscillations +=(power_amplitude*lorentz_value)
                power_density_oscillations = power_amplitude * lorentz_value
                # x = np.pi * 25 * self.freq_powerspectrum_uHz * 10**(-6)
                # power_density_oscillations *= (np.sin(x)/x)**2
                lorentz_funcs.append(power_density_oscillations)
                modes.append(ang_deg)
                
        return lorentz_funcs, modes, fit
                
                
                
                
                
    def calc_maximum_rms_amplitude_radial_ratio(self):
        
        power = -0.093
        t_red_star = 8900
        delta_t = 1250
        dwarf_suppression_beta = 1 - np.exp((self.star_teff - t_red_star)/(delta_t))

        star_max_rms_amplitude_ratio=  dwarf_suppression_beta * self.luminosity_ratio *(self.star_mass**-1) * ((self.star_teff / self.sun_teff) ** (-2))   
        
        return star_max_rms_amplitude_ratio
        
    def calc_nl_rms_amplitude_radial(self, nu_nl):
        
        
        star_rms_amplitude = (((self.star_max_rms_amplitude_radial ** 2) * np.exp(-((nu_nl-self.star_nu_max)**2)/(2*(self.star_standev_envelop**2))))**0.5)
        
        return star_rms_amplitude
        
        
        
    def calc_n_min_max(self):
        
        n_min = []
        n_max = []
        
        
        
        nu_envelope_min = self.star_nu_max - (3*self.star_standev_envelop)
        nu_envelope_max = self.star_nu_max + (3*self.star_standev_envelop)
        
        for i in self.angular_degree:
            
            n_min.append(int((nu_envelope_min/self.star_delta_nu) - ((i/2) + self.epsilon)))
            n_max.append(int((nu_envelope_max/self.star_delta_nu) - ((i/2) + self.epsilon)))
            
        return n_min, n_max
        
        

        
        
    # def calc_width_parameter(self, a, b):
    #     """
        

    #     Parameters
    #     ----------
    #     a : float
    #         Line width constants .
    #     b : float
    #         Line width constants.

    #     Returns
    #     -------
    #     width_param : float
    #         Line width.

    #     """
        
    #     width_param = a*(self.nu_max_ratio) + b 
        
    #     return width_param
    
    def calc_width_parameter(self, a, b, c):
        nu_max = self.sun_nu_max * ( self.star_mass/ (self.star_radius**2)) * (self.star_teff / self.sun_teff) ** (-0.5)
        P = a + (b*self.star_teff) + (c*nu_max)
        return P
    
        

        
        
def calc_power_amplitude(amp,width):
    
    # x = (2/np.pi)*((amp**2)/width)
    x = amp
    return x 
        
def calc_lorentz(freq, centroid, FWHM):
    
    
    #x = (1/np.pi) * ((FWHM/2) / ((freq - centroid)**2 + (FWHM/2)**2))
    x = ((FWHM/2)**2 / ((freq - centroid)**2 + (FWHM/2)**2))
    return x 


'''
# KIC 6225718 (Saxo2)
star_mass, star_radius, star_teff = 1.10, 1.22, 6313
star_name = "Saxo2"
'''

# KIC 6106415
star_mass,  star_radius, star_teff = 1.039, 1.213, 6037
star_name = "KIC 6106415"


sun_nu_max = 3090
sun_teff = 5772.0
sun_granulation_tau = 214
sun_granulation_sigma = 63

star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)
lorentz_funcs, ang_degs, fit = star.calc_powder_density()

# print(lorentz_funcs)
# print(ang_degs)

for i in range(0, len(lorentz_funcs)):
    if ang_degs[i] == 0.0:
        colour = 'red'
        # label = "l = 0"
        label = ""
        plt.axvline(star.freq_powerspectrum_uHz[np.argmax(lorentz_funcs[i])], ls=':', color=colour)
    elif ang_degs[i] == 1.0:
        colour = 'blue'
        # label = "l = 1"
        label = ""
        plt.axvline(star.freq_powerspectrum_uHz[np.argmax(lorentz_funcs[i])], ls=':', color=colour)
    elif ang_degs[i] == 2.0:
        colour = '#d35fb7' ##7ca1cc
        # label = "l = 2"
        label = ""
        plt.axvline(star.freq_powerspectrum_uHz[np.argmax(lorentz_funcs[i])], ls=':', color=colour)
    elif ang_degs[i] == 3.0:
        colour = 'black'
        # label = "l = 3"
        label = ""
        plt.axvline(star.freq_powerspectrum_uHz[np.argmax(lorentz_funcs[i])], ls=':', color=colour)
    if i % 21 != 0:
        label = ""
    plt.plot(star.freq_powerspectrum_uHz, lorentz_funcs[i], color = colour, label = label)
    
    
# plt.plot(star.freq_powerspectrum_uHz, oscillation_psd, color = "firebrick", label = "p-mode oscillations")
# plt.axvline(star.star_nu_max, ls='--', color='#4b0092', label = "ν_max")
plt.axhline(y = 4, xmin = 0.03, xmax = 0.67,  color = 'b', ls = "-", label = "Δν")
plt.axhline(y = 3, xmin = 0.57, xmax = 0.67,  color = 'black', ls = "-", label = "δν_13") ##cccccc
plt.axhline(y = 2.4, xmin = 0.335, xmax = 0.37,  color = 'r', ls = "-", label = "δν_02")
# plt.xlim(star.star_nu_max-350,star.star_nu_max+150)

xlimit = 80

plt.xlim(star.star_nu_max-xlimit,star.star_nu_max+xlimit)
plt.title(f"{star_name} oscillation mode degrees")
plt.xlabel("Frequency /μHz")
plt.ylabel("Power /ppm^2 μHz^-1")
plt.legend(loc = 'upper right')
plt.show()





