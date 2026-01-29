# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:37:38 2026

@author: mxt216
"""

import numpy as np
# from numpy.polynomial import Polynomial
from astropy.io import fits
import matplotlib.pyplot as plt

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
            Granulation amplitud of sun.
            
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
        
        #line space constants
        
        self.freq_powerspectrum_uHz = np.linspace(0.1, 10000.0, 500000) #range of frequencies where power spectrum is evaluated
        self.angular_degree = np.linspace(0,3,4) #List of angular degrees for oscillations 
        self.radial_mode = np.linspace(0,30,31) #list of radial modes for oscillations
        
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
        self.epsilon= 1.55
        
        
        self.alpha = self.width_parameter(2.95, 0.39)
        self.width_alpha = self.width_parameter(3.08, 3.32)
        self.delta_width_dip = self.width_parameter(-0.47, 0.62)
        self.W_dip = self.width_parameter(4637, -141)
        self.nu_dip = self.width_parameter(2984, 60)
        
        
        
        
        
    def calc_width_parameter(self, a, b):
        """
        

        Parameters
        ----------
        a : float
            Line width constants .
        b : float
            Line width constants.

        Returns
        -------
        width_param : float
            Line width.

        """
        
        width_param = a*(self.nu_max_ratio) + b 
        
        return width_param
                       
                       
        
        
        
        
        
            
            
            

            
        
        
        
        #oscillation stuff
        
    def single_component(self):
        """
        Models the background as one component only due to granulation
        Plot of power spectrum due to granulation  

        """
        
        
        
        
        self.granulation_component = self.calc_component_psd( self.star_granulation_sigma, self.star_granulation_tau)
        self.plot_granulation_psd()
        
        
    def multi_component(self):
        """
        Models the background as two components. One from facules and granuales.
        Plots the total background spectrum 
        
        Assumes that the facule and granulation amplitudes / timescales scale in the same way as granulation.

        

        """
        
        
        self.star_facule_sigma = self.sigma_ratio * self.sun_granulation_sigma
        self.star_facule_tau = self.star_tau_ratio * self.sun_facule_tau
        
        
        self.granulation_component = self.calc_component_psd( self.star_granulation_sigma, self.star_granulation_tau)
        
        self.facule_component = self.calc_component_psd( self.star_facule_sigma, self.star_facule_tau)
        
        self.total_background = self.granulation_component + self.facule_component
        
        self.plot_multi_component_psd()

        
        
        
        
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
    
    
        
    
    
    def calc_granulation_psd_per_uHz(self):
        """
        Old function, calculates only the granulation compoment 
        
        Calculates power spectrum value at an array of given frequencies using  equation 23 from Ball et al
            P(ν) = 4*sigma^2*tau / (1 + (2πνtau)^2)
            
        Returns array of power spectrum values at each frequency
            
        Has to be done in Hz because the term (2πνtau)^2) needs to be unitless, therefore the frequency needs to be s^-1 to cancel tau s


        Parameters
        ----------
        freq_uHz : array
            Frequency input array in microHz 
        tau_gran_s : float
            Granulation timescale in seconds
        sigma_gran : float
            Granulation amplitude of given star

        Returns
        -------
        psd_per_uHz : array
            power spectrum evaluated at microHz
        """
        freq_uHz = np.asarray(self.freq_uHz)
        freq_Hz = freq_uHz * 1e-6

        demominator_power  = 4
        
        psd_per_Hz = (4.0 * (self.sigma_gran ** 2) * self.tau_gran_s) / (1.0 + (2.0 * np.pi * freq_Hz * self.tau_gran_s) ** demominator_power)
        psd_per_uHz = psd_per_Hz * 1e-6 #converts to uHz 
        return psd_per_uHz
    
    def calc_component_psd(self, star_sigma, star_tau):
        """
        Calculates the power spectrum of one compoment of the background model. 
        The component depends on the variables passed in when the function is called
        ie: facule or granulaion
        
        Calculates power spectrum value at an array of given frequencies using  equation 23 from Ball et al
            P(ν) = 4*sigma^2*tau / (1 + (2πνtau)^2)
            
        Returns array of power spectrum values at each frequency
            
        Has to be done in Hz because the term (2πνtau)^2) needs to be unitless, therefore the frequency needs to be s^-1 to cancel tau s

        Parameters
        ----------
        star_sigma : float
            amplitude of background component.
        star_tau : float
            time scale of background compoment.

        Returns
        -------
        psd_per_uHz : array
            power spectrum evaluated at each frequency for specific background component microHz

        """
        
        freq_uHz = np.asarray(self.freq_powerspectrum_uHz)
        freq_Hz = freq_uHz * 1e-6
        
        demominator_power  = 4

        psd_per_Hz = (4.0 * (star_sigma ** 2) * star_tau) / (1.0 + (2.0 * np.pi * freq_Hz * star_tau) ** demominator_power)

        
        psd_per_uHz = psd_per_Hz * 1e-6
        
        return psd_per_uHz
        
    def plot_granulation_psd(self):
        """
        Plot granulation powers spectrum P(nu) at range of frequencies and outputs data

        Parameters
        ----------
        freq_uHz : array
            Frequency array in microHz.
        psd_per_uHz : array
            Granulation powerspectrum in 
        nu_max_star_uHz : float
            nu_max of the star in microHz.
        tau_gran_s : float
            Granulation timescale in seconds.
        mass_star : float
            Stellar mass in solar masses.
        r_star : float
            Stellar radius in solar radii.
        teff_star : float
            Effective temperature in Kelvin.


        """
        plt.figure()
        plt.loglog(self.freq_powerspectrum_uHz, self.granulation_component)
        plt.axvline(self.star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
        
        plt.ylim(bottom=0.1)
        plt.xlim(left=10) 
        
        
        plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
        plt.ylabel(r"Granulation power, ppm^2 / µHz")
        
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
    def plot_multi_component_psd(self):
        """
        Plots power specturm of multi component model 

        Returns
        -------
        None.

        """
        
        plt.figure()
        
        plt.loglog(self.freq_powerspectrum_uHz,self.total_background, label = 'Total ')
        plt.loglog(self.freq_powerspectrum_uHz, self.facule_component, label = 'Facule component')
        plt.loglog(self.freq_powerspectrum_uHz, self.granulation_component, label = 'Granulation component')
        plt.axvline(self.star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
        
        plt.ylim(bottom=0.1)
        plt.xlim(left=10)
        
        plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
        plt.ylabel(r"Granulation power, ppm^2 / µHz")
        
        
        plt.legend()
        plt.tight_layout()
        plt.show()

        
        
        
star_mass, star_radius, star_teff = 1.223, 1.357, 6325
star_name = "Kepler 410"
sun_nu_max = 3090
sun_teff = 5772.0
sun_granulation_tau = 214
sun_granulation_sigma = 23




star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)
star.single_component()


sun_granulation_tau = 214.3
sun_granulation_sigma = 62.4

sun_facule_tau = 65.8
sun_facule_sigma = 50.1

star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma,  star_name , sun_facule_tau , sun_facule_sigma )
star2.multi_component()     
        
        