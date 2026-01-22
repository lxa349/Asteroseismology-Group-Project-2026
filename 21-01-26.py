# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 14:05:44 2026

@author: maxto
"""

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt


class granulation_background:
    
    
    
    def __init__(self, mass_star, r_star, teff_star, nu_max_sun_uHz, teff_sun_K, tau_sun_s):
        """
        

        Parameters
        ----------
        mass_star : float
            Stellar mass in solar masses
        r_star : float
            Stellar radius in solar radii
        teff_star : float
            Effective temperature in Kelvin
        nu_max_sun_uHz : float
            nu max of sun (vmax) in microHz.
        teff_sun_K : float
            Effective temperature of the sun in kelvin.
        tau_sun_s : float
            granulation time scale of the sun in seconds.

        
        """
        
        
        #star properties
        self.mass_star = float(mass_star)
        self.r_star = float(r_star)
        self.teff_star = float(teff_star)
        
        #constants
        self.nu_max_sun_uHz = float(nu_max_sun_uHz)
        self.teff_sun_K = float(teff_sun_K)
        self.tau_sun_s = float(tau_sun_s)
        
        #weird constants
        self.sigma_sun =23  #I think this will end up being calculated using code so define inside the class and change later
        #setting sigma_sun = 1 reverts the function back to the proportional relationship 
        
        self.max_rms_amplitude_sun = 2.1  #
        
        #calculate star properties for granulation
        
        self.freq_uHz = np.linspace(0.1, 10000.0, 500000) #range of frequencies where power spectrum is evaluated

        
        self.nu_max_star_uHz = self.calc_nu_max_uHz() #v max
        self.luminosity_ratio = self.calc_luminosity_ratio() # L / solar luminosity, alpha L
        self.tau_gran_s = self.calc_tau_gran_s() # granulation time scale
        self.sigma_gran = self.calc_sigma_gran() # amplitude thing
        
        
        
        self.psd_per_uHz = self.calc_granulation_psd_per_uHz() #array of power spectrum value at each frequency in freq_uHz
        #self.psd_per_uHz = self.calc_granulation_psd_per_Uhz_canvas_equation()


        self.plot_granulation_psd() #plots granulation spectru
        
        #calculate star properties for oscillation
        



    
    def calc_nu_max_uHz(self):
        """
        Calculate nu_max using scaling relationship

        
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
                
        nu_max = self.nu_max_sun_uHz * (self.mass_star / (self.r_star**2)) * (self.teff_star / self.teff_sun_K) ** (-0.5)

        return nu_max
    
    def calc_nu_nl(self):
        
        nu_nl = 5
        
        return nu_nl
        
    def calc_luminosity_ratio(self):
        """
        Compute luminosity ratio L/Lsun using 
       
            L/Lsun = (R/Rsun)^2 * (Teff/Teff_sun)^4

        
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
        luminosity_ratio :float
            Luminosity ratio L/Lsun
        """
        
        luminosity_ratio = (self.r_star**2) * (self.teff_star / self.teff_sun_K) ** 4
        
        return luminosity_ratio
    
    def calc_tau_gran_s(self):
        """
        Compute granulation timescale tau_gran in seconds using equation from paper. Ball et al equation 21
        
        tau_gran = tau_sun * (nu_max / nu_max_sun)^(-1)

    

        Parameters
        ----------
        nu_max_star_uHz : float
            nu_max for the star in microHz

        Returns
        -------
        tau_gran : float
            time scale of granulation in seconds
        """
        
        tau_gran = self.tau_sun_s * (self.nu_max_star_uHz / self.nu_max_sun_uHz) ** (-1)

        return tau_gran
        
    def calc_sigma_gran(self):
        """
        Compute granulation amplitude sigma_gran using scaling relationship to remove constants from proportionallity constant. Ball et al equation 22

        Scaling relationship of granulation amplitude,
            sigma_star/ sigma_sun =
                (L/Lsun)^2 * (M/Msun)^(-3) * (Teff/Teff_sun)^(-5.5) * (nu_max/nu_max_sun)
                
        

       

        Parameters
        ----------
        nu_max_star_uHz : float
            nu_max for the star in microHz
        luminosity_ratio : float
            L/Lsun
        mass_star : float
            Stellar mass in solar masses
        teff_star : float
            Effective temperature in Kelvin
        sigma_sun : float
            Granulation amplitude of sun in 

        Returns
        -------
        sigma_star : float
            sigma_star in 
        """
        
        

        nu_ratio = self.nu_max_star_uHz / self.nu_max_sun_uHz
        sigma_ratio = (self.luminosity_ratio ** 2) * (self.mass_star ** -3) * (self.teff_star / self.teff_sun_K) ** (-5.5) * (nu_ratio)
        
        sigma_star = sigma_ratio * self.sigma_sun

        return sigma_star
    
    
    def calc_granulation_psd_per_uHz(self):
        """
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

    def calc_granulation_psd_per_Uhz_canvas_equation(self):
        
        """
        Same function as calc_granulation_psd_per_uHz but uses the equation provided on canvas instead of Ball et all
        Removes granulation time scale from the numerator 
        
        Calculates power spectrum value at an array of given frequencies using l
            P(ν) = 4*sigma^2 / (1 + (2πνtau)^2)
            
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
        
        psd_per_Hz = (4.0 * (self.sigma_gran ** 2)) / (1.0 + (2.0 * np.pi * freq_Hz * self.tau_gran_s) ** demominator_power)
        psd_per_uHz = psd_per_Hz * 1e6 #converts to uHz 
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
        plt.loglog(self.freq_uHz, self.psd_per_uHz)
        plt.axvline(self.nu_max_star_uHz, linestyle="--", label=r"$\nu_{\max}$")
        
        plt.ylim(bottom=0.1)
        plt.xlim(left=10) 
        
        
        plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
        plt.ylabel(r"Granulation power, ppm^2 / µHz")
        plt.title(
            f"Granulation Power Spectrum \n"
            f"M={self.mass_star} Msun, R={self.r_star} Rsun, Teff={self.teff_star} K ,L = {self.luminosity_ratio:.2f} \n"
            f"nu_max={self.nu_max_star_uHz:.0f} µHz, tau={self.tau_gran_s:.0f}s, star granulation amplitude = {self.sigma_gran:.2f}\n "
            f"sun amplitude = {self.sigma_sun}, sun nu_max = {self.nu_max_sun_uHz:.2f}\n"
            
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    
        

class osciillation_stuff:
    
    def __init__(self, mass_star, r_star, teff_star, nu_max_sun_uHz, teff_sun_K, tau_sun_s):
        
        self.mass_star = float(mass_star)
        self.r_star = float(r_star)
        self.teff_star = float(teff_star)
        
        #constants
        self.nu_max_sun_uHz = float(nu_max_sun_uHz)
        self.teff_sun_K = float(teff_sun_K)
        self.tau_sun_s = float(tau_sun_s)
        
        
    def calc_beta_temperature_suppression_term(self):
        """
        For a given star, calculates beta factor using  equation 17 from Ball et al
        
        beta = 1 - exp (t_eff_star - t_red) / delta_T)
    

        Returns
        -------
        beta : float
            Temperature suppression term  .

        """
        
        
        
        power = -0.093
        
        t_red_star = 8907*(self.luminosity_ratio**power)
        delta_t = 1250
        
        beta = 1 - np.exp((self.teff_star - t_red_star)/(delta_t))
        return beta
        
    def calc_maximum_rms_amplitude_radial(self):
        """
        For a given star, calculates the maximum rms amplitude for radial mode using equation 16 from Ball et al

        Returns
        -------
        max_rms_amplitude_star : float
            maximum rms aamplitude of the radial mode.

        """
        
        
        max_rms_amplitude_star = self.max_rms_amplitude_sun * self.dwarf_suppression_beta * self.luminosity_ratio *( self.mass_star**-1) * ((self.teff_star / self.teff_sun_K) ** (-2))
        
        return max_rms_amplitude_star
    
    def calc_envelope_gaussian_width(self):
        """
        Calculates the envelope gaussian width using equation 19 and the line below equation 20 from Ball et al
        
        Envelop = 0.66 * (Vmax)^0.88
        
        Parameters input are in units of microHz hence the conversion terms are removed from the Ball et al equations. 
        
        

        Returns
        -------
        width_envelope : float
            Gaussian width of the evelope .

        """
        
        
        envelope_FWHM = 0.66 * (self.nu_max_star_uHz ** 0.88)
        
        if  self.teff_star > self.teff_sun_K:
            envelope_FWHM = envelope_FWHM * (1+6e-4 * (self.teff_star-self.teff_sun_K))
            
        width_envelope = envelope_FWHM / (2 * np.sqrt(2*np.log(2)))
            
        return width_envelope
        
        
        
        
        
        
    
        
        
        
mass_star = 1.223
r_star = 1.357
teff_star = 6325
nu_max_sun_uHz = 3090.0
teff_sun_K = 5772.0
tau_sun_s = 250

    
star = granulation_background(mass_star, r_star, teff_star, nu_max_sun_uHz, teff_sun_K, tau_sun_s)