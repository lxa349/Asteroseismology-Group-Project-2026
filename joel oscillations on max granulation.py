#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 14:05:44 2026

@author: maxtojoel
"""


# =!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!
'''
This file was created by merging Max's granulation work on the file "22-01-26.py" (granulation branch) with Joel's oscillation work on the file "joels_envelope_of_doom.ipynb" (oscillation branch).

I have begun to incorporate Joel's code into Max's OOP code, but this is not polished and is currently incomplete.

I have attempted to use comments to explain how different aspects work, this is also unpolished and unfinished.

Some values, such as sun_sigma, may be changed later on.
'''
# =!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!

import numpy as np
# from numpy.polynomial import Polynomial
from astropy.io import fits
import matplotlib.pyplot as plt

# power spectrum class defines star object and its attributes (parameters) and methods (calculations for more parameters, or functions used in producing the power spectrum)
# ================================================================================================================================================================
class power_spectrum:
    
    
    
    def __init__(self, mass_star, r_star, teff_star, nu_max_sun_uHz, teff_sun_K, tau_sun_s, star_name ="" , sun_sigma = 62.4):
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
        star_name : string
            name of star, used for plots. Default  ""
        sun_sigma : float
            Granulation ampltiude of sun, default = 1
        
            

        
        """
        
        
        #star properties
        self.star_name = star_name
        
        self.mass_star = float(mass_star)
        self.r_star = float(r_star)
        self.teff_star = float(teff_star)
        
        
        #constants
        self.nu_max_sun_uHz = float(nu_max_sun_uHz)
        self.teff_sun_K = float(teff_sun_K)
        self.tau_sun_s = float(tau_sun_s)
        
        #weird constants
        self.sigma_sun = float(sun_sigma)  #I think this will end up being calculated using code so define inside the class and change later
        #setting sigma_sun = 1 reverts the function back to the proportional relationship 
        
        self.max_rms_amplitude_sun = 2.1  #
        
        #calculate star properties for granulation
        
        self.freq_uHz = np.linspace(0.1, 10000.0, 500000) #range of frequencies where power spectrum is evaluated

        
        self.nu_max_ratio = self.calc_nu_max_ratio() #vmax ratio 
        self.nu_max_star_uHz = self.nu_max_ratio * self.nu_max_sun_uHz #v max
        
        self.luminosity_ratio = self.calc_luminosity_ratio() # L / solar luminosity, alpha L
        
        self.tau_gran_s = self.calc_tau_gran_s() # granulation time scale
        
        self.sigma_gran_ratio = self.calc_sigma_gran_ratio() # amplitude thing
        self.sigma_gran = self.sigma_gran_ratio*self.sigma_sun
        
        self.max_rms_amp_rad = self.calc_maximum_rms_amplitude_radial()
        #calculate star properties for oscillation
        
        
        #Radial modes, angular degrees, visibilities
        self.l = np.linspace(0,3,4)
        self.n = np.linspace(0,30,31)
        self.visibilities = [1, 1.505, 0.620, 0.075]
        
        self.nu_max_sun_uHz = 3104.0 # microhertz; not sure what the actual value is so i'll take this one, should be somewhere around this value anyway
        # teff_sun_K = 5772.0
        # tau_sun_s = 250
        self.delta_nu_sun = 135.1
        self.epsilon = 1.55  
        # sun_sigma = 62.4
        
        
        self.alpha = self.width_parameter(2.95, 0.39)
        self.width_alpha = self.width_parameter(3.08, 3.32)
        self.delta_width_dip = self.width_parameter(-0.47, 0.62)
        self.W_dip = self.width_parameter(4637, -141)
        self.nu_dip = self.width_parameter(2984, 60)
        
        self.print_debug()
        
    def model_comparison(self):
        
        self.psd_per_uHz = self.calc_granulation_psd_per_uHz() #array of power spectrum value at each frequency in freq_uHz
        self.plot_granulation_psd() #plots granulation spectru

        self.psd_per_uHz = self.calc_granulation_psd_per_uHz_test()
        self.plot_granulation_psd()


    def run_granulation(self):
        
        self.psd_per_uHz = self.calc_granulation_psd_per_uHz()     
        self.plot_granulation_psd() #plots granulation spectrum
        
    def return_granulation(self):
        self.psd_per_uHz = self.calc_granulation_psd_per_uHz() 
        return self.psd_per_uHz
    
    


    def print_debug(self):
        
        
        print("")
        print(f"For the star {self.star_name}, with properties M={self.mass_star} Msun, R={self.r_star} Rsun, Teff={self.teff_star} K ")
        print(f"nu_max={self.nu_max_star_uHz:.0f}, nu_max ratio = {self.nu_max_ratio:2f} ")
        print(f"star granulation amplitude = {self.sigma_gran:.2f}, granulation ratio = {self.sigma_gran_ratio:2f} ")
        print(f"Granulation time scale = {self.tau_gran_s}")
        
    
    
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
                
        nu_max_ratio =  (self.mass_star / (self.r_star**2)) * (self.teff_star / self.teff_sun_K) ** (-0.5)

        return nu_max_ratio
    
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
        
    def calc_sigma_gran_ratio(self):
        """
        Compute granulation amplitude ratio  using scaling relationship to remove constants from proportionallity constant. Ball et al equation 22 for proportional relationship

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
        

        Returns
        -------
        sigma_star : float
            sigma_star in 
        """
        
        

        nu_ratio = self.nu_max_star_uHz / self.nu_max_sun_uHz
        sigma_ratio = (self.luminosity_ratio ** 2) * (self.mass_star ** -3) * (self.teff_star / self.teff_sun_K) ** (-5.5) * (nu_ratio)
        
        

        return sigma_ratio
    
    
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


    def calc_granulation_psd_per_uHz_test(self):
        
        freq_uHz = np.asarray(self.freq_uHz)
        freq_Hz = freq_uHz * 1e-6
        
        psd_per_Hz = (4.0 * (self.sigma_gran ** 2) * self.tau_gran_s) / (1.0 + ((2.0 * np.pi * freq_Hz * self.tau_gran_s) ** 2) + ((2.0 * np.pi * freq_Hz * self.tau_gran_s) ** 4))
        
        psd_per_uHz = psd_per_Hz * 1e-6 #converts to uHz 
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
            f"Granulation Power Spectrum for star {self.star_name}\n"
            f"M={self.mass_star} Msun, R={self.r_star} Rsun, Teff={self.teff_star} K ,L = {self.luminosity_ratio:.2f} \n"
            f"nu_max={self.nu_max_star_uHz:.0f} µHz, tau={self.tau_gran_s:.0f}s, star granulation amplitude = {self.sigma_gran:.2f}\n "
            f"granulation amplitude ratio = {self.sigma_gran_ratio:2f}, vMax ratio = {self.nu_max_ratio:2f} \n"
            f"sun amplitude = {self.sigma_sun}, sun nu_max = {self.nu_max_sun_uHz:.2f}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def calc_maximum_rms_amplitude_radial(self):
        power = -0.093
        t_red_star = 8907*(self.luminosity_ratio**power)
        delta_t = 1250
        dwarf_suppression_beta = 1 - np.exp((self.teff_star - t_red_star)/(delta_t))

        max_rms_amplitude_star = self.max_rms_amplitude_sun * dwarf_suppression_beta * self.luminosity_ratio *(self.mass_star**-1) * ((self.teff_star / self.teff_sun_K) ** (-2))   
        
        return max_rms_amplitude_star
    
    def calc_nl_rms_amplitude_radial(self, n, l):
        
        nu_max = self.nu_max_star_uHz
        #print(nu_max)
        delta_nu = self.delta_nu_sun * (self.mass_star**0.5) * (self.r_star**(-1.5))
        
        #nu_nl.append(delta_nu * (n + (l/2) + epsilon))
        
        nu_nl = delta_nu * (n + (l/2) + self.epsilon)
        
        envelope_FWHM = 0.66 * (nu_max ** 0.88)
        
        if self.teff_star > self.teff_sun_K:
            envelope_FWHM = envelope_FWHM * (1+6e-4 * (self.teff_star - self.teff_sun_K))
                    
        width_envelope = envelope_FWHM / (2 * np.sqrt(2*np.log(2)))
            
        nl_rms_ampliude_star = (((self.calc_maximum_rms_amplitude_radial() ** 2) * np.exp(-((nu_nl - nu_max)**2)/(2*(width_envelope**2))))**0.5)
        
        return nl_rms_ampliude_star
    
    def width_parameter(self, a, b):
        nu_max = self.nu_max_sun_uHz * (self.mass_star / (self.r_star**2)) * (self.teff_star / self.teff_sun_K) ** (-0.5)
        P = (a*(nu_max/3090)) + b
        return P
    
    def calc_width_nl(self, n, l):
        
        nu_max = self.nu_max_sun_uHz * (self.mass_star / (self.r_star**2)) * (self.teff_star / self.teff_sun_K) ** (-0.5)
        delta_nu = self.delta_nu_sun * (self.mass_star**0.5) * (self.r_star**(-1.5))
        nu_nl = delta_nu * (n + (l/2) + self.epsilon) #ERRORRRRRRRRRRRRRRRRRRRRRRR!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        Lnwidth = ((self.alpha * np.log(nu_nl/nu_max)) + np.log(self.width_alpha)) + (np.log(self.delta_width_dip)/(1 + (((2 * np.log(nu_nl/self.nu_dip))/(np.log(self.W_dip/nu_max)))**2)))
        width_nl = np.exp(Lnwidth)

        return width_nl
    
    def calc_power_density(self):
        nu_max = self.nu_max_sun_uHz * (self.mass_star / (self.r_star**2)) * (self.teff_star / self.teff_sun_K) ** (-0.5)
        envelope_FWHM = 0.66 * (nu_max ** 0.88)
        delta_nu = self.delta_nu_sun * (self.mass_star**0.5) * (self.r_star**(-1.5))

        if self.teff_star > self.teff_sun_K:
                envelope_FWHM = envelope_FWHM * (1+6e-4 * (self.teff_star - self.teff_sun_K))
        #print(envelope_FWHM)

        standev_envelope = envelope_FWHM / (2*np.sqrt(2*np.log(2)))
        nu_envelope_min = nu_max - (3*standev_envelope)
        nu_envelope_max = nu_max + (3*standev_envelope)

        n_min = []
        n_max = []

        for i in self.l:
            n_min.append(int((nu_envelope_min/delta_nu) - ((i/2) + self.epsilon)))
            n_max.append(int((nu_envelope_max/delta_nu) - ((i/2) + self.epsilon)))

        nu_range = self.freq_uHz
        #power_density = np.zeros(len(nu_range))
        power_density = np.zeros(len(nu_range))

        for j in self.l:
            #power_density = np.zeros(len(nu_range))
            
            for i in np.linspace(min(n_min), max(n_max), (max(n_max)-min(n_min))+1):
                nu_nl = delta_nu * (i + (j/2) + self.epsilon)
                amplitude_nl = self.visibilities[int(j)] * self.calc_nl_rms_amplitude_radial(i, j)
                width_nl = self.calc_width_nl(i, j)
                power_amplitude = calc_power_amplitude(amplitude_nl, width_nl)
                power_density +=(power_amplitude * lorentz(nu_range, nu_nl, width_nl))
        return power_density
    
# ================================================================================================================================================================        
        



# granulation_background() requires the star's mass and radius in solar masses, effective temperature, and the star's name
# also requires the sun's nu_max, granulation timescale (tau_sun_s) and amplitude (sun_sigma) set prior 

# obtains the PIC data
def get_pic_data():

    # CHANGE FILE PATH
    #PIC = fits.util.get_testdata_filepath('data/LOPS2PICtarget2.1.0.1-t-fg-c-scv.fits')
    PIC = fits.open('/home/isaac/Documents/Asteroseismology Project/LOPS2PICtarget2.1.0.1-t-fg-c-scv.fits') #see group projects canvas page
    #headers = np.asarray(PIC[1].header)
    data = np.asarray(PIC[1].data)
    print(f"{len(data["PICid"])} entries")
    
    # Makes a list (then a numpy array) of masses of stars below 1.5 solar masses
    joelslegacy = []

    for i in range(len(data["Mass"])):
        if data["Mass"][i] < 1.5:
            joelslegacy.append(data[i])

    data_reduced = np.asarray(joelslegacy)
    
    return data_reduced
    

# ================================================================================================================================================================
# Function for plotting Lorentz curves (Cauchy distributions)
def lorentz(freq, centroid, FWHM): 
    #scale parameter is the half-width-half-maximum value of the Lorentz curve
    return (1/np.pi) * ((FWHM/2) / ((freq - centroid)**2 + (FWHM/2)**2))

def calc_power_amplitude(amp,width):
    return (2/np.pi)*((amp**2)/width)

def max_freq(M, R, T):
    solar_temp = 5778
    dT = T / solar_temp
    return( float(M * R**(-2) * dT**(-0.5)) ) #returns max frequency in terms of max frequency of the Sun
# ================================================================================================================================================================

# Known values, to be incorporated into the power_spectrum class as attributes
nu_max_sun_uHz = 3090.0
teff_sun_K = 5777.0
tau_sun_s = 250      
        

def main():
    # star_index is the index for the element of stellar_data that's being used, ultimately this will be in a loop to go through each given star
    # changing this will change which star is being plotted
    # the whole of the code below should eventually be a method of the power spectrum class
    star_index = 596
    stellar_data = get_pic_data()
    
    # Define star and its parameters 
    for star_index in range(596, 100001, 25000):
        star_name = str(stellar_data["StarName"][star_index])
        star_mass = float(stellar_data["Mass"][star_index])
        star_radius = float(stellar_data["Radius"][star_index])
        star_teff = float(stellar_data["Teff"][star_index])
        
        test_star = power_spectrum(star_mass, star_radius, star_teff, nu_max_sun_uHz, teff_sun_K, tau_sun_s, star_name)
        
        granulation_psd = test_star.return_granulation()
        power_density = test_star.calc_power_density()   
        
        P = power_density + granulation_psd
        
        # overtones
        plt.plot(test_star.freq_uHz, power_density, color = "blue", label = "p-mode oscillations")
        # plt.axvline(max_freq(test_star.mass_star, test_star.r_star, test_star.teff_star) * 3100, ls='--', color='r', label = "nu_max")
        plt.axvline(test_star.nu_max_star_uHz, ls='--', color='r', label = "nu_max")
        plt.xlim(test_star.nu_max_star_uHz-600,test_star.nu_max_star_uHz+600)
        plt.title("Oscillations Power Spectrum")
        plt.xlabel("Frequency /microHertz")
        plt.ylabel("Power /ppm^2 Hz^-1")
        plt.legend()
        plt.show()
        
        # power spectrum
        plt.loglog(test_star.freq_uHz, P, color = "blue", label = "power spectrum")
        # plt.axvline(max_freq(test_star.mass_star, test_star.r_star, test_star.teff_star) * 3100, ls='--', color='r', label = "nu_max")
        plt.axvline(test_star.nu_max_star_uHz, ls='--', color='r', label = "nu_max")
        plt.grid(which='major')
        plt.title("Power Spectrum - Granulation and Oscillations")
        plt.ylabel("Power (ppm^2 Hz^-1)")
        plt.xlabel("Frequency (uHz)")
        plt.legend()
        plt.show()
        
        # # power spectrum zoomed
        # plt.loglog(test_star.freq_uHz, P, color = "blue", label = "power spectrum")
        # # plt.axvline(max_freq(test_star.mass_star, test_star.r_star, test_star.teff_star) * 3100, ls='--', color='r', label = "nu_max")
        # plt.axvline(test_star.nu_max_star_uHz, ls='--', color='r', label = "nu_max")
        # plt.grid(which='major')
        # plt.title("Power Spectrum - Granulation and Oscillations")
        # plt.ylabel("Power (ppm^2 Hz^-1)")
        # plt.xlabel("Frequency (uHz)")
        # plt.xlim(test_star.nu_max_star_uHz-600,test_star.nu_max_star_uHz+600)
        # plt.legend()
        # plt.show()

# ================================================================================================================================================================

main()