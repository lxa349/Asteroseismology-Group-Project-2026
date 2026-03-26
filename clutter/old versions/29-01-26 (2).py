# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:37:38 2026

@author: mxt216

"""



import numpy as np
# from numpy.polynomial import Polynomial
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
        
        self.sun_max_rms_amplitude = 2.1  
        
        
        
        #line space constants
        
        self.freq_powerspectrum_uHz = np.linspace(0.1, 10000.0, 500000) #range of frequencies where power spectrum is evaluated
        self.angular_degree = np.linspace(0,3,4) #List of angular degrees l for oscillations 
        self.radial_mode = np.linspace(0,30,31) #list of radial modes n for oscillations
        
        self.visibilites = [1, 1.505, 0.620, 0.075] #Scales amplitude of each angular degree oscillation. 
        
        
        
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
        
        self.star_delta_nu = self.sun_delta_nu * (self.star_mass**0.5) * (self.star_radius**(-1.5))
        
        self.max_rms_amplitude_radial_ratio = self.calc_maximum_rms_amplitude_radial_ratio()
        self.star_max_rms_amplitude_radial = self.max_rms_amplitude_radial_ratio *  self.sun_max_rms_amplitude
        
        self.star_standev_envelop = self.calc_envelope_width()
        
        
        #line width constants from paper
        self.alpha = self.calc_width_parameter(2.95, 0.39)
        self.width_alpha = self.calc_width_parameter(3.08, 3.32)
        self.delta_width_dip = self.calc_width_parameter(-0.47, 0.62)
        self.W_dip = self.calc_width_parameter(4637, -141)
        self.nu_dip = self.calc_width_parameter(2984, 60)
        


    def single_component(self):
        """
        Models the the background as one component due to granulation

        Returns
        -------
        granulation_component : arraay
            granulation component of background PSD .
        
        total_background : array
            Total backgorund pSD due to all background components

        """
        

        granulation_component = self.calc_component_psd( self.star_granulation_sigma, self.star_granulation_tau)
        
        total_background = granulation_component
        
        return total_background, granulation_component
        
        
    def multi_component(self):
        """
        Models the background as two components. One from granulation, one from facule

        Returns
        -------
        total_background : array
            Total backgorund pSD due to all background components
        facule_component : array
            facule component of background PSD
            

        """
       
        
        
        self.star_facule_sigma = self.sigma_ratio * self.sun_facule_sigma
        self.star_facule_tau = self.star_tau_ratio * self.sun_facule_tau
        
        
        granulation_component = self.calc_component_psd( self.star_granulation_sigma, self.star_granulation_tau)
        
        facule_component = self.calc_component_psd( self.star_facule_sigma, self.star_facule_tau)
        
        total_background = granulation_component + facule_component
        
        return total_background, granulation_component, facule_component
        
        
    
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
    
    def calc_nu_nl(self, n,l):
        """
        Calculates the frequency of  a given n and l value

        Parameters
        ----------
        n : interger
            radial mode.
        l : interger
            angular degree.

        Returns
        -------
        nu_nl : float
            frequency of the n l mode.

        """
        
        nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon)
        
        return nu_nl
    
    def calc_width_nl(self, n, l):
        """
        Calculates the FWHM of the n, l mode frequency peak 

        Parameters
        ----------
        n : interger
           radial mode.
        l : interger
           angular degree.

        Returns
        -------
        width_nl : float
            Width of the n,l frequency mode.

        """
        
        nu_nl = self.calc_nu_nl(n,l)
        
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
        width_envelope: float.
            FWHM of oscillation envelope

        """
        
        envelope_FWHM = 0.66 * (self.star_nu_max**0.88)
        
        if self.star_teff > self.sun_teff:
            envelope_FWHM = envelope_FWHM * (1+6e-4 * (self.star_teff-self.sun_teff))
            
        width_envelope = envelope_FWHM / (2 * np.sqrt(2*np.log(2)))
 
        return width_envelope

    def calc_powder_density(self):
        """
        Calculates the power density component due to the oscillations. 
        

        Returns
        -------
        power_density_oscillations : arrray
            Power spectrum due to oscillations .

        """
        
        #n_min, n_max =  self.calc_n_min_max()
        #n_min_max_linspace = np.linspace(min(n_min), max(n_max), (max(n_max)-min(n_min))+1) #lin space of n values  to itterate  over

        
        power_density_oscillations = np.zeros(len(self.freq_powerspectrum_uHz)) #creates base array same size as frequency array 
        
        for l in self.angular_degree: #itterates over all possible l angular degrees
            
            #self.radial_mode
            for n in self.radial_mode: #itterates over all possible n radial degrees 
            
                #calculates the propeties of the current mode
                nu_nl = self.calc_nu_nl(n,l) 
                width_nl = self.calc_width_nl(n, l)
                nl_rms_amplitude_radial = self.calc_nl_rms_amplitude_radial(nu_nl)
                
                amplitude_nl = self.visibilites[int(l)] * nl_rms_amplitude_radial #scales amplitude of angular mode
                
                power_amplitude = calc_power_amplitude(amplitude_nl, width_nl) #Unscaled height of frequency mode
                lorentz_value = calc_lorentz(self.freq_powerspectrum_uHz, nu_nl, width_nl) #Scaling for the height
                
                power_density_oscillations +=(power_amplitude*lorentz_value) #Scales height of frequency mode using lorentz, then adds to power specturm array
                
        return power_density_oscillations
                
                
    def calc_dwarf_suppression_beta_term(self):
        """
        Calculates the dwarf suppression term.
        Hot dwarfs have a apparent decrease in amplitude, beta term accounts for that when calculating maximum rms amplitude of all radial modes.

        Returns
        -------
        dwarf_suppression_beta : TYPE
            DESCRIPTION.

        """
        
        
        
        t_red_star = 8900

        
        delta_t = 1250

        dwarf_suppression_beta = 1 - np.exp((self.star_teff - t_red_star)/(delta_t))
        
        return dwarf_suppression_beta

        
                
                
                
    def calc_maximum_rms_amplitude_radial_ratio(self):
        """
        Calculates the maximum rms amplitude of all raidal mode, as a ratio to the suns value using scaling relationships. Ball et all equation 16
        Includes dwarf supression term

        Returns
        -------
        star_max_rms_amplitude_ratio : float
            ratio of the specific stars maximum rms amplitud to the suns .

        """
        
        power = -0.093
        
        dwarf_suppression_beta = self.calc_dwarf_suppression_beta_term()

        star_max_rms_amplitude_ratio=  dwarf_suppression_beta * self.luminosity_ratio *(self.star_mass**-1) * ((self.star_teff / self.sun_teff) ** (-2))   
        
        return star_max_rms_amplitude_ratio
        
    def calc_nl_rms_amplitude_radial(self, nu_nl):
        """
        Calculates the rms amplitude of the mode that has frequency nu_nl 

        Parameters
        ----------
        nu_nl : float
            frequency of mode with n radial mode and l angular mode.

        Returns
        -------
        star_rms_amplitude : float
            rms amplitude of mode.

        """
        
        
        
        star_rms_amplitude = (((self.star_max_rms_amplitude_radial ** 2) * np.exp(-((nu_nl-self.star_nu_max)**2)/(2*(self.star_standev_envelop**2))))**0.5)
        
        return star_rms_amplitude
        
        
        
    def calc_n_min_max(self):
        """
        Joel code, now defunct
        

        Returns
        -------
        n_min : TYPE
            DESCRIPTION.
        n_max : TYPE
            DESCRIPTION.

        """
        
        n_min = []
        n_max = []
        
        
        
        nu_envelope_min = self.star_nu_max - (3*self.star_standev_envelop)
        nu_envelope_max = self.star_nu_max + (3*self.star_standev_envelop)
        
        for i in self.angular_degree:
            
            n_min.append(int((nu_envelope_min/self.star_delta_nu) - ((i/2) + self.epsilon)))
            n_max.append(int((nu_envelope_max/self.star_delta_nu) - ((i/2) + self.epsilon)))
            
        return n_min, n_max
        
        

        
        
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
        Plots power specturm of multi component model. Defunt function, now plotted outside of the class

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

def calc_power_amplitude(amp,width):
    
    x = (2/np.pi)*((amp**2)/width)
    return x 
        
def calc_lorentz(freq, centroid, FWHM):
    
    
    #x = (1/np.pi) * ((FWHM/2) / ((freq - centroid)**2 + (FWHM/2)**2))
    x = ((FWHM/2)**2 / ((freq - centroid)**2 + (FWHM/2)**2))
    return x 
        
        
star_mass, star_radius, star_teff = 1.223, 1.357, 6325
star_name = "Kepler 410"
sun_nu_max = 3090
sun_teff = 5772.0
sun_granulation_tau = 214
sun_granulation_sigma = 63




star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)

total_background, granulation_component = star.single_component()
oscillation_psd = star.calc_powder_density()
total_psd = total_background + oscillation_psd

#plots

plt.loglog(star.freq_powerspectrum_uHz, total_psd, color = 'blue', label = "power spectrum")
plt.axvline(star.star_nu_max, ls = '--', color = 'r', label = "nu_max")
plt.grid(which='major')
plt.title("Power Spectrum - Granulation and Oscillations - One component")
plt.ylabel("Power (ppm^2 Hz^-1)")
plt.xlabel("Frequency (uHz)")
plt.legend()
plt.show()

plt.figure()
plt.loglog(star.freq_powerspectrum_uHz,total_background, label = 'Total ')
plt.loglog(star.freq_powerspectrum_uHz, granulation_component, label = 'Granulation component')
plt.axvline(star.star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
plt.ylim(bottom=0.1)
plt.xlim(left=10)
plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
plt.ylabel(r"Background power, ppm^2 / µHz")
plt.title("Background power spectrum. One component model")
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(star.freq_powerspectrum_uHz, oscillation_psd, color = "blue", label = "p-mode oscillations")
# plt.axvline(max_freq(test_star.mass_star, test_star.r_star, test_star.teff_star) * 3100, ls='--', color='r', label = "nu_max")
plt.axvline(star.star_nu_max, ls='--', color='r', label = "nu_max")
#plt.xlim(star.star_nu_max-600,star.star_nu_max+600)
plt.title("Oscillations Power Spectrum")
plt.xlabel("Frequency /microHertz")
plt.ylabel("Power /ppm^2 Hz^-1")
plt.legend()
plt.show()


#Multi component background plot 
"""


star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)
total_background, granulation_component, facule_component = star2.multi_component()
oscillation_psd = star2.calc_powder_density()
total_psd = total_background + oscillation_psd

plt.loglog(star2.freq_powerspectrum_uHz, total_psd, color = 'blue', label = "power spectrum")
plt.axvline(star2.star_nu_max, ls = '--', color = 'r', label = "nu_max")
plt.grid(which='major')
plt.title("Power Spectrum - Granulation and Oscillations")
plt.ylabel("Power (ppm^2 Hz^-1)")
plt.xlabel("Frequency (uHz)")
plt.legend()
plt.show()

plt.figure()
plt.loglog(star2.freq_powerspectrum_uHz,total_background, label = 'Total ')
plt.loglog(star2.freq_powerspectrum_uHz, facule_component, label = 'Facule component')
plt.loglog(star2.freq_powerspectrum_uHz, granulation_component, label = 'Granulation component')
plt.axvline(star2.star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
plt.ylim(bottom=0.1)
plt.xlim(left=10)
plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
plt.ylabel(r"Granulation power, ppm^2 / µHz")
plt.legend()
plt.tight_layout()
plt.show()
"""










"""
sun_granulation_tau = 214.3
sun_granulation_sigma = 62.4

sun_facule_tau = 65.8
sun_facule_sigma = 50.1

star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma,  star_name , sun_facule_tau , sun_facule_sigma )
star2.multi_component()     
"""
        
