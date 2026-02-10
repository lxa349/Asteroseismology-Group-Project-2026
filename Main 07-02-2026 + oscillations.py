# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:37:38 2026

@author: mxt216

"""



import numpy as np
# from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

class power_spectrum:
    
    
    def __init__(self,star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma,  star_name ="", sun_facule_tau = 1, sun_facule_sigma = 1, sun_supergranulation_sigma = 1.9*10**6, sun_supergranulation_tau = 129600):
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
        sun_supergranulation_tau : TYPE, optional
            Facule timescale of sun. The default is 129600.
        sun_supergranulation_sigma : TYPE, optional
            Facule amplitude of sun . The default is 1.9 * 10**6.

        
        """
        
        
        
        #sun constants
        
        self.sun_nu_max = float(sun_nu_max)
        self.sun_teff = float(sun_teff)
        
        self.sun_granulation_tau = float(sun_granulation_tau)
        self.sun_granulation_sigma = float(sun_granulation_sigma)
        
        self.sun_facule_tau = float(sun_facule_tau)
        self.sun_facule_sigma = float(sun_facule_sigma)
        
        self.sun_supergranulation_tau = float(sun_supergranulation_tau) #default value is best i could find from research: https://ui.adsabs.harvard.edu/abs/2018LRSP...15....6R/abstract#:~:text=Rieutord%2C%20Michel-,Abstract,a%20selection%20of%20recent%20findings. <-- see chapter about timescales
        self.sun_supergranulation_sigma = float(sun_supergranulation_sigma) #same again, different source: https://www.researchgate.net/profile/P-Palle/publication/234514213_A_measurement_of_the_background_solar_velocity_spectrum/links/00b495177361416eba000000/A-measurement-of-the-background-solar-velocity-spectrum.pdf
        
        self.sun_max_rms_amplitude = 2.1  
        
        
        
        #line space constants
        
        self.freq_powerspectrum_uHz = np.linspace(0.1, 10000.0, 500000) #range of frequencies where power spectrum is evaluated
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
        
        self.tau_ratio = self.calc_tau_ratio() #granulation time scale         
        self.sigma_ratio = self.calc_sigma_ratio()
        
        self.star_property_list = [[self.sun_granulation_sigma, self.sun_granulation_tau],[self.sun_facule_sigma, self.sun_facule_tau],[self.sun_supergranulation_sigma, self.sun_supergranulation_tau]]
        
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
        
        
    #Background models wrapper functions
        
    def theoretical_sigma_tau(self, model = "karoff"):
        """
        Compares the different models of granulation for a sigma, tau etc caluclated using theoreitcal scaling relationships

        Parameters
        ----------
        model : string, optional
            Decides on which model is used. The default is "karoff".
            "karoff" refers to model described in C. Karoff 2013, the first two terms of equation 2
            "ball" refers to the model described in Warrick H. Ball 2018, equation 23 
            "kallinger" refers to the model described in Kallinger et al: The connection between stellar granulation and oscillations. Equaation 2 model F, sumation terms only 

        Returns
        -------
        granulation_component : array
            The component of the background spectrum due to granulation.
        facule_component : array
            The component of the backgroudn spectrum due to facule.

        """
        
        
        
        if model == "karoff":
            
            star_granulation_tau = self.tau_ratio * self.sun_granulation_tau
            star_granulation_sigma = self.sigma_ratio * self.sun_granulation_sigma
            
            star_facule_sigma = self.sigma_ratio * self.sun_facule_sigma
            star_facule_tau = self.tau_ratio * self.sun_facule_tau
            
            
            
            print("Theory calculated values")
            print(f"star granulation sigma {star_granulation_sigma}, star granulation tau {star_granulation_tau}")        
            print(f"star facule sigma {star_facule_sigma}, star facule tau {star_facule_tau}") 
            
            granulation_component, facule_component = self.calc_karoff_granulation_facule_model(star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau)
        
        elif model == 'ball':
            
            star_granulation_tau = self.tau_ratio * self.sun_granulation_tau
            star_granulation_sigma = self.sigma_ratio * self.sun_granulation_sigma
            
            
            granulation_component =  self.calc_granulation_psd_ball(star_granulation_sigma, star_granulation_tau)
            facule_component = 0 #measure to get code to run, unsure if correct
            
        elif model == "kallinger":
            
            a,  b_1, b_2 = self.calc_scaling_values_kalinger() #calculates sigma, tau for both components of kallinger model using scaling relationships
                
            granulation_component, facule_component = self.calc_kallinger_model(a, b_1, b_2)
            
            
        else:
            
            granulation_component, facule_component = self.multi_component(star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau)

        return granulation_component, facule_component
    

        
            
            
        
        
    def defined_sigma_tau(self, star_granulation_sigma, star_granulation_tau, star_facule_sigma = 1, star_facule_tau = 1, model = "karoff"):
        """
        Compares the different models of granulation for a sigma, tau etc  using given values. Ie from a paper / test values
        
        Note that not all models acount for facule.

        Parameters
        ----------
        star_granulation_sigma : float
            sigma of granulation of the star.
        star_granulation_tau : float
            tau tiem scale of granulation of the star.
        star_facule_sigma : float, optional
            sigma of facule of the star. The default is 1.
        star_facule_tau : float, optional
            DESCRIPTION. The default is 1.
        model : string, optional
            Decides on which model is used. The default is "karoff".
            "karoff" refers to model described in C. Karoff 2013, the first two terms of equation 2
            "ball" refers to the model described in Warrick H. Ball 2018, equation 23 


        Returns
        -------
        granulation_component : array
            The component of the background spectrum due to granulation.
        facule_component : array
            The component of the backgroudn spectrum due to facule.


        """
        
        
        
        if model == "karoff":
            
            print("Practical defiend values ")
            print(f"star granulation sigma {star_granulation_sigma}, star granulation tau {star_granulation_tau}")        
            print(f"star facule sigma {star_facule_sigma}, star facule tau {star_facule_tau}") 
            
            granulation_component, facule_component = self.calc_karoff_granulation_facule_model(star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau)
            
        
        elif model == "ball":
            
            granulation_component, facule_component = self.calc_granulation_psd_ball(star_granulation_sigma, star_granulation_tau)
        
        else:
            
            granulation_component, facule_component = self.multi_component(star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau)
        
            
        
        return granulation_component, facule_component
        


        
        
    def multi_component(self, star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau):
        """
        Models the background due to granulation and facule as two components that take an identical form. One from granulation, one from facule

        Returns
        -------
        total_background : array
            Total backgorund pSD due to all background components
        facule_component : array
            facule component of background PSD
            

        """
       
        
      
        
        granulation_component = self.calc_component_psd( star_granulation_sigma, star_granulation_tau)
        
        facule_component = self.calc_component_psd( star_facule_sigma, star_facule_tau)
        
        
        return  granulation_component, facule_component
    
    def multi_component_scalable(self):
        """
        Not used atm 
        multi_component but via for loop. 
        

        Returns
        -------
        total_background : array
            Total backgorund pSD due to all background components

        """       
        
        mylist = []
        
        for i in self.star_property_list:
            background_component = self.calc_component_psd(i[0] * self.sigma_ratio, i[1] * self.tau_ratio)
            mylist.append(background_component)
        
        return mylist
    
    #Different background models
    
    def calc_kallinger_model(self, a, b_1, b_2):
        """
        Kallinger 2014 model of background granulation / other stuff. Equation 2 model f. Two loretnz functions with a fixed power of 4, with two seperate sigmas, taus etc
        
            
        P(v) = xi * (a^2 / b_1 ) / (1 + (v/b_1)^4) + xi * (a^2 / b_2 ) / (1 + (v/b_2)^4)
        
        granulation_component refers to the first term above. Facule is the 2nd term. 
        
        ai = rms amplitude (sigma) of ith component
        bi = (1 / 2 pi tau), tau = characteristic timescale 
            
            
            
            


        Returns
        -------
        granulation_component : array
            The component of the background spectrum due to granulation.
        facule_component : array
            The component of the backgroudn spectrum due to facule.

        """
        
        freq_uHz = np.asarray(self.freq_powerspectrum_uHz)
        #freq_Hz = freq_uHz * 1e-6
        
        xi = 2 * np.sqrt(2) / np.pi#normalisation factor, defined in paper 
        
        granulation_component = (xi * ((a**2)/b_1))/ (1 + (freq_uHz / b_1)**4)
        facule_component = (xi * ((a**2)/b_2))/ (1 + (freq_uHz / b_2)**4)
        
        return granulation_component, facule_component
        
        
    
    def calc_karoff_granulation_facule_model(self,  star_granulation_sigma, star_granulation_tau, star_facule_sigma, star_facule_tau):
        """
        Granulation and facule model according to Karoff 2013

        Parameters
        ----------
        star_granulation_sigma : TYPE
            DESCRIPTION.
        star_granulation_tau : TYPE
            DESCRIPTION.
        star_facule_sigma : TYPE
            DESCRIPTION.
        star_facule_tau : TYPE
            DESCRIPTION.

        Returns
        -------
        granulation_component : TYPE
            DESCRIPTION.
        facule_component : TYPE
            DESCRIPTION.

        """

         

        
        freq_uHz = np.asarray(self.freq_powerspectrum_uHz)
        freq_Hz = freq_uHz * 1e-6
        
        granulation_component = (3.47 * (star_granulation_sigma**2)*star_granulation_tau)/(1+(2*np.pi*freq_Hz * star_granulation_tau)**3.5)
        facule_component = (6.20 * (star_facule_sigma**2)*star_facule_tau)/(1+(2*np.pi*freq_Hz * star_facule_tau)**6.2)
        
        granulation_component = granulation_component * 1e-6
        facule_component = facule_component * 1e-6
        
        return granulation_component, facule_component
    
    def calc_granulation_psd_ball(self, star_granulation_sigma, star_granulation_tau):
        """
        
        Ball  2018 model fo granulation         
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
        freq_uHz = np.asarray(self.freq_powerspectrum_uHz)
        freq_Hz = freq_uHz * 1e-6

        demominator_power  = 4
        
        psd_per_Hz = (4.0 * (star_granulation_sigma ** 2) * star_granulation_tau) / (1.0 + (2.0 * np.pi * freq_Hz * star_granulation_tau) ** demominator_power)
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

    #Star properties calculator         
    
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
    
    def calc_scaling_values_kalinger(self):
        """
        Calculates the different components of the kalinger model for the given star using the relationships given in table 2
        ai = rms amplitude (sigma) of ith component
        bi = (1 / 2 pi tau), tau = characteristic timescale 
        
        Note with the model, each component isn't specified as being due to facule or granulation. Just keeping with variable naming schemes 
        
        Note the paper says granulation_sigma = facule_sigma and has only 1 rms amplitude
        Scaling relationships as given in kallinger paper table 2

        Returns
        -------
        a : float
            rms amplitude (sigma) of both components.
        b_1 : float
            term related to characteristic timesclale of the first background component.
        b_2 : float
            term related to characteristic timesclale of the second background component.

        """
        
        
        
        
        a = 3382 * (self.star_nu_max)**(-0.609) #a == sigma_granulation, without mass term
        print(f"a without mass {a}")

        #a = 3710 * ((self.star_nu_max)**(-0.613)) * ((self.star_mass)**(-0.26)) #with mass term 
        #print(f"a with mass {a}")

        b_1 = 0.317 * (self.star_nu_max)**(0.970)
        b_2 = 0.948 * (self.star_nu_max)**(0.992)
        print(f"b_1  {b_1}")
        print(f"b_2  {b_2}")


        
        """
        print("Theory calculated values")
        print(f"star granulation sigma {star_granulation_sigma}, star granulation tau {1 / (2 * np.pi * b_1 * 10**-6)}")        
        print(f"star facule sigma {star_granulation_sigma}, star facule tau {1 / (2 * np.pi * b_2 * 10**-6)}") 
        """
        
        return a, b_1, b_2
        
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
    
    
        
    
    

    
    #Start of oscillation functions 
    
    def calc_powder_density(self):
        """
        Calculates the power density component due to the oscillations. 
        Main function for oscillations
        

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
        
        #I have no actual idea how D relates to the splitting of the frequencies.
        #I have chucked in the relationship between delta nu_02 and delta nu.
        #In the asymptotic limit, dnu03 = 5/3 dnu02, which i have reflected here (Chaplin and Miglio 2013).
        #I'm gonna die
        
        # linear fit is from the Kepler LEGACY stars scatter plot, with some random Gaussian noise added in (parameters from star data).
        # Keen et al 2014 gives the second term (the frequency splitting term) to be simply the small frequency separation, hence what i have attempted here,
        # though they add it instead of subtracting it, i tested this and it made everything worse, though that may be due to the random element in fit
        
        fit = (0.05200 * self.star_delta_nu) + np.random.normal(1.42683, 1.41997)
        # fit = (0.05200 * self.star_delta_nu) + 1.42683
        # fit = 3
        
        if l % 2 == 0:
        
             nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * l * (l+1)
            
        elif l % 2 == 1:
            nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * (5/3) * l * (l+1)
            # nu_nl = self.star_delta_nu * ( n+(l/2) + self.epsilon) - fit * l * (l+1)
        
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
    
        
    
    

def calc_power_amplitude(amp,width):
    
    x = (2/np.pi)*((amp**2)/width)
    return x 
        
def calc_lorentz(freq, centroid, FWHM):
    
    x = ((FWHM/2)**2 / ((freq - centroid)**2 + (FWHM/2)**2)) #lorentz curve, !must be multipled by the amplitude!
    return x 

def plot_multi_component_psd(freq_powerspectrum_uHz,  granulation_component, facule_component,  star_nu_max, star_name,  practical_granulation,practical_facule, ylim):
    """
    Plots power specturm of multi component model. Used to compared theoretical sigma models and measured models.  

    Returns
    -------
    None.

    """
    
    total_background_practical = practical_facule + practical_granulation
    total_background = facule_component + granulation_component
    
    
    plt.figure()
    
    plt.semilogy(freq_powerspectrum_uHz, total_background, label='Total theory')
    plt.semilogy(freq_powerspectrum_uHz, facule_component, label='Facule theory')
    plt.semilogy(freq_powerspectrum_uHz, granulation_component, label='Granulation theory')

    plt.semilogy(freq_powerspectrum_uHz, practical_granulation, label='Gran practical')
    plt.semilogy(freq_powerspectrum_uHz, practical_facule, label='Facule practicaFl')
    plt.semilogy(freq_powerspectrum_uHz, total_background_practical, label='Total practical')

    
    plt.axvline(star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
    plt.title(f"Background spectrum due to granulation and facule for star {star_name}, ylim = {ylim}")
    
    plt.ylim(bottom=ylim)
    plt.xlim(left=10, right = 2700)
    
    plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
    plt.ylabel(r"Granulation power, ppm^2 / µHz")
    
    
    
    plt.legend()
    plt.show()
    
def plot_one_component_psd(freq_powerspectrum_uHz,  granulation_component, star_nu_max, star_name):
    """
    Plots models that only have one component

    Parameters
    ----------
    freq_powerspectrum_uHz : TYPE
        DESCRIPTION.
    granulation_component : TYPE
        DESCRIPTION.
    star_nu_max : TYPE
        DESCRIPTION.
    star_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plt.semilogy(freq_powerspectrum_uHz, granulation_component, label='Granulation background component ')
    
def plot_two_component_psd(freq_powerspectrum_uHz,  granulation_component, facule_component, oscillation_component, star_nu_max, star_name, model):
    """
    Plot models 

    Parameters
    ----------
    freq_powerspectrum_uHz : TYPE
        DESCRIPTION.
    granulation_component : TYPE
        DESCRIPTION.
    facule_component : TYPE
        DESCRIPTION.
    star_nu_max : TYPE
        DESCRIPTION.
    star_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    total_background = facule_component + granulation_component + oscillation_component

    
    plt.figure()
    
    
    plt.loglog(freq_powerspectrum_uHz, total_background, label='Total theory')
    plt.loglog(freq_powerspectrum_uHz, facule_component, label='Facule theory')
    plt.loglog(freq_powerspectrum_uHz, granulation_component, label='Granulation theory')
    plt.loglog(freq_powerspectrum_uHz, oscillation_component, label = 'Oscillations')
    plt.title(f"{model}  of a two component background of star {star_name}")
    plt.axvline(star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
    plt.xlim(right=1000)
    plt.ylim(0.1)

    
    plt.legend()

    plt.show()
    
    
#Sun constants    
sun_nu_max = 3090
sun_teff = 5772.0


#All values taken from  Karoff 2013. Comparison of karoff model with observed values and scaling vaalues
sun_granulation_tau, sun_granulation_sigma, sun_facule_tau, sun_facule_sigma = 214, 62.4, 65.8, 50.1

"""
star_mass,  star_radius, star_teff   =  1.01, 1.15, 5416
star_granultion_sigma_practical, star_granulation_tau_practical, star_facule_sigma_practical, star_facule_tau_practical = 62.8, 280.8, 76.5, 66.
star_name = "KIC 6603624"

#compaes karoff model with theoreitcal scaling realtionship sigma values to sigma values from karoff 2013 paper 
star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name, sun_facule_tau , sun_facule_sigma)
granulation_component_practical, facule_component_practical = star.defined_sigma_tau(star_granultion_sigma_practical, star_granulation_tau_practical, star_facule_sigma_practical, star_facule_tau_practical, model="karoff")
granulation_component_theory, facule_component_theory = star.theoretical_sigma_tau(model="karoff")
"""

"""
Values taken from kepler input catalogue 
Kallinger model testing graphs.
doesn't require mass, hence = 1'

star_mass,  star_radius, star_teff   =  1, 11.4, 4730
star_name = "KIC 7949599"
"""

star_mass,  star_radius, star_teff   =  1, 13.6, 4577
star_name = "KIC 5091962"


star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)
granulation_component, facule_component = star.theoretical_sigma_tau(model = "kallinger")
oscillation_component = star.calc_powder_density()
plot_two_component_psd(star.freq_powerspectrum_uHz, granulation_component, facule_component, oscillation_component,  star.star_nu_max, star_name,  model = "kallinger")

star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)
granulation_component, facule_component = star2.theoretical_sigma_tau(model = "karoff")
oscillation_component = star2.calc_powder_density()
plot_two_component_psd(star2.freq_powerspectrum_uHz, granulation_component, facule_component, oscillation_component,  star2.star_nu_max, star_name,  model = "karoff")



"""
ylim = 1
#plot_multi_component_psd(star.freq_powerspectrum_uHz, granulation_component_theory, facule_component_theory, star.star_nu_max, star_name, granulation_component_practical, facule_component_practical, ylim)

star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau, sun_granulation_sigma)
granulation_component = star2.theoretical_sigma_tau(model = 'ball')


sun_granulation_tau = 214.3
sun_granulation_sigma = 62.4

sun_facule_tau = 65.8
sun_facule_sigma = 50.1

star2 = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma,  star_name , sun_facule_tau , sun_facule_sigma )
star2.multi_component()     

        

     
star_mass, star_radius, star_teff = 1.223, 1.357, 6325
star_name = "Kepler 410"
sun_nu_max = 3090
sun_teff = 5772.0
sun_granulation_tau = 214
sun_granulation_sigma = 63




star = power_spectrum(star_mass, star_radius, star_teff, sun_nu_max, sun_teff, sun_granulation_tau,  sun_granulation_sigma, star_name)

muh_components = star.multi_component_scalable()

total_background = muh_components[0] + muh_components[1] + muh_components[2]

#total_background, granulation_component, facule_component, supergranulation_component = star.multi_component_scalable()
oscillation_psd = star.calc_powder_density()
total_psd = total_background + oscillation_psd

#plots

plt.loglog(star.freq_powerspectrum_uHz, total_psd, color = 'blue', label = "power spectrum")
plt.axvline(star.star_nu_max, ls = '--', color = 'r', label = "nu_max")
plt.grid(which='major')
#plt.xlim(500, 5000)
plt.title("Power Spectrum - Granulation and Oscillations")
plt.ylabel("Power (ppm^2 Hz^-1)")
plt.xlabel("Frequency (uHz)")
plt.legend()
plt.show()


plt.figure()
#plt.loglog(star.freq_powerspectrum_uHz,total_background, label = 'Total ')
plt.loglog(star.freq_powerspectrum_uHz, muh_components[2], label = 'Supergranulation component')
plt.loglog(star.freq_powerspectrum_uHz, muh_components[1], label = 'Facule component')
plt.loglog(star.freq_powerspectrum_uHz, muh_components[0], label = 'Granulation component')
plt.axvline(star.star_nu_max, linestyle="--", label=r"$\nu_{\max}$")
plt.ylim(bottom=0.1)
plt.xlim(left=10)
plt.xlabel(r"Frequency $\nu$ ($\mu$Hz)")
plt.ylabel(r"Granulation power, ppm^2 / µHz")
plt.legend()
plt.tight_layout()
plt.show()
"""



