# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 00:37:02 2025

@author: maxto
"""

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


class angularEventGenerator:
    """
    Generates cos(theta) values acording to the differential cross-section for the reaction. 
    Within a specified [cosMin,cosMax] using rejection sampling
        
    """

    def __init__(self,a1=0,a2=0,cosMin=-1,cosMax=1,nSamples = 1000):
        """
        
        Parameters
        ----------
        a1 : float, optional
            a2 coefficent from equal. The default is 0.
        a2 : float, optional
            a2 coefficent from equal. The default is 0.
        cosMin : float, optional
            Min acceptance angle. The default is -1.
        cosMax : float, optional
            Max acceptance angle. The default is 1.
        nSamples : Integer, optional
            Number of events to be generated. The default is 1000.

        Raises
        ------
        ValueError
            Checks accepetance angle interval to ensure validity.

        """
        
        self.a1 = float(a1)
        self.a2 = float(a2)
        
        self.cosMin = float(cosMin)
        self.cosMax = float(cosMax)
        
        self.checkInterval()
        self.ensurePositive()
        
        self.nSamples = int(nSamples)
        self.pmax = self.getMax()
    
            
    def probF(self,x):
        """
        Value of the function F = 1 + a1x + a2x^2, at x 

        Parameters
        ----------
        x : integer
            Point to evaluate the function.

        Returns
        -------
        value : float
            Value of the function .

        """
        value = 1.0 + self.a1 * x + self.a2 * x**2
        return value

    def ensurePositive(self):
        """
        checks that p(x) > 0 for all values in the interval

        Raises
        ------
        ValueError
            p(x) is less than 0 on the interval.

        """
        
        xValues = np.linspace(self.cosMin, self.cosMax,1000)
        probFValues = self.probF(xValues) #array of probF values evaluated at xValues 
        
        if np.any(probFValues<0): #Checks all values of probF on the interval, raises error if any value is less than 0 
            raise ValueError(f"PDF became negative in the chosen intervral [{self.cosMin}, {self.cosMax}]. Different a1, a2 values required")
        
    def checkInterval(self):
        """
        Checks the accepetance interval of the class

        Raises
        ------
        ValueError
            Depending on what is wrong with the interval.

        """
        
        if self.cosMin < -1:
            raise ValueError(f"cosMin {self.cosMin} cannot be less than -1")

        if self.cosMax > 1:
            raise ValueError(f"cosMax {self.cosMax} cannot be greater than 1")
        
        if self.cosMin > self.cosMax:
            raise ValueError(f"cosMin {self.cosMin} must be less than cosMax {self.cosMax}")
        
    

        
    def getMax(self):
        """
        Finds the max value of the probability function (pMax) 



        Returns
        -------
        pMax : float
            Maximum value.

        """
        
        xValues = np.linspace(self.cosMin, self.cosMax,1000)
        probFValues = self.probF(xValues) #array of probF values evaluated at xValues 
        
        pMax = np.max(probFValues)
        
        return pMax
      
    
    def rejectionSamplingGenerator(self):
        """Applies rejection sampling to generated nOn points on the range defined by [cosMin, cosMax].

        Parameters
        ----------
        nEvents : int
            Number of events to be generated

        Returns
        -------
        returnArray : array
        Array with nEvents accepeted samples
        """
        
        nEvents = self.nSamples #Temp assignment
        
        rng = np.random.default_rng()
        returnList = []
        
        """Rejection sampling method"""
        while len(returnList) < nEvents: #Loops until accepted values is greater than number of events to generate
            x = rng.uniform(self.cosMin, self.cosMax, size=1000)  #
            c = rng.uniform(0.0, self.pmax, size=1000)
            f = self.probF(x) # Value of function at x
            
            keep = x[f > c]
            returnList.extend(keep)
            
        returnArray = np.asarray(returnList[:nEvents])

        return returnArray
    
    def convertToTheta(self,xValues):
        """
        converts input array of cos(theta) values to an array of theta values

        Parameters
        ----------
        xValues : array
            Array of cos(theta) events.

        Returns
        -------
        returnList : array
            Array of theta events.

        """
        
        xVal= np.asarray(xValues, dtype=float) #Converts input data into correct form. A numpy array with float data types
        returnList = np.arccos(np.clip(xVal, -1, 1))  #fixes bug where x values would be slightly outside of [-1,1], eg 1.00000002. This just changes values outside of the interval to -1 or 1 
        
        return  returnList
    
    def addResolutionLimitToData(self, xArray, resolution): #Step 3, resolution limitation 
        """
        Adds the resolution limitation to the generated data

        Parameters
        ----------
        xArray : array
            Events without resolution affect applied.
        resolution : float
            resolution limit to apply to events.

        Returns
        -------
        xSmeared : array
            Events with resolution affect applied.

        """
       

        rng = np.random.default_rng()
        xArray = np.asarray(xArray, dtype=float)
        
        thetaTrue = self.convertToTheta(xArray) #Converts x events to theta 
        
        resolutionUncertainy = rng.normal(loc = 0, scale = resolution, size = thetaTrue.shape) #Random value generated by a gaussian to simulate the resolution uncertainty 
        
        thetaSmeared = thetaTrue + resolutionUncertainy #Adds uncertainy factor from resolution to event
        
        thetaSmeared = np.mod(thetaSmeared, 2*np.pi)
        thetaSmeared = np.where(thetaSmeared > np.pi, 2*np.pi - thetaSmeared, thetaSmeared)
        
        xSmeared = np.cos(thetaSmeared)
        return xSmeared 
    

    #Step 2, method 1
    
    def forwardBackCounter(self,data):
        """Counts the number of forward and backward events in the data array.
        Forward event, cos(theta)>0
        Backwardd event, cos(theta)<0
        
        Parameters
        ----------
        data : array
            An array of cos(theta) (events)  generated by rejection sampling

        Returns
        -------
        NF : int
            The number of forward events 
        NB : int
            The number of backward events 
        AFB : float
            Forward back asymmetry 
        
        """
     
        data = np.asarray(data, dtype=float) #Converts input data into correct form. A numpy array with float data types
        
        NF = np.count_nonzero(data > 0.0) #Number of forward events (N+)
        NB = np.count_nonzero(data < 0.0) #Number of backward events (N-)
        
        AFB = (NF-NB)/(NF+NB) #Formula for forward back asymmetry  
        
        return NF, NB, AFB
     
    def a1PerfectAccepetance(self, AFB):
        """
        Calculates a1 from AFB for the full accepetance range of cos(theta) [-1,1] 
        
        
        Parameters
        ----------
        AFB : int
            Forward back asymmetry .

        Returns
        -------
        a1 : float
            Differential cross section coefficent .

        """
        
        a2 = self.a2
        a1 = 2 * AFB * (1+(a2)/3)

        return a1
    
    
    def a1Symmetric(self, AFB):
        """
        Calcultes a1 on a symmetric limited acceptance interval
        
        Show algerbra for rearranging to c 
        

        Parameters
        ----------
        AFB : Float
            AFB Coefficent from formula.

        Returns
        -------
        a1 : float
            Coefficent from  differential cross section formula.

        """
 
        c =self.cosMax
        a2 = self.a2
        
        a1 = AFB * 2 * (1/c) * (1 + ((a2)/3)*c**2)  
        
        return a1
    
    def sigmaA1Predicted(self, AFB, N):
        """
        For a given AFB and number of events, calculates the theoretical standard deviation
        
        Show algebraa
        
        Parameters
        ----------
        AFB : Float
            AFB Coefficent from formula.
        N : integer
            Number of events for in the data set for the AFB value.

        Returns
        -------
        sigmaA1 : float
            Predicted standard deviation for a1.

        """
            
        k = (2.0) * (1.0 + (self.a2)/3.0)
        sigmaAFB = np.sqrt((1-AFB**2)/N)
        
        sigmaA1 = k*sigmaAFB
        
        return sigmaA1
            
            
        
    def a1AFB(self,data): 
        """
        Calculates a1 using a method dependent on the accepetance angle. 
        
        If the cos theta range is pefect ([-1,1]), then a1PerfectAccepetance() is used and a1 predicted is returned
        If the cos theta range isn't pefect ([-P,P]), then a1Limited() is used

        Parameters
        ----------
        data : array
            An array of cos(theta) events


        Returns
        -------
        a1 : float
            Coefficent from  differential cross section formula.
        sigmaPredicted : float, for perfect acceptance
            Theoretical standard deviation 
            
        """
        
        NF, NB, AFB = self.forwardBackCounter(data)
        N = NF + NB #Can't use N = self.nSamples (resolution investigation clipping)
        
        if (self.cosMin==-1 and self.cosMax==1): #perfect acceptance
            a1 = self.a1PerfectAccepetance(AFB)
            sigmaPredicted = self.sigmaA1Predicted(AFB,N)
            
            return a1, sigmaPredicted
        
        elif -self.cosMin == self.cosMax: #symmetric but not perfect
          
            a1 = self.a1Symmetric(AFB)
            sigmaPredicted = 0
            
        return a1, sigmaPredicted
    
    #method 2
    
    def binnedFitA1ChangingWidth(self,data, returnFitInfo=False):
        """
        Binned fitting routine.   

        Parameters
        ----------
        data : array
            An array of cos(theta) events
        returnFitInfo : Boolean, default false
            Bug fixing variable. False then only return a1
            
        
        Returns
        -------
        a1 : float
            a1 coefficent from binned fitting 
        a2 : float
            a2 coefficent from binned fitting
        sigmaA1Predicted : float
            estimated error on a1 from binned fitting
        sigmaA2Predicted : float
            estimated error on a2 from binned fitting
        chi2Val  : float
            chi squared value of fit
        dof : int
            degrees of freedom from fit

        """
        
        """Bounds for a1, a2 parameters"""
        a1Lower = -10
        a1Upper = 10
        
        a2Lower = -3
        a2Upper = 5
        
        p0 = [0,0] #Inital guess for a1 fit
        #bounds = ([a1Lower,a2Lower],[a1Upper,a2Upper]) #Bounds for a1 a2 fit 


        data = np.asarray(data, dtype=float) #Converts input data into correct form. A numpy array with float data types
        N = data.size #same as current sampleSize in for loop 
        
        
        
        maxBins = int(round(10*N**(1/3)))
        minBins = 20
        targetCountsPerBin=50
        
        #Decides on the number of bins
        nBins = int(np.floor(N/float(targetCountsPerBin)))
        nBins = int(np.clip(nBins,minBins,maxBins))
            
        
        #histogram of data to be fit 
        counts, binEdges = np.histogram(data,bins=nBins,range=(self.cosMin,self.cosMax))
        counts = counts.astype(float)
        
        

        binCentres = (binEdges[:-1]+binEdges[1:])/2
        
        #print(f"changing bin width for {N} samples {actualBinWidth}")
        
        def model(_, a1, a2): #method doesn't use bin centres,  calculates everything from bin edges but curve_fit requires 3argument, hence _
           p = Polynomial([1.0, a1, a2])
           Pint = p.integ()
           Z = Pint(self.cosMax) - Pint(self.cosMin)  # normalization over full range
           # exact expected counts in each bin=
           mu = N * (Pint(binEdges[1:]) - Pint(binEdges[:-1])) / Z
           return mu
        
          
        sigma = np.sqrt(np.maximum(counts, 1.0))
        
   
        popt, pcov = curve_fit(model, binCentres, counts,sigma = sigma, absolute_sigma=True, maxfev=10000 ) 
        
        a1 = float(popt[0]) #a1 fit value from matrix
        a2 = float(popt[1])
        
        sigmaA1Predicted = np.sqrt(pcov[0,0])
        sigmaA2Predicted = np.sqrt(pcov[1,1])

        #chi^2 calculator
        muBest = model(binCentres, a1, a2)
        chi2Val = np.sum((counts - muBest) ** 2 / (sigma**2))
        dof = nBins - 2 - 1
        
       
        
        
        if returnFitInfo:
            return a1, a2, sigmaA1Predicted, sigmaA2Predicted, chi2Val, dof
        else:
            return a1    #test return for fixing a bug
    
        
    
        

    
def plotTest(data,N):
    
    """Test utility function to plot distribution of generated events"""
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(data, bins=60, range=(-1,1), alpha=0.6, label="samples")
    plt.xlabel("x = cos(theta)")
    plt.ylabel("Counts per bin")
    plt.title(f"dσ/d(cosθ) ∝ 1 + a1 x + a2 x²  n={N},)")
    plt.tight_layout()
    plt.show()
    


def accuracyCalculator(aArray, aReal):
    """
    Calculates accuracy statistics of an array of a_n values (aArray) compared to the real a_n value  (aReal)
    Calculates:
    - the mean value. 
    - mean percentage error compared to aReal
    - statistical bias 
    - standard deviation 
    - root mean squared error 

    Parameters
    ----------
    aArray : Array
        Array of a_n coefficents. (a1 or a2 from fitting)
    aReal : Float
        Real predefined value of a_n. (real value of a1, a2 used to generate the original event dataa)
.

    Returns
    -------
    meanA : float
        Mean a_n value from input.
    meanPercentageError : float
        Absolute mean percentage error of a_n.
    stdA : float
        standard deviation of a_n.
    biasA : float
        Bias of the method: meanA - aReal.
    rmse : float
        Root mean squared error: sqrt(standardDeivation^2 + bias^2)

    """
    aArray = np.asarray(aArray, dtype=float)
    
    meanA = np.mean(aArray)
    meanPercentageError = np.mean(np.abs((aArray - aReal) / aReal * 100.0)) # mean percentage error
    stdA = np.std(aArray, ddof=1) #standard deviation
    biasA = meanA - aReal #bias
    rmse = float(np.hypot(stdA,biasA)) # sqrt(std^2 + bias^2)
    
    return meanA, meanPercentageError, stdA, biasA, rmse

def errorCalculator(predictedSigma, aArray, aReal):
    """
    For a given number of events and repeated runs of fitting, calculates accuracy statisitcs for an array of a_n coefficents with a corresponding predicted sigma.
    For 1 fitting run, pull = (aFitted - aReal) / predictedSigma
    pullValue, array of pull values from each run
    
    Parameters
    ----------
    predictedSigma : array
        Array of predicted sigma from each fitting run.
    aArray : Array
        Array of a_n coefficents. (a1 or a2 from fitting)
    aReal : Float
        Real predefined value of a_n. (real value of a1, a2 used to generate the original event dataa)

    Returns
    -------
    meanSigma : float
        Mean of the predicted sigmas 
    pullMean : float
        Mean of the per run pull values.
    pullStd : float
        Standaaard deviation of thee pull values.

    """
    
    predictedSigma = np.asarray(predictedSigma, dtype=float) #Converts data to numpy array with float data type
    aArray   = np.asarray(aArray,   dtype=float)
    
    meanSigma = np.mean(predictedSigma) #Mean value of all the predicted standard deviation 
    pullValue = (aArray-aReal)/predictedSigma #Array of each runs pull value
    pullMean = np.mean(pullValue) 
    pullStd = np.std(pullValue) #standard deviation of pull values
    
    return meanSigma, pullMean, pullStd



def affectOfN(sampleSizes, a1Real, a2Real, cosMin, cosMax, nRepeats):
    """
    Studies the affect of the number of generated events on the accuracy of  a1,a2 
    

    Parameters
    ----------
    sampleSizes : array
        List of different number of events to test.
    a1Real : float
        Value of a1 used to generate data.
    a2Real : float
        Value of a2 used to generate data..
    cosMin : float
        min acceptance angle.
    cosMax : float
        max acceptance angle.
    nRepeats : integeer
        number of repeated tests per value in samplesSize

    Returns
    -------
    
    """
 
    
    method1Mean = [] #
    method1MeanPercentageErrorRepeats = []
    method1Std = []
    method1Bias = []
    method2Chi2OverNdf = []
    
    method2MeanA1 = []
    method2MeanPercentageErrorRepeatsA1 = []
    method2StdA1 = []
    method2BiasA1 = []
    
    method2MeanA2 = []
    method2MeanPercentageErrorRepeatsA2 = []
    method2StdA2 = []
    method2BiasA2 = []

    method1SigmaPredMean = []
    method1PullMean = []
    method1Pullstd = []
    
    method2SigmaPredMeanA1 = []
    method2PullMeanA1 = []
    method2PullstdA1 = []
    
    method2SigmaPredMeanA2 = []
    method2PullMeanA2 = []
    method2PullstdA2 = []
    
    rmseMethod1A1 = []
    rmseMethod2A1 = []
    rmseMethod2A2 = []
    


    for N in sampleSizes: #Repeats for every number of events in the list sampleSizes
    
        """Data storage for values generated using N events generated"""   
    
        m1Vals= [] #list of a1 values generated by afb method 
        m1Sigmas = [] #list of errors on a1 from afb method
        
        m2ValsA1= []    #list of a1 values generated by binned fit method
        m2SigmasA1 = [] #list of estimated errors on a1 generated by fitting routine 
        
        m2ValsA2= [] #list of a2 values generated by binned fit method using 
        m2SigmasA2 = [] #list of estimated errors on a2 generated by fitting routine 
        
        m2Chi2Vals = [] #chi^2 value for each fit
        m2DofVals = [] #Degrees of freedom for each fit
        

        for _ in range(nRepeats): #repeats finding a1, a2 from both methods nRepeats time
        
            gen = angularEventGenerator(a1=a1Real, a2=a2Real, cosMin=cosMin, cosMax=cosMax,nSamples=N) #initalise class with these values parameters
            data = gen.rejectionSamplingGenerator() #generates data according to class values. Different N values


            a1M1, sigmaM1Predicted = gen.a1AFB(data)  #a1 value and error from afb method
            m1Vals.append(a1M1)
            m1Sigmas.append(sigmaM1Predicted)
            
            a1M2, a2M2, sigmaA1M2Predicted, sigmaA2M2Predicted,  chi2M2, dofM2 = gen.binnedFitA1ChangingWidth(data, returnFitInfo=True) 
            #a1,a2, estimated errors on both, chi^2, degrees of  freedom from binned fitting method
            
            
            m2SigmasA1.append(sigmaA1M2Predicted)
            m2ValsA1.append(a1M2)
            
            m2SigmasA2.append(sigmaA2M2Predicted)
            m2ValsA2.append(a2M2)
                 
            m2Chi2Vals.append(chi2M2)
            m2DofVals.append(dofM2)
            
        """Calculates and stores stats of the nRepeats repeated runs. Runs for every value in sampleSizes"""
        
       

        """Method 1 AFB values for a1 Parameter. """
        
        m1MeanN, m1PctErrN, m1StdN, m1BiasN, rmseM1A1 = accuracyCalculator(m1Vals, a1Real) #Calculates a1 accuracy values
       
        
        rmseMethod1A1.append(rmseM1A1)
        method1Mean.append(m1MeanN)
        method1MeanPercentageErrorRepeats.append(m1PctErrN)
        method1Std.append(m1StdN)
        method1Bias.append(m1BiasN)
        
        meanSigma1, pullMean1, pullStd1 = errorCalculator(m1Sigmas, m1Vals, a1Real)
        method1SigmaPredMean.append(meanSigma1)
        method1PullMean.append(pullMean1)
        method1Pullstd.append(pullStd1)
        
        """Method 2 binned fitting values for a1 Parameter"""
        
        m2MeanNA1, m2PctErrNA1, m2StdNA1, m2BiasNA1, rmseM2A1 = accuracyCalculator(m2ValsA1, a1Real)
           

        rmseMethod2A1.append(rmseM2A1)
        method2MeanA1.append(m2MeanNA1)
        method2MeanPercentageErrorRepeatsA1.append(m2PctErrNA1)
        method2StdA1.append(m2StdNA1)
        method2BiasA1.append(m2BiasNA1)
        
        meanSigma2A1, pullMean2A1, pullStd2A1 = errorCalculator(m2SigmasA1, m2ValsA1, a1Real)
        method2SigmaPredMeanA1.append(meanSigma2A1)
        method2PullMeanA1.append(pullMean2A1)
        method2PullstdA1.append(pullStd2A1)
        
        """Method 2 binned fitting values for a2 Parameter"""
        
        m2MeanNA2, m2PctErrNA2, m2StdNA2, m2BiasNA2, rmseM2A2 = accuracyCalculator(m2ValsA2, a2Real)

        rmseMethod2A2.append(rmseM2A2)
        method2MeanA2.append(m2MeanNA2)
        method2MeanPercentageErrorRepeatsA2.append(m2PctErrNA2)
        method2StdA2.append(m2StdNA2)
        method2BiasA2.append(m2BiasNA2)
        
        meanSigma2A2, pullMean2A2, pullStd2A2 = errorCalculator(m2SigmasA2, m2ValsA2, a2Real)
        method2SigmaPredMeanA2.append(meanSigma2A2)
        method2PullMeanA2.append(pullMean2A2)
        method2PullstdA2.append(pullStd2A2)
        

        m2Chi2Vals = np.asarray(m2Chi2Vals, dtype=float)
        m2DofVals = np.asarray(m2DofVals, dtype=float)
        
        
        

        """
        Prints information from N event run
        """
            
        print(f"[Number of events] N={N}, with [Number of repeats] M ={nRepeats}")
        
        
        print(f"  Method 1 (a1): mean={m1MeanN:.5f}, bias={m1BiasN:.5f}, %err={m1PctErrN:.2f}, "
              f"std={m1StdN:.5f}, ⟨σpred⟩={meanSigma1:.5f}, pull μ={pullMean1:.5f}, pull σ={pullStd1:.3f}"
              f" RMSE = {rmseM1A1:.5f}")
        
        # Method 2 (fit, a1)
        print(f"  Method 2 (a1): mean={m2MeanNA1:.5f}, bias={m2BiasNA1:.5f}, %err={m2PctErrNA1:.2f}, "
              f"std={m2StdNA1:.5f}, ⟨σpred⟩={meanSigma2A1:.5f}, pull μ={pullMean2A1:.5f}, "
              f"pull σ={pullStd2A1:.3f}, <chi2/ndf>={method2Chi2OverNdf[-1]:.2f}, RMSE = {rmseM2A1:.5f}")
        
        # Method 2 (fit, a2)
        print(f"  Method 2 (a2): mean={m2MeanNA2:.5f}, bias={m2BiasNA2:.5f}, %err={m2PctErrNA2:.2f}, "
              f"std={m2StdNA2:.5f}, ⟨σpred⟩={meanSigma2A2:.5f}, pull μ={pullMean2A2:.5f}, "
              f"pull σ={pullStd2A2:.3f}, RMSE = {rmseM2A2:.5f}")
        
    

            
        
        
    Ngrid = np.array(sampleSizes, dtype=float) #x axis
    
    Ngrid     = np.array(sampleSizes, dtype=float)
    invSqrtN = 1.0 / np.sqrt(Ngrid)

    # Normalisation so the guide roughly goes through the first point
    krmsea1 = rmseMethod1A1[0] / invSqrtN[0]
    krmsea2 = rmseMethod2A2[0] / invSqrtN[0]
    
    
    # --- Plot 1: RMSE(a1), both methods ---
    plt.figure()
    plt.plot(Ngrid, rmseMethod1A1, "o-", label="Method 1 RMSE(a1)")
    plt.plot(Ngrid, rmseMethod2A1, "s-", label="Method 2 RMSE(a1)")
    plt.plot(Ngrid, krmsea1 * invSqrtN, "k--", label=" 1/√N")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("RMSE(a1)")
    plt.title("RMSE(a1) vs N with 1/√N trend")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: RMSE(a2), Method 2 ---
    plt.figure()
    plt.plot(Ngrid, rmseMethod2A2, "d-", label="Method 2 RMSE(a2)")
    plt.plot(Ngrid, krmsea1 * invSqrtN, "k--", label=" 1/√N")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("RMSE(a2)")
    plt.title("RMSE(a2) vs N with 1/√N trend (Method 2)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


    kstda1 = method1Std[0] / invSqrtN[0]
    kstda2 = method2StdA2[0] / invSqrtN[0]

    plt.figure()
    plt.plot(Ngrid, method1Std,   "o-", label="Method 1 σ_emp(a1)")
    plt.plot(Ngrid, method2StdA1, "s-", label="Method 2 σ_emp(a1)")
    plt.plot(Ngrid, kstda1 * invSqrtN, "k--", label=" 1/√N")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("Empirical std σ(a1)")
    plt.title("Empirical σ(a1) vs number of events ")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Ngrid, method2StdA2, "d-", label="Method 2 σ_emp(a2)")
    plt.plot(Ngrid, kstda2 * invSqrtN, "k--", label="1/√N")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("Empirical std σ(a2)")
    plt.title("Empirical σ(a2) vs number of events (Method 2)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    """1 / sqrt(n) function code"""
    
    Nref = Ngrid[0]
    k = method1Std[0] * np.sqrt(Nref)        
    refcurve = k / np.sqrt(Ngrid)

    



    
    plt.figure()
    plt.plot(Ngrid, method1Pullstd, "o-", label="Method 1 (AFB) pull width")
    plt.plot(Ngrid, method2PullstdA1, "s-", label="Method 2 (fit) pull width (a1)")
    plt.plot(Ngrid, method2PullstdA2, "s-", label="Method 2 (fit) pull width (a2)")
    plt.axhline(1.0, linestyle="--", color="k", label="ideal = 1")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("Pull std for a1")
    plt.title("Pull width vs N (a1)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
    
    # Useful extras: fit quality vs N
    plt.figure()
    plt.plot(Ngrid, method2Chi2OverNdf, "o-", label="Method 2 ⟨χ²/ndf⟩")
    plt.axhline(1.0, linestyle="--", color="k", label="ideal = 1")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("⟨χ²/ndf⟩")
    plt.title("Fit quality vs N (Method 2)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
    plt.figure()
    plt.plot(Ngrid, method1SigmaPredMean, "o--", label="Method 1 ⟨σ_pred(a1)⟩")
    plt.plot(Ngrid, method1Std, "o-", label="Method 1 empirical std(a1)")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("Uncertainty on a1")
    plt.title("Predicted vs empirical σ(a1) vs N (method1)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    plt.figure()
    plt.plot(Ngrid, method2SigmaPredMeanA1, "s--", label="Method 2 ⟨σ_pred(a1)⟩")
    plt.plot(Ngrid, method2StdA1, "s-", label="Method 2 empirical std(a1)")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("Uncertainty on a1")
    plt.title("Predicted and empirical σ(a1) vs N (method2)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    # --- Predicted vs empirical σ: a2, Method 2 (fit) ---
    plt.figure()
    plt.plot(Ngrid, method2SigmaPredMeanA2, "d--", label="predicted σ(a2)")
    plt.plot(Ngrid, method2StdA2,          "d-",  label="empirical σ(a2)")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("σ(a2)")
    plt.title("Predicted and empirical σ(a2) vs N (method2)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
    
    
    plt.figure()
    plt.plot(Ngrid, method1Bias, "o-", label="method 1 a1 Bias")
    plt.plot(Ngrid, method2BiasA2, "o--", label="method 2 A2 Bias")
    plt.plot(Ngrid, method2BiasA1, "o-", label="method 2 A1 Bias")
    plt.xscale("log")
    plt.xlabel("Number of events N (log")
    plt.ylabel("Bias")
    plt.title("method 1 a1 Bias")
    plt.legend()
    
    
    plt.figure()
    plt.plot(Ngrid, method1Std, "d--", label="empircal σ(a1)")
    plt.plot(Ngrid, method2StdA2,          "d-",  label="empirical σ(a2) method 2")
    plt.plot(Ngrid, method2StdA1,          "o-",  label="empirical σ(a1)  method 2")
    plt.plot(Ngrid, refcurve, "--", label="1/sqrt(N)")
    plt.xscale("log")
    plt.xlabel("Number of events N (log)")
    plt.ylabel("σ(a2)")
    plt.title("empirical σ for a1,a2 vs N")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
   
    
    
 

    return 5


    
    


def affectOfResolution(resolutionList, nSamples, a1Real, a2Real, cosMin, cosMax, nRepeats):
    """
    Studies the affect of resolution on the accuracy of a1,a2

    Parameters
    ----------
    resolutionList : array
        List of different resolutions to test.
    nSamples : int
        number of events to generate.
    a1Real : float
        Value of a1 used to generate data.
    a2Real : float
        Value of a2 used to generate data..
    cosMin : float
        min acceptance angle.
    cosMax : float
        max acceptance angle.
    nRepeats : int
        number of repeated tests per value in samplesSize

    Returns
    -------
    

    """
    
    
    method1Mean = [] #
    method1MeanPercentageErrorRepeats = []
    method1Std = []
    method1Bias = []
    method2Chi2OverNdf = []
    
    method2MeanA1 = []
    method2MeanPercentageErrorRepeatsA1 = []
    method2StdA1 = []
    method2BiasA1 = []
    
    method2MeanA2 = []
    method2MeanPercentageErrorRepeatsA2 = []
    method2StdA2 = []
    method2BiasA2 = []

    method1SigmaPredMean = []
    method1PullMean = []
    method1Pullstd = []
    
    method2SigmaPredMeanA1 = []
    method2PullMeanA1 = []
    method2PullstdA1 = []
    
    method2SigmaPredMeanA2 = []
    method2PullMeanA2 = []
    method2PullstdA2 = []
    
    rmseMethod1A1 = []
    rmseMethod2A1 = []
    rmseMethod2A2 = []
    
    
    resolutionValues = []

    for currentResError in resolutionList: #Repeats for every resolution in the list resolutionList
       """Data storage for values generated using currentResError resolution. """   
       
       m1Vals= [] #list of a1 values generated by afb method 
       m1Sigmas = [] #list of errors on a1 from afb method
       
       m2ValsA1= []    #list of a1 values generated by binned fit method
       m2SigmasA1 = [] #list of estimated errors on a1 generated by fitting routine 
       
       m2ValsA2= [] #list of a2 values generated by binned fit method using 
       m2SigmasA2 = [] #list of estimated errors on a2 generated by fitting routine 
       
       m2Chi2Vals = [] #chi^2 value for each fit
       m2DofVals = [] #Degrees of freedom for each fit
    
       for _ in range(nRepeats):  #repeats finding a1, a2 from both methods nRepeats time
        
            gen = angularEventGenerator(a1=a1Real, a2=a2Real,cosMin=cosMin, cosMax=cosMax,nSamples=nSamples)     
            xTrue = gen.rejectionSamplingGenerator() #generates data according to class values
            
            data = gen.addResolutionLimitToData(xTrue, currentResError) #adds resolution effect to data
    
            a1M1, sigmaM1Predicted = gen.a1AFB(data) #a1 value and error from afb method
            m1Vals.append(a1M1)
            m1Sigmas.append(sigmaM1Predicted)
            
            a1M2, a2M2, sigmaA1M2Predicted, sigmaA2M2Predicted,  chi2M2, dofM2 = gen.binnedFitA1ChangingWidth(data, returnFitInfo=True)
            #a1,a2, estimated errors on both, chi^2, degrees of freedom from binned fitting method
    
            
            m2SigmasA1.append(sigmaA1M2Predicted)
            m2ValsA1.append(a1M2)
            
            m2SigmasA2.append(sigmaA2M2Predicted)
            m2ValsA2.append(a2M2)
                 
            m2Chi2Vals.append(chi2M2)
            m2DofVals.append(dofM2)
            
       """Calculates and stores stats of the nRepeats repeated runs. Runs for every value in resolutionList""" 
    
    
       """Method 1 AFB values for a1 Parameter"""
        
       m1MeanN, m1PctErrN, m1StdN, m1BiasN, rmseM1A1 = accuracyCalculator(m1Vals, a1Real)
       
        
       rmseMethod1A1.append(rmseM1A1)
       method1Mean.append(m1MeanN)
       method1MeanPercentageErrorRepeats.append(m1PctErrN)
       method1Std.append(m1StdN)
       method1Bias.append(m1BiasN)
        
       meanSigma1, pullMean1, pullStd1 = errorCalculator(m1Sigmas, m1Vals, a1Real)
       method1SigmaPredMean.append(meanSigma1)
       method1PullMean.append(pullMean1)
       method1Pullstd.append(pullStd1)
        
       """Method 2 binned fitting values for a1 Parameter"""
        
       m2MeanNA1, m2PctErrNA1, m2StdNA1, m2BiasNA1, rmseM2A1 = accuracyCalculator(m2ValsA1, a1Real)
           

       rmseMethod2A1.append(rmseM2A1)
       method2MeanA1.append(m2MeanNA1)
       method2MeanPercentageErrorRepeatsA1.append(m2PctErrNA1)
       method2StdA1.append(m2StdNA1)
       method2BiasA1.append(m2BiasNA1)
        
       meanSigma2A1, pullMean2A1, pullStd2A1 = errorCalculator(m2SigmasA1, m2ValsA1, a1Real)
       method2SigmaPredMeanA1.append(meanSigma2A1)
       method2PullMeanA1.append(pullMean2A1)
       method2PullstdA1.append(pullStd2A1)
        
       """Method 2 binned fitting values for a2 Parameter"""
        
       m2MeanNA2, m2PctErrNA2, m2StdNA2, m2BiasNA2, rmseM2A2 = accuracyCalculator(m2ValsA2, a2Real)

       rmseMethod2A2.append(rmseM2A2)
       method2MeanA2.append(m2MeanNA2)
       method2MeanPercentageErrorRepeatsA2.append(m2PctErrNA2)
       method2StdA2.append(m2StdNA2)
       method2BiasA2.append(m2BiasNA2)
        
       meanSigma2A2, pullMean2A2, pullStd2A2 = errorCalculator(m2SigmasA2, m2ValsA2, a2Real)
       method2SigmaPredMeanA2.append(meanSigma2A2)
       method2PullMeanA2.append(pullMean2A2)
       method2PullstdA2.append(pullStd2A2)
       resolutionValues.append(currentResError)
        
       m2Chi2Vals = np.asarray(m2Chi2Vals, dtype=float)
       m2DofVals = np.asarray(m2DofVals, dtype=float)
        
       red_vals        = m2Chi2Vals / m2DofVals
       chi2red_mean    = float(np.mean(red_vals))
       method2Chi2OverNdf.append(chi2red_mean)
        
      
       
        
       print(f"[res-scan] σθ={currentResError:.4f} rad, N={nSamples}, acc=[{cosMin},{cosMax}],  a1 real = {a1Real}, a2 real = {a2Real} ")
      
       """
       
       print(f"method 1 a1 rmse {rmseM1A1:.5f}")
       print(f"method 2 a1 rmse {rmseM2A1:.5f}")
       print(f"method 2 a2 rmse {rmseM2A2:.5f}")
       
       """
       

       

       print(f"  Method 1 (a1): mean={m1MeanN:.5f}, bias={m1BiasN:.5f}, %err={m1PctErrN:.2f}, "
              f"std={m1StdN:.5f}, ⟨σpred⟩={meanSigma1:.5f}, pull μ={pullMean1:.5f}, pull σ={pullStd1:.3f}"
              f" RMSE = {rmseM1A1:.5f}")
        
        # Method 2 (fit, a1)
       print(f"  Method 2 (a1): mean={m2MeanNA1:.5f}, bias={m2BiasNA1:.5f}, %err={m2PctErrNA1:.2f}, "
              f"std={m2StdNA1:.5f}, ⟨σpred⟩={meanSigma2A1:.5f}, pull μ={pullMean2A1:.5f}, "
              f"pull σ={pullStd2A1:.3f}, <chi2/ndf>={method2Chi2OverNdf[-1]:.2f}, RMSE = {rmseM2A1:.5f}")
        
        # Method 2 (fit, a2)
       print(f"  Method 2 (a2): mean={m2MeanNA2:.5f}, bias={m2BiasNA2:.5f}, %err={m2PctErrNA2:.2f}, "
              f"std={m2StdNA2:.5f}, ⟨σpred⟩={meanSigma2A2:.5f}, pull μ={pullMean2A2:.5f}, "
              f"pull σ={pullStd2A2:.3f}, RMSE = {rmseM2A2:.5f}")
        
     
       
       
    
    Ngrid = np.array(resolutionValues, dtype=float)
    
    plt.figure()
    plt.plot(Ngrid, rmseMethod1A1, "o-", label="Method 1 RMSE(a1)")
    plt.plot(Ngrid, rmseMethod2A1, "s-", label="Method 2 RMSE(a1)")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("RMSE(a1)")
    plt.title("RMSE(a1) vs resolution ")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Ngrid, rmseMethod2A2, "d-", label="Method 2 RMSE(a2)")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("RMSE(a2)")
    plt.title("RMSE(a2) vs resolution  (Method 2)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(Ngrid, method2Chi2OverNdf, "o-", label="Method 2 ⟨χ²/ndf⟩")
    plt.axhline(1.0, linestyle="--", color="k", label="ideal = 1")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("⟨χ²/ndf⟩")
    plt.title("Fit quality vs vs resolution   (Method 2)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    plt.figure()
    plt.plot(Ngrid, method1Std,   "o-", label="Method 1 σ_emp(a1)")
    plt.plot(Ngrid, method2StdA1, "s-", label="Method 2 σ_emp(a1)")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("Empirical std σ(a1)")
    plt.title("Empirical σ(a1) vs resolution ")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Ngrid, method2StdA2, "d-", label="Method 2 σ_emp(a2)")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("Empirical std σ(a2)")
    plt.title("Empirical σ(a2) vs resolution (Method 2)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
   

    return 5


def affectOfInterval(acceptanceList, nSamples, a1Real, a2Real, nRepeats):
    """
    Studies the affect of the acceptance interval on the accuracy of a1,a2


    Parameters
    ----------
    acceptanceList : array
        DESCRIPTION.
    nSamples : int
        number of events to generate.
    a1Real : float
        Value of a1 used to generate data.
    a2Real : float
        Value of a2 used to generate data...
    nRepeats : int
        number of repeated tests per value in samplesSize

    Returns
    -------
    

    """
    
   
    
    method1Mean = [] #
    method1MeanPercentageErrorRepeats = []
    method1Std = []
    method1Bias = []
    method2Chi2OverNdf = []
    
    method2MeanA1 = []
    method2MeanPercentageErrorRepeatsA1 = []
    method2StdA1 = []
    method2BiasA1 = []
    
    method2MeanA2 = []
    method2MeanPercentageErrorRepeatsA2 = []
    method2StdA2 = []
    method2BiasA2 = []

    
    method2SigmaPredMeanA1 = []
    method2PullMeanA1 = []
    method2PullstdA1 = []
    
    method2SigmaPredMeanA2 = []
    method2PullMeanA2 = []
    method2PullstdA2 = []
    
    rmseMethod1A1 = []
    rmseMethod2A1 = []
    rmseMethod2A2 = []
    

    acceptanceValues = []
    
    for c in acceptanceList: #Repeats for every acceptance interval in the list acceptanceList
        
        cosMinChange = -float(c) #Gets min and max interval from list
        cosMaxChange = float(c)
        
        m1Vals= [] #list of a1 values generated by afb method 
        m1Sigmas = [] #list of errors on a1 from afb method
        
        m2ValsA1= []    #list of a1 values generated by binned fit method
        m2SigmasA1 = [] #list of estimated errors on a1 generated by fitting routine 
        
        m2ValsA2= [] #list of a2 values generated by binned fit method using 
        m2SigmasA2 = [] #list of estimated errors on a2 generated by fitting routine 
        
        m2Chi2Vals = [] #chi^2 value for each fit
        m2DofVals = [] #Degrees of freedom for each fit

        for _ in range(nRepeats): #repeats finding a1, a2 from both methods nRepeats time
           
            gen = angularEventGenerator(a1=a1Real, a2=a2Real, cosMin=cosMinChange, cosMax=cosMaxChange, nSamples=nSamples)
            data = gen.rejectionSamplingGenerator() #generates data according to class values. Different accepetance 

            a1M1, sigmaM1Predicted = gen.a1AFB(data)
            m1Vals.append(a1M1)
            #a1, a2, sigmaA1Predicted, sigmaA2Predicted, chi2Val, dof
            
            a1M2, a2M2, sigmaA1M2Predicted, sigmaA2M2Predicted,  chi2M2, dofM2 = gen.binnedFitA1ChangingWidth(data, returnFitInfo=True)
            #a1,a2, estimated errors on both, chi^2, degrees of  freedom from binned fitting method

            
            m2SigmasA1.append(sigmaA1M2Predicted)
            m2ValsA1.append(a1M2)
            
            m2SigmasA2.append(sigmaA2M2Predicted)
            m2ValsA2.append(a2M2)
                 
            m2Chi2Vals.append(chi2M2)
            m2DofVals.append(dofM2)

        """Calculates and stores stats of the nRepeats repeated runs. Runs for every value in acceptanceList"""


   
        
        """Method 1 AFB values for a1 Parameter. """

        
        m1MeanN, m1PctErrN, m1StdN, m1BiasN, rmseM1A1 = accuracyCalculator(m1Vals, a1Real)
       
        
        rmseMethod1A1.append(rmseM1A1)
        method1Mean.append(m1MeanN)
        method1MeanPercentageErrorRepeats.append(m1PctErrN)
        method1Std.append(m1StdN)
        method1Bias.append(m1BiasN)
        

        """Method 2 binned fitting values for a1 Parameter"""
        
        m2MeanNA1, m2PctErrNA1, m2StdNA1, m2BiasNA1, rmseM2A1 = accuracyCalculator(m2ValsA1, a1Real)
           

        rmseMethod2A1.append(rmseM2A1)
        method2MeanA1.append(m2MeanNA1)
        method2MeanPercentageErrorRepeatsA1.append(m2PctErrNA1)
        method2StdA1.append(m2StdNA1)
        method2BiasA1.append(m2BiasNA1)
        
        meanSigma2A1, pullMean2A1, pullStd2A1 = errorCalculator(m2SigmasA1, m2ValsA1, a1Real)
        method2SigmaPredMeanA1.append(meanSigma2A1)
        method2PullMeanA1.append(pullMean2A1)
        method2PullstdA1.append(pullStd2A1)
        
        """Method 2 binned fitting values for a2 Parameter"""
        
        m2MeanNA2, m2PctErrNA2, m2StdNA2, m2BiasNA2, rmseM2A2 = accuracyCalculator(m2ValsA2, a2Real)

        rmseMethod2A2.append(rmseM2A2)
        method2MeanA2.append(m2MeanNA2)
        method2MeanPercentageErrorRepeatsA2.append(m2PctErrNA2)
        method2StdA2.append(m2StdNA2)
        method2BiasA2.append(m2BiasNA2)
        
        meanSigma2A2, pullMean2A2, pullStd2A2 = errorCalculator(m2SigmasA2, m2ValsA2, a2Real)
        method2SigmaPredMeanA2.append(meanSigma2A2)
        method2PullMeanA2.append(pullMean2A2)
        method2PullstdA2.append(pullStd2A2)
        
        acceptanceValues.append(c)

        m2Chi2Vals = np.asarray(m2Chi2Vals, dtype=float)
        m2DofVals = np.asarray(m2DofVals, dtype=float)
        
       
        

        print(f"[Accepetance interval] [-{c},{c}], with [Number of repeats] M ={nRepeats}")
        
        
        print(f"  Method 1 (a1): mean={m1MeanN:.5f}, bias={m1BiasN:.5f}, %err={m1PctErrN:.2f}, "
              f"std={m1StdN:.5f}, "
              f" RMSE = {rmseM1A1:.5f}")
        
        # Method 2 (fit, a1)
        print(f"  Method 2 (a1): mean={m2MeanNA1:.5f}, bias={m2BiasNA1:.5f}, %err={m2PctErrNA1:.2f}, "
              f"std={m2StdNA1:.5f}, ⟨σpred⟩={meanSigma2A1:.5f}, pull μ={pullMean2A1:.5f}, "
              f"pull σ={pullStd2A1:.3f}, <chi2/ndf>={method2Chi2OverNdf[-1]:.2f}, RMSE = {rmseM2A1:.5f}")
        
        # Method 2 (fit, a2)
        print(f"  Method 2 (a2): mean={m2MeanNA2:.5f}, bias={m2BiasNA2:.5f}, %err={m2PctErrNA2:.2f}, "
              f"std={m2StdNA2:.5f}, ⟨σpred⟩={meanSigma2A2:.5f}, pull μ={pullMean2A2:.5f}, "
              f"pull σ={pullStd2A2:.3f}, RMSE = {rmseM2A2:.5f}")
        

    Ngrid = np.array(acceptanceValues, dtype=float)
    
    plt.figure()
    plt.plot(Ngrid, rmseMethod1A1, "o-", label="Method 1 RMSE(a1)")
    plt.plot(Ngrid, rmseMethod2A1, "s-", label="Method 2 RMSE(a1)")
    plt.xlabel("accepetance angle c [-c,c]")
    plt.ylabel("RMSE(a1)")
    plt.title("RMSE(a1) vs accepetance angle c [-c,c] ")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Ngrid, rmseMethod2A2, "d-", label="Method 2 RMSE(a2)")
    plt.xlabel("accepetance angle c [-c,c]")
    plt.ylabel("RMSE(a2)")
    plt.title("RMSE(a2) vs accepetance angle c [-c,c] ")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(Ngrid, method2Chi2OverNdf, "o-", label="Method 2 ⟨χ²/ndf⟩")
    plt.axhline(1.0, linestyle="--", color="k", label="ideal = 1")
    plt.xlabel("accepetance angle c [-c,c]")
    plt.ylabel("⟨χ²/ndf⟩")
    plt.title("Fit quality vs accepetance angle c [-c,c]   (Method 2)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
    
    plt.figure()
    plt.plot(Ngrid, method1Std,   "o-", label="Method 1 σ_emp(a1)")
    plt.plot(Ngrid, method2StdA1, "s-", label="Method 2 σ_emp(a1)")
    plt.xlabel("accepetance angle c [-c,c]")
    plt.ylabel("Empirical std σ(a1)")
    plt.title("Empirical σ(a1) vs accepetance angle c [-c,c]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Ngrid, method2StdA2, "d-", label="Method 2 σ_emp(a2)")
    plt.xlabel("Resolution (radians)")
    plt.ylabel("Empirical std σ(a2)")
    plt.title("Empirical σ(a2) vs resolution (Method 2)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    

   


def stepTwo( a1Real, a2Real, cosMinPerfect, cosMaxPerfect, nRepeats):
    
    cosMin = cosMinPerfect
    cosMax = cosMaxPerfect
    
    sampleSizes = ( 10_000,20_000, 50_000,100_000, 250_000, 500_000,1_000_000) #10_000_000 takes a couple mins to finish. samples smaller than 1000 have massive errors, distorts the grahph     
    #sampleSizes = (10_000,50_000,100_000,250_000)
    #sampleSizes = (10_00,100_000,1_000_000)
    
    eventNumberAffect = affectOfN(sampleSizes, a1Real, a2Real, cosMin, cosMax, nRepeats)

def stepThree(a1Real, a2Real, cosMinPerfect, cosMaxPerfect, nRepeats):
    
    nSamples = 500_000
    acceptanceList = [1,0.9, 0.8, 0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    resolution = 0.003
    resolutionList = [0,0.003, 0.03,0.05, 0.1, 0.3,0.5,1,1.5,2]
    
    cosMin = cosMinPerfect
    cosMax = cosMaxPerfect
    
    # Resolution scan at fixed N and acceptance
    resolutionAffect = affectOfResolution(resolutionList, nSamples, a1Real, a2Real, cosMin, cosMax, nRepeats)

    #Acceptance scan at fixed N
    acceptanceAffect = affectOfInterval(acceptanceList, nSamples, a1Real, a2Real, nRepeats)
    
    
a1Real = 0.04 
a2Real = 1.0 
cosMinPerfect = -1
cosMaxPerfect = 1
nRepeats=1000 #Number of repeats per sample size

stepTwo(a1Real, a2Real, cosMinPerfect, cosMaxPerfect, nRepeats)

stepThree(a1Real, a2Real, cosMinPerfect, cosMaxPerfect, nRepeats)



