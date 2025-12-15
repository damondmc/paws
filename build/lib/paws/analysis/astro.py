# Import necessary libraries
import numpy as np

# Function to compute the upper limit of strain based on the age of the source
def age_strainLimit(tau, Izz, I0, r):
    """
    Calculate the strain limit based on the age of a gravitational wave source.
    
    Parameters:
    tau (float): Characteristic age of the source in years.
    Izz (float): Principal moment of inertia of the source in kg*m^2.
    I0 (float): Reference moment of inertia (standard value) in kg*m^2.
    r (float): Distance to the source in kiloparsecs (kpc).
    
    Returns:
    float: Upper limit of strain (h0).
    """
    # Strain calculation formula
    h0 = 2.3e-24 * (1./r) * np.sqrt((1000/tau) * (Izz/I0))
    return h0

# check the number 
# Function to compute the gravitational wave strain from ellipticity
def h0_from_ellipticity(freq, e, Izz, I0, r):
    """
    Calculate the strain based on the source's ellipticity.
    
    Parameters:
    freq (float): Gravitational wave frequency in Hz.
    e (float): Ellipticity of the source.
    Izz (float): Principal moment of inertia of the source in kg*m^2.
    I0 (float): Reference moment of inertia (standard value) in kg*m^2.
    r (float): Distance to the source in kiloparsecs (kpc).
    
    Returns:
    float: Strain (h0).
    """
    # Strain calculation formula using ellipticity
    h0 = 1.1e-24 * (e/1.0e-6) * (Izz/I0) * (freq/1.0e3)**2 * (1./r)
    return h0

# Function to compute the ellipticity from a given strain value
def ellipticity_from_h0(freq, h0, Izz, I0, r):
    """
    Calculate the source's ellipticity from a given strain value.
    
    Parameters:
    freq (float): Gravitational wave frequency in Hz.
    h0 (float): Strain of the gravitational wave.
    Izz (float): Principal moment of inertia of the source in kg*m^2.
    I0 (float): Reference moment of inertia (standard value) in kg*m^2.
    r (float): Distance to the source in kiloparsecs (kpc).
    
    Returns:
    float: Ellipticity (e).
    """
    # Ellipticity calculation formula
    e = 9.46e-5 * (h0 /1e-24) * (I0/Izz) * (r/1.) * (1e2/freq)**2 
    return e


### check the number 
# Function to compute the strain from the braking index parameter alpha
def h0_from_alpha(freq, alpha, Izz, I0, r):
    """
    Calculate the strain based on the braking index parameter alpha.
    
    Parameters:
    freq (float): Gravitational wave frequency in Hz.
    alpha (float): Braking index parameter (dimensionless).
    Izz (float): Principal moment of inertia of the source in kg*m^2.
    I0 (float): Reference moment of inertia (standard value) in kg*m^2.
    r (float): Distance to the source in kiloparsecs (kpc).
    
    Returns:
    float: Strain (h0).
    """
    # Strain calculation formula using alpha
    h0 = 3.6e-23 * (alpha/1.0e-3) * (freq/1.0e3)**3 * (1./r)
    return h0

# Function to compute the braking index parameter alpha from a given strain value
def alpha_from_h0(freq, h0, Izz, I0, r):
    """
    Calculate the braking index parameter alpha from a given strain value.
    
    Parameters:
    freq (float): Gravitational wave frequency in Hz.
    h0 (float): Strain of the gravitational wave.
    Izz (float): Principal moment of inertia of the source in kg*m^2.
    I0 (float): Reference moment of inertia (standard value) in kg*m^2.
    r (float): Distance to the source in kiloparsecs (kpc).
    
    Returns:
    float: Braking index parameter (alpha).
    """
    # Alpha calculation formula
    alpha = 0.028 * (h0/1e-24) * (r/1.) * (1e2/freq)**3
    return alpha
