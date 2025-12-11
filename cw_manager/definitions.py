def phaseParamName(order):
    """
    Returns parameter names based on derivative order.
    """
    freqParamName = ["freq", "f1dot", "f2dot", "f3dot", "f4dot"]
    freqDerivParamName = ["df", "df1dot", "df2dot", "df3dot", "df4dot"]        
    return freqParamName[:order+1], freqDerivParamName[:order+1]

def injParamName():
    return ["Alpha", "Delta", "refTime", "aPlus", "aCross", "psi", "Freq"]

def taskName(target_name, stage, cohDay, order):
    """
    Standardized task naming convention.
    """
    return f'{target_name}_{stage}_TCoh{cohDay}_O{order}'
