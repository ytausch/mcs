import numpy as np


def getalp(alpu, alpo, gTp, pTGp):
    """
        #get minimizer alp in [alpu,alpo] for a univariate quadratic
        #	q(alp)=alp*gTp+0.5*alp^2*pTGp
        #lba	lower bound active
        #uba	upper bound active
        #
        #ier	 0 (finite minimizer)
        #	 1 (unbounded minimum)
    """
    # alpu,alpo,gTp,pTGp = alpu,alpo,g[k],q[k] #  SHOULD BE CLOSE ATER DEBUGGING - OK

    lba = False
    uba = False

    # determine unboundedness
    ier = 0
    if alpu == -np.inf and (pTGp < 0 or (pTGp == 0 and gTp > 0)):
        ier = 1
        lba = True
    if alpo == np.inf and (pTGp < 0 or (pTGp == 0 and gTp < 0)):
        ier = 1
        uba = True
    if ier:
        alp = np.nan
        return alp, lba, uba, ier

    # determine activity
    if pTGp == 0 and gTp == 0:
        alp = 0
    elif pTGp <= 0:
        # concave case minimal at a bound
        if alpu == -np.inf:
            lba = False
        elif alpo == np.inf:
            lba = True
        else:
            lba = (2 * gTp + (alpu + alpo) * pTGp > 0)
            # end  alpu == -np.inf
        uba = not lba
    else:
        alp = -gTp / pTGp  # unconstrained optimal step
        lba = (alp <= alpu)  # lower bound active
        uba = (alp >= alpo)  # upper bound active

    if lba:
        alp = alpu
    if uba:
        alp = alpo

    if abs(alp) == np.inf:
        gTp, pTGp, alpu, alpo, alp, lba, uba, ier
    return alp, lba, uba, ier
