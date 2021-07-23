import numpy as np
from sol_options import Options
from prox_Fsvd_corr import prox_fsvd_corr
from ProxG import prox_G
from ProxH import prox_H

def calculate_spinner_both_lambdas_corr(Y, svd_Z, lambda_N, lambda_L, WGTs):

    # cases
    p = svd_Z["Vt"].shape[0]
    # Solver options
    deltaInitial1 = Options.deltaInitial1
    deltaInitial2 = Options.deltaInitial2
    scaleStep = Options.scaleStep
    ratioStep = Options.ratioStep
    mu = Options.mu
    deltaInc = Options.deltaInc
    deltaDecr = Options.deltaDecr
    ratioInc = Options.ratioInc
    ratioDecr = Options.ratioDecr
    maxIters = Options.maxIters
    epsPri = Options.epsPri
    epsDual = Options.epsDual


    Dk = np.zeros((p,p))
    W1k = np.zeros((p,p))
    W2k = np.zeros((p,p))


    # ADMM loop
    ## ADMM loop
    delta1 = deltaInitial1
    delta2 = deltaInitial2
    counterr = 0
    CsB = []
    DsB = []
    DsDp = []
    Dlts1 = []
    Dlts2 = []

    while True:
        Bnew = prox_fsvd_corr(svd_Z,Dk, W1k, delta1)
        Cnew = prox_G(Dk, W2k, delta2, lambda_N)
        Dnew = prox_H(Bnew, Cnew, delta1, delta2, W1k, W2k, lambda_L, WGTs)
        W1k = W1k + delta1 * (Dnew - Bnew)
        W2k = W2k + delta2 * (Dnew - Cnew)

        rk1 = Cnew - Bnew
        rk2 = Dnew - Bnew
        sk = Dnew - Dk
        rknorm1 = np.linalg.norm(rk1, 'fro')
        Bnorm = np.linalg.norm(Bnew, 'fro')
        rknormR1 = rknorm1 / Bnorm
        rknorm2 = np.linalg.norm(rk2, 'fro')
        rknormR2 = rknorm2 / Bnorm
        sknorm = np.linalg.norm(sk, 'fro')
        sknormR = sknorm / np.linalg.norm(Dk, 'fro')
        counterr = counterr + 1
        #print(counterr)
        CsB.append(rknormR1)
        DsB.append(rknormR2)
        DsDp.append(sknormR)
        Dlts1.append(delta1)
        Dlts2.append(delta2)
        Dk = Dnew

        # rations update
        if counterr % 20 == 10:
            if rknorm1 > mu * rknorm2:
                ratioStep = ratioStep * ratioInc
            else:
                if rknorm2 > mu * rknorm1:
                    ratioStep = ratioStep / ratioDecr
        # scale update
        if counterr % 20 == 0:
            if np.mean([rknorm1, rknorm2]) > mu * sknorm:
                scaleStep = scaleStep * deltaInc
            else:
                if sknorm > mu * np.mean([rknorm1, rknorm2]):
                    scaleStep = scaleStep / deltaDecr

        delta1 = scaleStep * deltaInitial1
        delta2 = scaleStep * ratioStep * deltaInitial2

        # stopping cryteria
        if ((rknormR1 < epsPri) and (rknormR2 < epsPri) and (sknormR < epsDual)):
            break
        if counterr > maxIters:
            break
        if Bnorm < 1e-16:
            Bnew = np.zeros((p, p))
            Cnew = np.zeros((p, p))
            Dnew = np.zeros((p, p))
            break

    out = {}
    out["counterr"] = counterr
    out["Dlts1"] = Dlts1
    out["Dlts2"] = Dlts2
    out["Blast"] = Bnew
    out["Clast"] = Cnew
    out["Dlast"] = Dnew
    out["CsB"] = CsB
    out["DsB"] = DsB
    out["DsDp"] = DsDp
    out["B"] = Dnew

    return out



