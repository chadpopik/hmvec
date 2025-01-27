import sys, os
homepath = "/global/homes/c/cpopik/"
packpath = homepath+"Packages/"

import numpy as np
import scipy.constants as constants
from scipy.interpolate import interp2d,interp1d
from copy import deepcopy

sys.path.append(packpath + "hmvec/")
import camb
import hmvec as hm

# sys.path.append(homepath + "Capybara/")
# from HODModels import Kou2023, More2015

# sys.path.append(packpath + "SOLikeT/")
# from soliket.szlike import cosmo, gnfw, projection_functions
# from soliket.szlike.projection_functions import kpc_cgs,C_CGS,ME_CGS,MP_CGS,sr2sqarcmin,XH
# from soliket.constants import (MPC2CM,C_M_S,h_Planck,k_Boltzmann,electron_mass_kg,proton_mass_kg,hydrogen_fraction,T_CMB,ST_CGS,MSUN_CGS,G_CGS)




def Get_Spectra(hodname, HaloModel, nocut=False, pkname="y", logMbounds=False, param_override=None, NoM200=False):
    m200critz = HaloModel.m200critz
    r200critz = HaloModel.r200critz
    family='pres'
    pparams = {}
    pparams['battaglia_pres_gamma'] = HaloModel.p['battaglia_pres_gamma']
    pparams['battaglia_pres_alpha'] = HaloModel.p['battaglia_pres_alpha']
    pparams.update(hm.battaglia_defaults[family])

    if NoM200 is True:
        pkname = pkname+"_NoM200"
        m200critz = HaloModel.ms
        r200critz = hm.R_from_M(HaloModel.ms,HaloModel.rho_critical_z(HaloModel.zs)[:,None],delta=200.)
        
        if nocut is False:
            if param_override is not None:
                for key in param_override.keys():
                    if key=='battaglia_gas_gamma': pparams[key] = param_override[key]
                    elif key in hm.battaglia_defaults[family]: pparams[key] = param_override[key]
                    else: pass

            presFunc = lambda x: hm.P_e_generic_x(x,m200critz[...,None],r200critz[...,None],HaloModel.zs[:,None,None],HaloModel.omb,HaloModel.omm,HaloModel.rhocritz[...,None,None],
                                alpha=pparams['battaglia_pres_alpha'],gamma=pparams['battaglia_pres_gamma'],
                                P0_A0=pparams['P0_A0'],P0_alpham=pparams['P0_alpham'],P0_alphaz=pparams['P0_alphaz'],
                                xc_A0=pparams['xc_A0'],xc_alpham=pparams['xc_alpham'],xc_alphaz=pparams['xc_alphaz'],
                                beta_A0=pparams['beta_A0'],beta_alpham=pparams['beta_alpham'],beta_alphaz=pparams['beta_alphaz'])
            cgs = HaloModel.rvirs/r200critz
            ks,pkouts = hm.generic_profile_fft(presFunc,cgs,r200critz[...,None],HaloModel.zs,HaloModel.ks,HaloModel.xmax,HaloModel.nxs,do_mass_norm=False)
            HaloModel.pk_profiles[pkname] = pkouts.copy()*4*np.pi*(HaloModel.sigmaT/(HaloModel.mElect*constants.c**2))* \
                                                (r200critz**3*((1+HaloModel.zs)**2/HaloModel.h_of_z(HaloModel.zs))[...,None])[...,None]
    
    if nocut is not False:  # Get rid of the cutoff x value when FFTing P(x) to P(k)
        
        if param_override is not None:
            for key in param_override.keys():
                if key=='battaglia_gas_gamma': pparams[key] = param_override[key]
                elif key in hm.battaglia_defaults[family]: pparams[key] = param_override[key]
                else: pass
        
        pkname = pkname+"_NoCut"
        presFunc = lambda x: hm.P_e_generic_x(x,m200critz[...,None],r200critz[...,None],HaloModel.zs[:,None,None],HaloModel.omb,HaloModel.omm,HaloModel.rhocritz[...,None,None],
                            alpha=pparams['battaglia_pres_alpha'],gamma=pparams['battaglia_pres_gamma'],
                            P0_A0=pparams['P0_A0'],P0_alpham=pparams['P0_alpham'],P0_alphaz=pparams['P0_alphaz'],
                            xc_A0=pparams['xc_A0'],xc_alpham=pparams['xc_alpham'],xc_alphaz=pparams['xc_alphaz'],
                            beta_A0=pparams['beta_A0'],beta_alpham=pparams['beta_alpham'],beta_alphaz=pparams['beta_alphaz'])
        ks_nocut,pkouts_nocut = generic_profile_fft_nocut(presFunc,r200critz[...,None],HaloModel.zs,HaloModel.ks,HaloModel.xmax,HaloModel.nxs)
        HaloModel.pk_profiles[pkname] = pkouts_nocut.copy()*4*np.pi*(HaloModel.sigmaT/(HaloModel.mElect*constants.c**2))* \
                                        (r200critz**3*((1+HaloModel.zs)**2/HaloModel.h_of_z(HaloModel.zs))[...,None])[...,None]
            
    if logMbounds is not False:  # Zero the integrand when getting spectra over ms outside the range 
        mrange, mrangestr = (HaloModel.ms>10**logMbounds[0]) & (HaloModel.ms < 10**logMbounds[1]), f"_{logMbounds[0]}<logM<{logMbounds[1]}"
        hodname_mrange, y_mrange = hodname+mrangestr, pkname+mrangestr
        HaloModel.hods[hodname_mrange], HaloModel.pk_profiles[y_mrange] = deepcopy(HaloModel.hods[hodname]), deepcopy(HaloModel.pk_profiles[pkname])
        HaloModel.pk_profiles[y_mrange][:, ~mrange, :]=0
        for key in ['Nc', 'Ns', 'NsNsm1', 'NcNs']: HaloModel.hods[hodname_mrange][key][:, ~mrange]=0
        
        Pgg, Pyy = HaloModel.get_power(hodname_mrange, verbose=False), HaloModel.get_power(y_mrange, verbose=False)
        Pgy = HaloModel.get_power(hodname_mrange, y_mrange, verbose=False)
        Pgy_1h, Pgy_2h = HaloModel.get_power_1halo(hodname_mrange, y_mrange), HaloModel.get_power_2halo(hodname_mrange, y_mrange, verbose=False)
        hodname, pkname = hodname_mrange, y_mrange
        
    else: 
        Pgg, Pyy = HaloModel.get_power(hodname, verbose=False), HaloModel.get_power(pkname, verbose=False)
        Pgy = HaloModel.get_power(hodname, pkname, verbose=False)
        Pgy_1h, Pgy_2h = HaloModel.get_power_1halo(hodname, pkname), HaloModel.get_power_2halo(hodname, pkname, verbose=False)

    results = {'HOD_Name': hodname, 'y_name': pkname,
            'ks': HaloModel.ks, 'Pgg': Pgg, 'Pyy': Pyy, 'Pgy': Pgy, 'Pgy_1h': Pgy_1h, 'Pgy_2h': Pgy_2h, 'R200c': r200critz}
    try: results['pthFunc'] = presFunc
    except: results['pthFunc']=None
    
    return results





def make_halo_model(zs, ks, ms=None, params=None, mass_function="sheth-torman", xmax=25, nxs=1000):
    # First, make the Halo Model
    HaloModel = hm.HaloModel(zs,ks,ms=ms, params=params, mass_function=mass_function)
    
    # Define some values used to construct the halo model and its components for easier access.
    HaloModel.rhocritz = HaloModel.rho_critical_z(HaloModel.zs)
    if HaloModel.mdef=='vir':
        HaloModel.delta_rhos1 = HaloModel.rhocritz*HaloModel.deltav(HaloModel.zs)
    elif HaloModel.mdef=='mean':
        HaloModel.delta_rhos1 = HaloModel.rho_matter_z(HaloModel.zs)*200.
    HaloModel.rvirs = HaloModel.rvir(HaloModel.ms[None,:],HaloModel.zs[:,None])
    HaloModel.cs = HaloModel.concentration()
    HaloModel.delta_rhos2 = 200.*HaloModel.rho_critical_z(HaloModel.zs)
    HaloModel.m200critz = hm.mdelta_from_mdelta(HaloModel.ms,HaloModel.cs,HaloModel.delta_rhos1,HaloModel.delta_rhos2)
    HaloModel.r200critz = hm.R_from_M(HaloModel.m200critz,HaloModel.rho_critical_z(HaloModel.zs)[:,None],delta=200.)
    
    HaloModel.omb = HaloModel.p['ombh2'] / HaloModel.h**2.
    HaloModel.omm = HaloModel.om0
    
    HaloModel.rvirs = HaloModel.rvir(HaloModel.ms[None,:],HaloModel.zs[:,None])
    HaloModel.cgs = HaloModel.rvirs/HaloModel.r200critz
    HaloModel.sigmaT=constants.physical_constants['Thomson cross section'][0] # units m^2
    HaloModel.mElect=constants.physical_constants['electron mass'][0] / hm.default_params['mSun']# units kg

    HaloModel.XH=.76
    HaloModel.eFrac=2.0*(HaloModel.XH+1.0)/(5.0*HaloModel.XH+3.0)
    
    HaloModel.xmax = xmax
    HaloModel.nxs = nxs
    
    HaloModel.add_battaglia_pres_profile("y",family='pres', xmax=xmax,nxs=nxs, ignore_existing=True)
    
    return HaloModel





def generic_profile_fft_nocut(rhofunc_x,rss,zs,ks,xmax,nxs):
    xs = np.linspace(0.,xmax,nxs+1)[1:]
    rhos = rhofunc_x(xs)
    if rhos.ndim==1: rhos = rhos[None,None]
    else: assert rhos.ndim==3
    # u(kt)
    kts,ukts = hm.fft.fft_integral(xs,rhos)
    uk = ukts/kts[None,None,:]
    kouts = kts/rss/(1+zs[:,None,None]) # divide k by (1+z) here for comoving FIXME: check this!
    ukouts = np.zeros((uk.shape[0],uk.shape[1],ks.size))
    for i in range(uk.shape[0]):
        for j in range(uk.shape[1]):
            pks = kouts[i,j]
            puks = uk[i,j]
            puks = puks[pks>0]
            pks = pks[pks>0]
            ukouts[i,j] = np.interp(ks,pks,puks,left=puks[0],right=0)
    return ks, ukouts





def HODmean(thing, HODname, HaloModel):
    return np.trapz(np.trapz(thing*HaloModel.nzm[..., None]*HaloModel._get_hod(HODname), HaloModel.ms, axis=1), HaloModel.zs, axis=0) / \
np.trapz(np.trapz(HaloModel.nzm[..., None]*HaloModel._get_hod(HODname), HaloModel.ms, axis=1), HaloModel.zs, axis=0)


def r200_mean(HODname, HaloModel, cmass_ws=False):
    if cmass_ws is True:
        p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
        return np.average(HaloModel.r200critz, weights=p, axis=1)
    else:
        hod = HaloModel._get_hod(HODname)[..., 0]
        return np.trapz(HaloModel.nzm*HaloModel.r200critz*hod, HaloModel.ms, axis=1) \
                /np.trapz(HaloModel.nzm*hod, HaloModel.ms, axis=1)

def unitconversion(HODname, HaloModel, cmass_ws=False):
    if cmass_ws is True:
        p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
        r200_cubed_mean = np.average(HaloModel.r200critz**3, weights=p, axis=1)
    else:
        hod = HaloModel._get_hod(HODname)[..., 0]
        r200_cubed_mean = np.trapz(HaloModel.nzm*HaloModel.r200critz**3*hod, HaloModel.ms, axis=1) \
                            /np.trapz(HaloModel.nzm*hod, HaloModel.ms, axis=1)
    return 4*np.pi*(HaloModel.sigmaT/(HaloModel.mElect*constants.c**2))* \
                (r200_cubed_mean*((1+HaloModel.zs)**2/HaloModel.h_of_z(HaloModel.zs)))




# Reverse FFT some mass averaged P(k) to P(x)
# Needs min/max/number for x to get P for, also a R_200 value
def Pex_rev_inter_FFT(ukouts_inter, xmin_in, xmax_in, nxs_in, r200_inter, HaloModel):
    # Create an array of x values we want out of our RFFT, must be equally spaced
    xs_inter = np.linspace(0, 2*xmax_in, int(2*(2*xmax_in/xmin_in))+1)[1:]
    # Make the k's that will produce those xs
    kts_inter = np.fft.rfftfreq(xs_inter.size, ((xs_inter[-1]-xs_inter[0])/xs_inter.size)) *2*np.pi 
    # Scale the k's the same way that the halo model k's are scaled for proper interpolation
    if ukouts_inter.ndim==2: ksout_inter = kts_inter/r200_inter[...,None]/(1+HaloModel.zs[:,None])
    elif ukouts_inter.ndim==3: ksout_inter = kts_inter/r200_inter[...,None]/(1+HaloModel.zs[:,None,None])

    # Interpolate P(k) with hmvec k's to P(k) with k's that can be used in the RFFT
    uk_inter = np.zeros((*ukouts_inter.shape[:-1],ksout_inter.shape[-1]))  # P(k) we're going to
    for ind in np.ndindex(uk_inter.shape[:-1]):
        pks = HaloModel.ks  # k's we're starting with
        puks = ukouts_inter[ind]  # P(k) we're starting with
        puks = puks[pks>0]
        pks = pks[pks>0]
        uk_inter[ind] = np.interp(ksout_inter[ind],pks,puks,left=puks[0],right=0)

    # Take out any factors added in to get what would be the raw results of the FFT, then RFFT back
    ukts_inter = uk_inter * kts_inter
    integrand_inter = -ukts_inter/((xs_inter[-1]-xs_inter[0])/xs_inter.size)
    Pex_inter = 2*np.fft.irfft(integrand_inter*1j, xs_inter.size)/xs_inter
    del kts_inter, ksout_inter, uk_inter, ukts_inter, integrand_inter

    # Now interpolate P(x) with RFFT x's to the actual x array we want
    xs_in = np.geomspace(xmin_in, xmax_in, nxs_in)  # x's we're going to
    Pex_in = np.zeros((*Pex_inter.shape[:-1],xs_in.size))  # the profile we're going to
    for ind in np.ndindex(Pex_in.shape[:-1]):
        pks = xs_inter  # x's we're starting with
        puks = Pex_inter[ind]  # P(x) we're starting with
        puks = puks[pks>0]
        pks = pks[pks>0]
        Pex_in[ind] = np.interp(xs_in,pks,puks,left=puks[0],right=0)

    del xs_inter, Pex_inter
    return xs_in, Pex_in



def Pth1h_rev(HaloModel, results):
    Pgy = results['Pgy_1h']/(1-np.exp(-(HaloModel.ks/HaloModel.p['kstar_damping'])**2.))
    weight = HaloModel._get_hod(results['HOD_Name'])*HaloModel.nzm[..., None]
    ybar = np.trapz(Pgy, HaloModel.zs, axis=0)/np.trapz(np.trapz(weight, HaloModel.ms, axis=1), HaloModel.zs, axis=0)
    
    conversion_factor=4*np.pi*(HaloModel.sigmaT/(HaloModel.mElect*constants.c**2))* \
                (results['R200c']**3*((1+HaloModel.zs[...,None])**2/HaloModel.h_of_z(HaloModel.zs)[..., None]))
    convfact_av = np.trapz(np.trapz(conversion_factor[..., None]*weight, HaloModel.ms, axis=1), HaloModel.zs, axis=0)/np.trapz(np.trapz(weight, HaloModel.ms, axis=1), HaloModel.zs, axis=0)
    Pekbar = ybar/convfact_av
    
    ktterm = np.trapz(np.trapz((results['R200c']*(1+HaloModel.zs)[..., None])[..., None]*weight, HaloModel.ms, axis=1), HaloModel.zs, axis=0)/np.trapz(np.trapz(weight, HaloModel.ms, axis=1), HaloModel.zs, axis=0)
    ktouts_rev = HaloModel.ks*ktterm
    uktouts_rev = Pekbar*ktouts_rev
    xs = np.linspace(0.,HaloModel.xmax,HaloModel.nxs+1)[1:]
    kts_inter = np.fft.rfftfreq(xs.size, (xs[-1]-xs[0])/xs.size)*2*np.pi
    ukts_inter = np.interp(kts_inter,ktouts_rev,uktouts_rev, left=uktouts_rev[0],right=0)
    Pex_inter = 2*np.fft.irfft(-ukts_inter*1j/((xs[-1]-xs[0])/xs.size), xs.size)/xs
    
    weight2 = HaloModel._get_hod(results['HOD_Name'], lowklim=True)[:, :, 0]*HaloModel.nzm
    r200av = np.trapz(np.trapz(results['R200c']*weight2, HaloModel.ms, axis=1), HaloModel.zs, axis=0) \
/ np.trapz(np.trapz(weight2, HaloModel.ms, axis=1), HaloModel.zs, axis=0)
            
    return xs*r200av, Pex_inter/HaloModel.eFrac/(3.086e24/1.989e33)
    



def Pth_rev_old(P, results, xmin, xmax, nxs, HaloModel, cmass_ws=False):
    Pek = results[P]/unitconversion(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws)[..., None]
    if P == "Pgy_1h":
        integrand = HaloModel.nzm[..., None] * HaloModel._get_hod(results['HOD_Name'])
        Pek = Pek/np.trapz(integrand, HaloModel.ms, axis=-2)/(1-np.exp(-(HaloModel.ks/HaloModel.p['kstar_damping'])**2.))
    xs_rev, Pex_rev = Pex_rev_inter_FFT(Pek, xmin, xmax, nxs, r200_mean(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws), HaloModel)
    rs_rev = xs_rev[None, ...]*r200_mean(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws)[..., None]
    Pthx_rev = Pex_rev/HaloModel.eFrac/(3.086e24/1.989e33)
    return rs_rev, Pthx_rev










# def Pex_rev_inter_FFT3(ukouts_inter, xmin_in, xmax_in, nxs_in, HaloModel, HODName):
#     # Create an array of x values we want out of our RFFT, must be equally spaced
#     xs_inter = np.linspace(0, 2*xmax_in, int(2*(2*xmax_in/xmin_in))+1)[1:]
#     # Make the k's that will produce those xs
#     kts_inter = np.fft.rfftfreq(xs_inter.size, ((xs_inter[-1]-xs_inter[0])/xs_inter.size)) *2*np.pi 
#     # Scale the k's the same way that the halo model k's are scaled for proper interpolation
#     r200_inter = hmvec.R_from_M(HaloModel.ms,HaloModel.rho_critical_z(HaloModel.zs)[:,None],delta=200.)
#     ksout_inter = 
    
#     kt_inter = np.zeros((*ukouts_inter.shape[:-1],ksout_inter.shape[-1]))  # P(k) we're going to
#     for ind in np.ndindex(uk_inter.shape[:-1]):
#         pks = HaloModel.ks  # k's we're starting with
#         puks = ukouts_inter[ind]  # P(k) we're starting with
#         puks = puks[pks>0]
#         pks = pks[pks>0]
#         uk_inter[ind] = np.interp(ksout_inter[ind],pks,puks,left=puks[0],right=0)
    

#     # Interpolate P(k) with hmvec k's to P(k) with k's that can be used in the RFFT
#     uk_inter = np.zeros((*ukouts_inter.shape[:-1],ksout_inter.shape[-1]))  # P(k) we're going to
#     for ind in np.ndindex(uk_inter.shape[:-1]):
#         pks = HaloModel.ks  # k's we're starting with
#         puks = ukouts_inter[ind]  # P(k) we're starting with
#         puks = puks[pks>0]
#         pks = pks[pks>0]
#         uk_inter[ind] = np.interp(ksout_inter[ind],pks,puks,left=puks[0],right=0)

#     # Take out any factors added in to get what would be the raw results of the FFT, then RFFT back
#     ukts_inter = uk_inter * kts_inter
#     integrand_inter = -ukts_inter/((xs_inter[-1]-xs_inter[0])/xs_inter.size)
#     Pex_inter = 2*np.fft.irfft(integrand_inter*1j, xs_inter.size)/xs_inter
#     del kts_inter, ksout_inter, uk_inter, ukts_inter, integrand_inter

#     # Now interpolate P(x) with RFFT x's to the actual x array we want
#     xs_in = np.geomspace(xmin_in, xmax_in, nxs_in)  # x's we're going to
#     Pex_in = np.zeros((*Pex_inter.shape[:-1],xs_in.size))  # the profile we're going to
#     for ind in np.ndindex(Pex_in.shape[:-1]):
#         pks = xs_inter  # x's we're starting with
#         puks = Pex_inter[ind]  # P(x) we're starting with
#         puks = puks[pks>0]
#         pks = pks[pks>0]
#         Pex_in[ind] = np.interp(xs_in,pks,puks,left=puks[0],right=0)

#     del xs_inter, Pex_inter
#     return xs_in, Pex_in






def Pex_rev_inter_FFT2(ukouts_inter, xmin_in, xmax_in, nxs_in, r200_inter, HaloModel):
    # Create an array of x values we want out of our RFFT, must be equally spaced
    xs_inter = np.linspace(0, 2*xmax_in, int(2*(2*xmax_in/xmin_in))+1)[1:]
    # Make the k's that will produce those xs
    kts_inter = np.fft.rfftfreq(xs_inter.size, ((xs_inter[-1]-xs_inter[0])/xs_inter.size)) *2*np.pi 
    # Scale the k's the same way that the halo model k's are scaled for proper interpolation
    if ukouts_inter.ndim==2: ksout_inter = kts_inter/r200_inter[...,None]/(1+HaloModel.zs[:,None])
    elif ukouts_inter.ndim==3: ksout_inter = kts_inter/r200_inter[...,None]/(1+HaloModel.zs[:,None,None])

    # Interpolate P(k) with hmvec k's to P(k) with k's that can be used in the RFFT
    uk_inter = np.zeros((*ukouts_inter.shape[:-1],ksout_inter.shape[-1]))  # P(k) we're going to
    for ind in np.ndindex(uk_inter.shape[:-1]):
        pks = HaloModel.ks  # k's we're starting with
        puks = ukouts_inter[ind]  # P(k) we're starting with
        puks = puks[pks>0]
        pks = pks[pks>0]
        uk_inter[ind] = np.interp(ksout_inter[ind],pks,puks,left=puks[0],right=0)

    # Take out any factors added in to get what would be the raw results of the FFT, then RFFT back
    ukts_inter = uk_inter * kts_inter
    integrand_inter = -ukts_inter/((xs_inter[-1]-xs_inter[0])/xs_inter.size)
    Pex_inter = 2*np.fft.irfft(integrand_inter*1j, xs_inter.size)/xs_inter
    del kts_inter, ksout_inter, uk_inter, ukts_inter, integrand_inter

    # Now interpolate P(x) with RFFT x's to the actual x array we want
    xs_in = np.geomspace(xmin_in, xmax_in, nxs_in)  # x's we're going to
    Pex_in = np.zeros((*Pex_inter.shape[:-1],xs_in.size))  # the profile we're going to
    for ind in np.ndindex(Pex_in.shape[:-1]):
        pks = xs_inter  # x's we're starting with
        puks = Pex_inter[ind]  # P(x) we're starting with
        puks = puks[pks>0]
        pks = pks[pks>0]
        Pex_in[ind] = np.interp(xs_in,pks,puks,left=puks[0],right=0)

    del xs_inter, Pex_inter
    return xs_in, Pex_in





def Pth_rev2(P, results, xmin, xmax, nxs, HaloModel, cmass_ws=False):
    Pek = results[P]/unitconversion(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws)[..., None]
    if P == "Pgy_1h":
        integrand = HaloModel.nzm[..., None] * HaloModel._get_hod(results['HOD_Name'])
        Pek = Pek/np.trapz(integrand, HaloModel.ms, axis=-2)/(1-np.exp(-(HaloModel.ks/HaloModel.p['kstar_damping'])**2.))
    xs_rev, Pex_rev = Pex_rev_inter_FFT2(Pek, xmin, xmax, nxs, r200_mean(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws), HaloModel)
    rs_rev = xs_rev[None, ...]*r200_mean(results['HOD_Name'], HaloModel, cmass_ws=cmass_ws)[..., None]
    Pthx_rev = Pex_rev/HaloModel.eFrac/(3.086e24/1.989e33)
    return rs_rev, Pthx_rev






















def Pth_mean(results, xs, HaloModel, param_override=None, cmass_ws=False):
    family='pres'
    pparams = {}
    pparams['battaglia_pres_gamma'] = HaloModel.p['battaglia_pres_gamma']
    pparams['battaglia_pres_alpha'] = HaloModel.p['battaglia_pres_alpha']
    pparams.update(hm.battaglia_defaults[family])
    
    if param_override is not None:
        for key in param_override.keys():
            if key=='battaglia_gas_gamma': pparams[key] = param_override[key]
            elif key in hm.battaglia_defaults[family]: pparams[key] = param_override[key]
            else: pass

    presprof = hm.P_e_generic_x(xs, HaloModel.m200critz[...,None],HaloModel.r200critz[...,None],HaloModel.zs[:,None,None],HaloModel.omb,HaloModel.omm,HaloModel.rhocritz[...,None,None],
                            alpha=pparams['battaglia_pres_alpha'],gamma=pparams['battaglia_pres_gamma'],
                            P0_A0=pparams['P0_A0'],P0_alpham=pparams['P0_alpham'],P0_alphaz=pparams['P0_alphaz'],
                            xc_A0=pparams['xc_A0'],xc_alpham=pparams['xc_alpham'],xc_alphaz=pparams['xc_alphaz'],
                            beta_A0=pparams['beta_A0'],beta_alpham=pparams['beta_alpham'],beta_alphaz=pparams['beta_alphaz'])
    
    hod = HaloModel._get_hod(results['HOD_Name'])[..., 0]
    return np.trapz(presprof/HaloModel.eFrac/(3.086e24/1.989e33)*HaloModel.nzm[..., None]*hod[..., None], HaloModel.ms, axis=1)\
            /np.trapz(HaloModel.nzm*hod, HaloModel.ms, axis=1)[..., None]


# def Pth_mean_so(results, rs, HaloModel):
#     xs = rs/r200_mean(results['HOD_Name'], HaloModel)[..., None]
#     p = HaloModel._get_hod(results['HOD_Name'])[..., 0]*
#     pth = presFunc(xs[:, None, :])/eFrac/(3.086e24/1.989e33)
#     pths_av=[]
#     for i in range(xs.size):
#         for j in range(HaloModel.zs.size):
#             pth.append(np.average(pth, weights=p, axis=0))
#     return 