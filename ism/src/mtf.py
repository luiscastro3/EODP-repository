from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff*Hdefoc*Hwfe*Hdet*Hsmear*Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/fc)
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO

        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w

        eps = 1e-6
        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fAltxx, fActxx] = np.meshgrid(fAlt, fAct, indexing='ij')
        f2D = np.sqrt(fAltxx * fAltxx + fActxx * fActxx)

        fn2D = f2D/(1/w)
        fnAlt = fAlt/(1/w)
        fnAct = fAct/(1/w)

        f_cutoff = D/(lambd*focal)
        fr2D = f2D/f_cutoff

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        #TODO

        rows, cols = fr2D.shape
        Hdiff = np.zeros((rows,cols), dtype=float)

        for i in range(rows):
            for j in range(cols):
                fr = abs(fr2D[i, j])

                if fr < 1.0:
                    Hdiff[i, j] = (2.0 / np.pi) * (np.arccos(fr) - fr * np.sqrt(1.0 - fr ** 2))
                else:
                    Hdiff[i, j] = 0.0
        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        #TODO

        x = np.pi*defocus*fr2D*(1-fr2D)

        J1 = x/2 - (x**3)/16 + (x**5)/384 - (x**7)/18432

        Hdefoc = 2*J1/x
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        #TODO

        Hwfe = np.exp(-fr2D*(1-fr2D)*(kLF*(wLF/lambd)**2+kHF*(wHF/lambd)**2))
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO

        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        #TODO

        Hsmear_1d = np.sinc(ksmear*fnAlt)

        Hsmear = np.tile(Hsmear_1d[:, np.newaxis], (1, ncolumns))
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO

        Hmotion = np.sinc(fn2D*kmotion)
        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #TODO
        # Determine the zero-frequency center indices
        idalt = nlines // 2
        idact = ncolumns // 2

        # Extract only the positive frequency side (from the center to the right)
        # Slices for ACT: varying ACT (columns), fixed ALT (lines) at center
        fnAct_pos = fnAct[idact:]
        diff_act = Hdiff[idalt, idact:]
        defoc_act = Hdefoc[idalt, idact:]
        wfe_act = Hwfe[idalt, idact:]
        det_act = Hdet[idalt, idact:]
        smear_act = Hsmear[idalt, idact:]
        motion_act = Hmotion[idalt, idact:]
        sys_act = Hsys[idalt, idact:]

        # Slices for ALT: varying ALT (lines), fixed ACT (columns) at center
        fnAlt_pos = fnAlt[idalt:]
        diff_alt = Hdiff[idalt:, idact]
        defoc_alt = Hdefoc[idalt:, idact]
        wfe_alt = Hwfe[idalt:, idact]
        det_alt = Hdet[idalt:, idact]
        smear_alt = Hsmear[idalt:, idact]
        motion_alt = Hmotion[idalt:, idact]
        sys_alt = Hsys[idalt:, idact]

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # ---------------------------------------------------------
        # 1. Plot System MTF - slice ACT
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 7))
        plt.plot(fnAct_pos, diff_act, label='Diffraction MTF', alpha=0.8)
        plt.plot(fnAct_pos, defoc_act, label='Defocus MTF', alpha=0.8)
        plt.plot(fnAct_pos, wfe_act, label='WFE Aberrations MTF', alpha=0.8)
        plt.plot(fnAct_pos, det_act, label='Detector MTF', alpha=0.8)
        plt.plot(fnAct_pos, smear_act, label='Smearing MTF', alpha=0.8)
        plt.plot(fnAct_pos, motion_act, label='Motion blur MTF', alpha=0.8)
        plt.plot(fnAct_pos, sys_act, label='System MTF', color='black',
                 linewidth=2.5)  # Thick black line for System MTF[cite: 4]

        # Nyquist frequency line at 0.5[cite: 4]
        plt.vlines(0.5, 0, 1, colors='black', linestyles='dashed', linewidth=2.5, label='f Nyquist')

        plt.title('System MTF - slice ACT', fontsize=14)  # [cite: 4]
        plt.xlabel('Spatial frequencies f/(1/w) [-]', fontsize=12)  # [cite: 4]
        plt.ylabel('MTF', fontsize=12)  # [cite: 4]
        plt.grid(True, alpha=0.6)  # Grid lines[cite: 4]
        plt.legend(loc='lower left', fontsize=9)  # Legend placement[cite: 4]

        plt.xlim(-0.02, 0.52)  # Constrain spatial frequencies around [0, 0.5][cite: 4]
        plt.ylim(-0.05, 1.05)  # Constrain MTF values around [0, 1][cite: 4]

        act_filename = os.path.join(directory, f'system_mtf_act_{band}.png')
        plt.savefig(act_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # 2. Plot System MTF - slice ALT
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 7))
        plt.plot(fnAlt_pos, diff_alt, label='Diffraction MTF', alpha=0.8)
        plt.plot(fnAlt_pos, defoc_alt, label='Defocus MTF', alpha=0.8)
        plt.plot(fnAlt_pos, wfe_alt, label='WFE Aberrations MTF', alpha=0.8)
        plt.plot(fnAlt_pos, det_alt, label='Detector MTF', alpha=0.8)
        plt.plot(fnAlt_pos, smear_alt, label='Smearing MTF', alpha=0.8)
        plt.plot(fnAlt_pos, motion_alt, label='Motion blur MTF', alpha=0.8)
        plt.plot(fnAlt_pos, sys_alt, label='System MTF', color='black',
                 linewidth=2.5)  # Thick black line for System MTF[cite: 5]

        # Nyquist frequency line at 0.5[cite: 5]
        plt.vlines(0.5, 0, 1, colors='black', linestyles='dashed', linewidth=2.5, label='f Nyquist')

        plt.title('System MTF - slice ALT', fontsize=14)  # [cite: 5]
        plt.xlabel('Spatial frequencies f/(1/w) [-]', fontsize=12)  # [cite: 5]
        plt.ylabel('MTF', fontsize=12)  # [cite: 5]
        plt.grid(True, alpha=0.6)  # Grid lines[cite: 5]
        plt.legend(loc='lower left', fontsize=9)  # Legend placement[cite: 5]

        plt.xlim(-0.02, 0.52)  # Constrain spatial frequencies around [0, 0.5][cite: 5]
        plt.ylim(-0.05, 1.05)  # Constrain MTF values around [0, 1][cite: 5]

        alt_filename = os.path.join(directory, f'system_mtf_alt_{band}.png')
        plt.savefig(alt_filename, dpi=300, bbox_inches='tight')
        plt.close()


