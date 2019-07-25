# example applications
This section provides a brief list of projects that used these tools. The code was originally written during my PhD work on 3D radiation dosimetry *via* optical CT readout of radiochromic gels, and therefore my list of example applications is all optical-CT specific. However, as there is nothing unique mathematically about optical CT as compared to x-ray CT, there are many areas where the code could be useful. 

## optical CT
In addition to using these tools for standard FBP reconstruction of optical CT data from my scanning-laser optical CT system <sup>1</sup>, there were a few other cases where these GPU-accelerated operations are useful.

### refracted rays in optical CT
Unlike x-rays, visible light is strongly refracted. Most optical CT dosimeters are imaged within an "aquarium" of matching refractive index fluid, to preserve standard CT geometry. However, for some plastic dosimeters, these fluids are cumbersome due to high viscosity and difficulty in cleaning (oil-based). Therefore it would be convenient, from a logistical sense, if these could be imaged within a lower refractive index material, using iterative CT techniques to accurately reconstruct an interior region of the cylindrical dosimeter. One of my PhD projects was to experimentally demonstrate this technique <sup>1</sup>, which had previously been published in computer simulation studies<sup>2</sup>.

I originally used the [ASTRA](https://www.astra-toolbox.com/) toolbox to perform these mismatched-index optical CT experiments, however with the addition of the "general3D" geometry option to the code, it was later used for similar experiments in our lab.

### iterative reconstruction
I used these tools to study the effects of a previously published iterative reconstruction algorithm (OSC-TV)<sup>4,5</sup> on the quantitative accuracy of reconstructions. In particular, we found that care must be taken when reconstructing very small regions of attenuation (in our context, the dose distribution from small field irradiations), as it is possible to introduce quantitative error by over-regularization, an effect that is accentuated by noisy projection data <sup>6</sup>.

## references

<sup>1</sup>K. H. Dekker, J. J. Battista, and K. J. Jordan, “Fan-beam scanning laser optical computed tomography for large volume dosimetry,” J. Phys.: Conf. Ser., vol. 847, no. 1, p. 012008, 2017.

<sup>2</sup>K. H. Dekker, J. J. Battista, and K. J. Jordan, “Optical CT imaging of solid radiochromic dosimeters in mismatched refractive index solutions using a scanning laser and large area detector,” Medical Physics, vol. 43, no. 8, pp. 4585–4597, Aug. 2016.

<sup>3</sup>S. J. Doran and D. N. B. Yatigammana, “Eliminating the need for refractive index matching in optical CT scanners for radiotherapy dosimetry: I. Concept and simulations,” Physics in Medicine and Biology, vol. 57, no. 3, pp. 665–683, Feb. 2012.

<sup>4</sup>D. Matenine, J. Mascolo-Fortin, Y. Goussard, and P. Després, “Evaluation of the OSC-TV iterative reconstruction algorithm for cone-beam optical CT,” Medical Physics, vol. 42, no. 11, pp. 6376–6386, Nov. 2015.

<sup>5</sup>D. Matenine, Y. Goussard, and P. Després, “GPU-accelerated regularized iterative reconstruction for few-view cone beam CT,” Medical Physics, vol. 42, no. 4, pp. 1505–1517, Apr. 2015.

<sup>6</sup>K. H. Dekker, J. J. Battista, and K. J. Jordan, “Technical Note: Evaluation of an iterative reconstruction algorithm for optical CT radiation dosimetry,” Medical Physics, vol. 44, no. 12, pp. 6678–6689, Dec. 2017.


