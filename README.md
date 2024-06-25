# Photometry

This repository constitutes the Python, CL and IRAF codes that act as the pipeline to perform relative photometry on TransNeptunian Objects using IRAF. 

## A quick guide to use the pipeline:

The first and obvious step would be to obtain the FITS images. These codes are developed on images obtained using Canada-France-Hawaii Telescope under the CLASSY (The Classical and Large-a Distant Solar SYstem) Survey. So the pipeline only works once you have the calibrated images from the telescope/survey team. 

The work is divided into two stages:
1) Initial Stage - Preprocessing data.
2) Photometry Stage - Pipeline to get results from CSV files.

The pipeline has some computer requirements: The latest version of IRAF, Python 3 (v3.11.5), DS9, Ximtool, Jupyter Notebook and good RAM based on the nights and data you are working with. 
   
The initial stage consists of 9 Steps/files named Step*: where * goes from 1 to 9. The 1st step starts with full Telescope images. One starts by displaying images in DS9 to identify the target object. There are two different approaches to this: 1) The team informs the user about where to find the object using RA and Dec. 2) Display the full CCD mosaic on DS9 and repeat for several exposures. Switch back and forth and find moving targets at around moving at ~3 arcsec per hour speed. Once you have figured out the CCD number, the step 1 to 5 will walk you through how to get the txt values that contain the magnitude value and other data of the TNO obtained through IRAF.

The rest of the steps are performed on Jupyter Notebook. For my project, I decided to manually calculate uncertainty on each magnitude value but you can use the one returned from IRAF as well as long as they are within the expected range (Consult an expert for this!). Steps 8, 9 and 9v2 will help you obtain the delta magnitude of TNO compared to a star where v2 is for the data with INDEF values.




