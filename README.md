# Photometry

This repository constitutes the Python, CL and IRAF codes that act as the pipeline to perform relative photometry on TransNeptunian Objects using IRAF. 

## A quick guide to use the pipeline:

The first and obvious step would be to obtain the FITS images. These codes are developed on images obtained using Canada-France-Hawaii Telescope under the CLASSY (The Classical and Large-a Distant Solar SYstem) Survey. So the pipeline only works once you have the calibrated images from the telescope/survey team. 

The work is divided into two stages:
1) Initial Stage - Preprocessing data.
2) Photometry Stage - Pipeline to get results from CSV files.

The pipeline has some computer requirements: The latest version of IRAF, Python 3 (v3.11.5), DS9, Ximtool, Jupyter Notebook and good RAM based on the nights and data you are working with. 
   
The initial stage consists of 9 Steps/files named Step*: where * goes from 1 to 9. The 1st step starts with full Telescope images. One starts by displaying images in DS9 to identify the target object. There are two different approaches to this: 1) The team informs the user about where to find the object using RA and Dec. 2) Display the full CCD mosaic on DS9 and repeat for several exposures. Switch back and forth and find moving targets at around moving at ~3 arcsec per hour speed. Once you have figured out the CCD number, the step 1 to 5 will walk you through how to get the txt values that contain the magnitude value and other data of the TNO obtained through IRAF.

The rest of the steps are performed on Jupyter Notebook. For my project, I decided to manually calculate uncertainty on each magnitude value but you can use the one returned from IRAF as well as long as they are within the expected range (Consult an expert for this!). Steps 8, 9 and 9v2 will help you obtain the delta magnitude of TNO compared to a star where v2 is for the data with INDEF values that get converted to NaN.

Once you have the csv files, move the .py files to the folder where csv files are, and then all you need to do is call the .py file based on what model you want you use (I prefer Jupyter Notebook for this). Here's an example of how to do this:


```
import pandas as pd # import pandas to read the csv
from S1_model_fitting_modified_forcedpeak import analyze_light_curve


indef = [2] #Frames which have NaN values now
cosmicrays = [4,5,6]  # Frames which you have inspected to have cosmic rays entering the TNO aperture
star_pollution = list(range(12,33))  # Frames where the TNO aperture gets polluted by a star or galaxy
bad_frames = [56,83,86] # Frames that are noisy
rows_to_skip = indef+cosmicrays+star_pollution+bad_frames

# define the data interval so here these three present the otime range for three nightd
intervals = [
    (0, 6),(92,100),(116,124)
]

data = pd.read_csv('tno_star1.csv',skiprows=rows_to_skip)
time = data['OTIME']
delta_mag = data['DELTAMAG7']
merr = data['DELTAERR7']

analyze_light_curve(time, delta_mag, merr)

```

One can use any pipeline provided based on the task requirement. Some of the pipelines I have are:
- Single sine curve model
- Single sine curve model with forced period range on the periodogram
- Single sine curve model with forced period range on the periodogram and a predictor for future data
- Single sine curve model with forced period range on the periodogram and a single and double phase fold (latest and prefered model)
- Double sine curve model
- Double sine curve model with forced period range on the periodogram
- Double sine curve model with forced period range on the periodogram and a predictor for future data
- Double sine curve model with forced period range on the periodogram and a single and double phase fold (latest and prefered model)


This repository is still under editing so please contact me at simran@phas.ubc.ca if you need the latest version of if you have any questions. Thank you!
