Step 0: Environment
Clone github
conda env create -f environment.yml
or
conda env create -f cross_platform_environment.yml
The latter may work 

Step 1: Encoding
Sample code to open the template datum:
import pandas as pd
from utils import load_pickle
datum = load_pickle('../template_datum.pkl')
df = pd.DataFrame.from_dict(datum)

Replace embeddings in template_datum with your own. 

Put embeddings in:	 /data/pickles/

In Makefile: 
Change PKL_IDENTIFIER to match your embeddings
Change NUM_LAYERS, LAYERS as necessary
Change OUT_NAME
Change CMD based on your setup. 
	May need your own “submit.sh” file for submitting multiple jobs at once

Then run: 
make run-layered-sig-encoding

Verification:
Sometimes not all encodings run. To check, run:
make verify-encoding

Outputs layers that are missing files and the numbers of files that are complete. 
Re-run those layers and verify again. 

Getting Plots:
Run: 
make plot-layered-sig-encoding

Output: 
Per roi:
Inverted u plot
Encoding plot
Scaled encoding plot
Lag layer plot

