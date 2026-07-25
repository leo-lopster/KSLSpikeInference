# KSLSpikeInference

Spike inference toolkit for two-photon microscope imagery at KSL Lab. This toolkit uses the CASCADE model by Rupprecht et al. (2021), originally published on Nature Neuroscience.

## Workflow
<div>
  
1. **Working with .npy files**<br>
Use `process_using_napari.ipynb` for preprocessing, ROI extraction, and dF/F trace extraction directly from .npy files obtained on microscopes. Alternatively, modify `SlideBook_WF_Timelapse_Dask_Load.ipynb` for image processing (Credit: Li-Ting, KSL Lab) if you have SlideBook installed locally.

</div>
<div>
  
  2. **Working with extracted traces**<br>
  Under `/CascadeTorch/scripts`, `model_tester.py` and `model_tester_batch.py` can be found with their capabilites listed at filehead. Before spike inference can be performed, note that traces must be preprocessed to `.csv` files with the following format (designated by `model_tester.py`). Additional columns are allowed but will be ignored by python scripts.

  <div align="center">
    
  | Trace_ROI1 | Trace_ROI2 | Trace_ROI3 |
  | -------- | -------- | -------- |
  | 0.0001    | 0.0003   | 0.0005   |
  | 0.0002    | 0.0004   | 0.0006   |

  *Standard format for calcium traces (ΔF/F).*
  </div>
  

  Resampling of traces to the sampling rate of the selected pretrained model is done by `model_tester.py`. Simply indicate sampling frame rate of input traces with the option `--input-fps` when executing `model_tester.py` or `model_tester_batch.py`. 
  Some parsing examples can be found at `/CascadeTorch/scripts/parse_csv_Mark.py` and `CascadeTorch/scripts/parse_npy_LT.py`  

</div>
