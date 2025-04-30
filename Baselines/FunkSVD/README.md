# FunkSVD

To train and evaluate the model run the ```FunkSVD.ipynb``` notebook in this subfolder.

There is no need to run the cells related to grid search:
It suffices for the first part (The one without implicit values), to of course run all the cells in the beginning with helper functions, and then directly run the last cell in the section marked with a "REPRODUCE" comment at it's beginning. 
As for the second part (with implicit values), there is no grid search as we directly train with the already found best parameters from the first part.

# Requirements

- Python 3.8 or newer  
- pandas  
- numpy  
- matplotlib  
- scikit-surprise  