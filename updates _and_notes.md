# Notes and updates

# Notes
    
- batch in KAN.train 

- SVD method is added to calculate the coef. 

- nan in loss can be caused by nan in KAN.grid during backpropogation and KANLayer.coef during calculating due to the ill-conditioned matrix of mat during the initialization. 

# Updates
## 05/20/2024
- KAN.py
    - forword: 
        - Add input node mask. parameter for the changes of input dim after pruning
        codes add to avoid nan in bais

    - train:
        - update the default device parameter. device = None
        Add input node mask. parameter for the changes of input dim after pruning

    - prune:
        - Add 'outfrac' mode. Prune the model by the fraction of the outflow importance

    - add method:
        - prune_reset:
            - method to reset the masked nodes and edges if the pruning results are not staistifed.

- KANLayer.py
    - initialize_grid_from_parent
        - add coef_method from self
    - __init__
        - delete nn.parameter for self.grid: grid could be nan during backpropogation if it is in nn.parameter


