-------------------------
###### Expected Sarsa
--------------------

Implementation of expected sarsa in continuous state and action space.  
Work in progress ...


-------------------------
###### Installation 
--------------------
 Follow these steps: 
 ```
   * clone the repository
   * cd Expected_sarsa_continuous_space  #  ( to Change directory to Expected_sarsa_continuous_space )
   * pip install -e . 
 (-e : Install a project in editable mode  from a local project path or a VCS url , "." signifies present directory )

```

-------------------------
###### To-Do List
--------------------
 - [x] use LSTM to model Actor (action predictor)   -- LSTM models MDP pretty well 
 - [x] save sample action space data 
 - [ ] pre-train LSTM (actor network) (LSTM takes longer to train why not pretrain.) 
 - [ ] Gaussian Mixture model (Maximum Likelihood Estimate approach) for probabilistic distribution evaluation of preicted values from LSTM.
 - [ ] Weight action values by the probabilities of MLE
 
 -------------------------
 #### Note on GMM-MLE
 -------------------------

![equation](https://latex.codecogs.com/gif.latex?%24%24%5Cmathbf%7Bz%7D%20%3D%20%24%24%20%5Ctext%7BMultinomial%20Gaussian%20Mixture%20for%20an%20action%20space%20E.g%20Throttle%20action%20space%20%7D%20%5Cnewline)

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BWe%20use%20MLE-Gradient-descent%20to%20estimate%20the%20parameters%20of%20the%20mixtures.%20%7D%20%5Cnewline%20%5Ctext%7BFor%20Gaussian%20mixture%2C%20this%20implies%20the%20means%20and%20standard%20deviations%20of%20the%20mixture%20components.%7D%20%5Cnewline)

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BGiven%20a%20new%20point%2C%20%7D%20%24%5Ctextbf%7Bx%7D%24%20%5Ctext%7B%2C%20infer%20which%20component%20of%20the%20mixture%2C%20%7D%20%24%5Cmathb%7Bz%7D%24%20%5Ctext%7B%2C%20it%20is%20likely%20to%20belong.%20That%20is%20%3A%20%7D%20%24%5Cmathb%7BP%7D%28%5Cmathbf%7Bz%7D%5Cvert%20x%29%24%20.%20%5Cnewline)

![equation](https://latex.codecogs.com/gif.latex?%24%5Cmathb%7BP%7D%28%5Cmathbf%7Bz%3D1%7D%5Cvert%20x%29%20%3D%20%28%5Cmathb%7BP%7D%28z%3D1%29%20*%20P%28x%20%5Cvert%20z%3D1%29%29%20/%20%5Csum_i%20P%28z%3Di%29%20P%20%28x%20%5Cvert%20z%3D%20i%29%24%20%5Cnewline%20%5Ctext%20%7BThat%20is%2C%20we%20compute%20the%20posterior%20inference%20that%20x%20is%20from%20first%20component%28z%3D1%29.%20i%20%3D%20number%20of%20components%20in%20the%20mixture.%7D)


![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BThen%20%7D%20%24%5Cmathb%7BP%7D%28%5Cmathbf%7Bx%7D%5Cvert%20z%29%24%20%5Ctext%7B%20would%20be%20evaluated%20using%20the%20parameters%20of%20the%20most%20likely%20component.%20%7D%20%5Cnewline%20%5Ctext%7B%28That%20is%2Cthe%20parameters%20of%20the%20most%20likely%20component%2C%20%7D%20%24%5Cmathb%7BP%7D%28%5Cmathbf%7Bz%3Di%7D%5Cvert%20x%29%24%20%5Ctext%7B%2C%20computed%20above.%7D%29 )



