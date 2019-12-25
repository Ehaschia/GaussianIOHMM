# Mixture Gaussian Input Output Hidden Markov Model

### TODO
* ~~Implement placeholder for mixture gaussian case.~~
* Unit Test for Mixture Gaussian Sequence Labeling.
* ~~grid search tunner~~:
    * ~~unknown error: not running but the thread is not ready.~~
    * ~~clean code~~
    * ~~qbs generator and submitter~~
* ~~Implement NER part~~
* Implement new train strategy: first training mu. Then update variance
    * it seems work. Need to investigate the best strategy. 
* Investigate the benefit of our graph model. (Like known several truth label)
* Investigate how to express interpretability.
* Implement IOHMM
* Investigate the use of inverse wishart prior
* accelerate:
    * No update var: pre calculate inverse and store.
    * Update Var: implement pseudo inverse like [Moore–Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

Next week (12/25-12/31) need to do:
* Implement IOHMM
* Investigate the use of inverse wishart prior

### Issue

The current issues that are under processing dictionary

1.   ~~sequence labeling forward issue:~~ 
      
      Before fixed, we used `forward=True` in `gaussian_multiply_integral` function during backward.

2.  ~~Loss increase during training.~~
    
    After fixed issue 1, the loss changed sharply and may increase during training.
    
    **Fixed**: Due to in and out var init is too small. Re-implement random init part.
3. Current loss also shown increase during training. Current temporary fix is not update variance. 