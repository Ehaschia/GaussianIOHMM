# Mixture Gaussian Input Output Hidden Markov Model

### TODO
* Implement placeholder for mixture gaussian case.
* Unit Test for Mixture Gaussian Sequence Labeling
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


Next week (12/17-12/24) need to do:
* Implement IOHMM
* ~~fix grid search tunner bug~~
* ~~mixture gaussian unit test~~

### Issue

The current issue under issue dictionary

*   sequence labeling forward issue: 

    Before fixing, we used `forward=True` in `gaussian_multiply_integral` function during backward.