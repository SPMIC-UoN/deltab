# Brain age modelling and prediction

This module supports training of a model to predict age from a set of IDPs or
other measures. The model can then be used to determine a subject's 'brain age'
from a set of the same measured, and hence a 'brain delta' that expresses
how much older or younger their brain appears compared to their actual age.

Based on method described in Smith et al (Neuroimage 200 (2019) 528–539)

https://doi.org/10.1016/j.neuroimage.2019.06.017

Multiple models are defined in this paper including a simple model that has been
shown to be biased when the model retains age dependence within the prediction.
A corrected 'unbiased' model can be used to address this issue. In addition
quadratic age dependence can be modelled and an alternate approach in which
age is regarded as a predictor of IDPs rather than the other way round can
be used.

Reproduction of simulation results can be found in the examples/ directory
