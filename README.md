LKJ.jl
======

*A Julia package implementing the Lewandowski, Kurowicka and Joe (LKJ) probability distribution on the space of correlation matrices.*

[![Build Status](https://travis-ci.org/mbrueckner/LKJ.jl.svg?branch=master)](https://travis-ci.org/mbrueckner/LKJ.jl)

The probability density function of a d-dimensional LKJ-distribution (''d \\geq 2'') with parameter ''\\boldsymbol{\\eta} \\geq 1'' is:
```math
f(\\mathbf{A}; \\boldsymbol{\\eta}) = c_d |det(A)|^{\\eta-1}
```
where ''\\mathbf{A}'' is ''d \\times d'' correlation matrix and ''c_d'' is a normalization constant depending on the dimension 'd'.

References
----------
Lewandowski, Kurowicka, Joe (2009): Generating random correlation matrices based on vines and extended onion method. Journal of Multivariate Analysis. (DOI 10.1016/j.jmva.2009.04.008)