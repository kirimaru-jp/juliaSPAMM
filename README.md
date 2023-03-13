# juliaSPAMM
Population assessment of harp and hooded seals in the West Ice (along the coast of Greenland) and the East Ice.

It is a Julia implementation of [rSPAMM](https://github.com/NorskRegnesentral/rSPAMM) package for seal population dynamics. 
As a just-in-time language, we do not need to develop models in 2 different languages R and C++ like in rSPAMM, but only one as Julia. 

Julia also strongly support automatic differentation (both forward and reverse mode - used in Deep Learning), while rSPAMM requires user to use Template Model Builder to do that. Since TMB is a C++ framework, one needs to know C++ to develop population models.
