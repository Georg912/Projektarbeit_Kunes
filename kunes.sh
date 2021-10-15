#!/bin/bash

cd Projectarbeit_Kunes/

firefox --new-window https://www.wolframcloud.com/ &

code -r ../Projectarbeit_Kunes

jupyter lab ../Projectarbeit_Kunes &

#code -r Dynamical_Systems.ipynb

#soffice --calc Concepts_Slides/concepts_1.pptx &
evince Concepts_Slides/concepts_3.pdf &

evince Mathematica_Slides/CCMP_math.pdf &

exit
