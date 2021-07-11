#!/bin/bash

cd Projectarbeit_Kunes/

firefox --new-window https://www.wolframcloud.com/ &

code -r ../Projectarbeit_Kunes

code -r System_Dynamics.ipynb

soffice --calc Concepts_Slides/concepts_1.pptx &

evince Programms_Mathematica/CCMP_math.pdf &

exit
