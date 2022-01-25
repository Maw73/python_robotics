# Inverse kinematic task
The task is to solve the inverse kinematic task for a general 6R mechanism using a general Grobner Basis 
computation.

The following steps are needed:
- Providing a mechanism with the following Denavit-Hartenberg parameters (sin(alpha), cos(alpha), a, d)
- Providing the end effector pose as a rational matrix 
- Formulating the algebraic equations for the inverse kinematic task
- Finding a Grobner basis for the IKT equations
- Recovering the real solutions for sines and cosines from a Groebner basis
- Recovering the joint angles from the computed sines and cosines
