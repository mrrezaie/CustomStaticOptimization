# cunstom Static Optimization 
WIP

The input files are data associated with Rajagopal model's walking simulation.

Typical Static Optimization, i.e., minimize the sum of squared muscle activation, with and without weighted cost function:

Muscles activity
![sample](activity3.png)

Joint contact force
![sample](KJCF3.png)

# Equations
$$ PCSA = {MIF \over 60} $$

$$ V = OFL * PCSA $$

$$ R = {{OFL * \cos \alpha} \over {OFL * \cos \alpha + TSL}} $$

$$ F_0 = min\sum_{i=1}^{N muscles} a_i^2 $$

$$ F_1 = min \sum_{i=1}^{N muscles} V_i * a_i^2 $$

$$ F_2 = min \sum_{i=1}^{N muscles} {a_i^2 \over R_i} $$

$$ F_3 = min \sum_{i=1}^{N muscles} {V_i \over R_i} * a_i^2 $$

PCSA: physiological cross-sectional area

MIF: maximum isometric force

OFL: optimal fiber length

TSL: tendon slack length

$`\alpha`$: pennation angle

$`OFL * \cos \alpha`$: fiber length along tendon

$`OFL * \cos \alpha + TSL`$: muscle-tendon length

V: muscle volume

R: muscle to muscle-tendon length ratio

F0: typical cost function

F1: muscle volume-weighted cost function

F2: 1/ratio-weighted cost function

F3: muscle volume/ratio-weighted cost function