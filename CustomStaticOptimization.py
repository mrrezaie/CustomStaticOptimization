# %%
import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, Bounds, LinearConstraint
import cvxpy as cp

def readExp(file, sep='\t', unit=None):
	''' Read OpenSim STO and MOT files and return a dict[keys=labels, vaules=data]
	or any other format that the headers are separated from labels and data by 'endheader' line.
	unit = degree/radian'''
	with open(file, mode='r') as f:
		while True:
			line = f.readline()

			if line.startswith('nRows'):
				row = int(line.split('=')[-1])
			elif line.startswith('datarows'): # old MOT files
				row = int(line.split(' ')[-1])
			# else: row = None

			if line.startswith('nColumns'):
				column = int(line.split('=')[-1])
			elif line.startswith('datacolumns'):  # old MOT files
				column = int(line.split(' ')[-1])
			# else: column = None

			if line.startswith('inDegrees'):
				inDegrees = line.split('=')[-1].split('\n')[0]

			if line.startswith('endheader'):
				line = f.readline() # get labels
				break
		labels = line.split('\n')[0].split(sep)
		a = f.readlines()

	# if row==None or column==None:
	# 	data = np.empty((len(a), len(labels)), dtype=np.float64)
	# else:
	data = np.empty((row, column), dtype=np.float64)

	for ii,i in enumerate(a): # rows
		try:
			for jj,j in enumerate(i.split(sep)): # columns
				data[ii,jj] = float(j)
		except: pass
	del a
	
	# convert the units from degrees to radians
	if unit=='radian' and inDegrees=='yes': # convert degrees to radians
		data[:, 1:] = np.deg2rad(data[:, 1:]) # exclude the first column (time)
	elif unit=='degree' and inDegrees=='no': # convert radians to degrees
		data[:, 1:] = np.rad2deg(data[:, 1:]) # exclude the first column (time)
	
	# convert to dict
	data2 = dict() 
	for i,ii in enumerate(labels):
		data2[ii] = data[:,i]
	return data2


m = readExp('inverse_dynamics.sto') #'' # moment
q = readExp('Kinematics_q.sto', unit='radian') # angle in radian
u = readExp('Kinematics_u.sto', unit='radian') # velocity in radian/s
model = osim.Model('subject01_simbody.osim')

time = q['time']
timeBool = np.logical_and(0.62<=time, time<=1.404)
time = time[timeBool]
frame = len(time)

del q['time']
del u['time']
del m['time']

# adjust the order of moment columns to match the kinematics
m2 = dict()
for i in list(q.keys()):
	try:
		m2[i] = m[i+'_moment']
	except:
		m2[i] = m[i+'_force']

# plt.plot(m['knee_angle_r_moment'])
# plt.plot(m2['knee_angle_r'])
# plt.show(block=False)

m = m2 
del m2

for i in list(q.keys()):
	q[i] = q[i][timeBool]
	u[i] = u[i][timeBool]
	m[i] = m[i][timeBool]

state = model.initSystem()
muscles = model.updMuscles()
nameMuscles = [i.getName() for i in muscles]
nMuscles = muscles.getSize()
coordinates = model.getCoordinateSet()
nCoordinates = coordinates.getSize()
nameCoordinates = [i.getName() for i in coordinates]

# https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Muscle.html#ac2ddf201cb1a2263f9a1c6baa3d7f314
MIF= np.empty(nMuscles) # muscle maximum isometric force
L  = np.empty((frame, nMuscles)) # muscle length
FL = np.empty((frame, nMuscles)) # fiber length
TL = np.empty((frame, nMuscles)) # tendon length
OFL= np.empty(nMuscles) # optimal fiber length
TSL= np.empty(nMuscles) # tendon slack length
S  = np.empty((frame, nMuscles)) # muscle strength
FF = np.empty((frame, nMuscles)) # fiber force
AFF= np.empty((frame, nMuscles)) # active fiber force
PFF= np.empty((frame, nMuscles)) # passive fiber force
TF = np.empty((frame, nMuscles)) # tendon force
MA = np.empty((frame, nCoordinates, nMuscles)) # muscle moment arm

# store muscle parameters
for i in range(frame):
	print('Muscle Parameters ... at', time[i])
	Q = osim.Vector(); Q.resize(nCoordinates)
	U = osim.Vector(); U.resize(nCoordinates)
	for jj in range(nCoordinates):
		Q.set(jj, q[nameCoordinates[jj]][i])
		U.set(jj, u[nameCoordinates[jj]][i])
	state.setQ(Q)
	state.setU(U)
	model.realizeDynamics(state)

	for j in range(nMuscles):
		muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(j))
		muscle.set_ignore_activation_dynamics(True) # disable activation dynamics
		muscle.set_ignore_tendon_compliance(True) # disable tendon compliance
		model.equilibrateMuscles(state)

		L[i,j]  = muscle.getLength(state)
		FL[i,j] = muscle.getFiberLength(state)
		TL[i,j] = muscle.getTendonLength(state)
		OFL[j]  = muscle.getOptimalFiberLength()
		TSL[j]  = muscle.getTendonSlackLength()
		TL[i,j]   = muscle.getTendonLength(state)
		MIF[j]  = muscle.getMaxIsometricForce()
		FF[i,j] = muscle.getFiberForce(state)
		AFF[i,j]= muscle.getActiveFiberForce(state)
		PFF[i,j]= muscle.getPassiveFiberForce(state)
		TF[i,j] = muscle.getTendonForce(state)

		muscle.setActivation(state, 1)
		S[i,j] = muscle.getFiberForce(state)

		for k in range(nCoordinates):
			coordinate = coordinates.get(k)
			coordinate.setValue(state, q[nameCoordinates[k]][i])
			MA[i,k,j] = muscle.computeMomentArm(state, coordinate)


# plt.plot(MA[:, nameCoordinates.index('knee_angle_r'), nameMuscles.index('rect_fem_r')])
# plt.show(block=False)

# plt.plot(L[:, nameMuscles.index('rect_fem_r')], label='muscle length')
# plt.plot(FL[:, nameMuscles.index('rect_fem_r')], label='fiber length')
# plt.plot(TL[:, nameMuscles.index('rect_fem_r')], label='tendon length')
# plt.legend()
# plt.show(block=False)

# plt.plot(S[:, nameMuscles.index('rect_fem_r')], label='strength')
# plt.plot(FF[:, nameMuscles.index('rect_fem_r')], label='fiber force', marker='o')
# plt.plot(AFF[:, nameMuscles.index('rect_fem_r')], label='active fiber force')
# plt.plot(PFF[:, nameMuscles.index('rect_fem_r')], label='passive fiber force')
# plt.plot(TF[:, nameMuscles.index('rect_fem_r')], label='tendon fiber force', marker='+')
# plt.plot(AFF[:, nameMuscles.index('rect_fem_r')] + PFF[:, nameMuscles.index('rect_fem_r')], label='active+passive fiber force')
# plt.legend()
# plt.show(block=False)

# plt.plot(L[:, nameMuscles.index('rect_fem_r')], label='muscle length')
# plt.plot(FL[:, nameMuscles.index('rect_fem_r')], label='fiber length')
# plt.plot(TL[:, nameMuscles.index('rect_fem_r')], label='tendon length')
# plt.axhline(OFL[nameMuscles.index('rect_fem_r')], label='optimal fiber length', color='c')
# plt.axhline(TSL[nameMuscles.index('rect_fem_r')], label='tendon slack length', color='r')
# plt.legend()
# plt.show(block=False)


#####################################################################
# %% OPTIMIZATION
#####################################################################
# https://stackoverflow.com/questions/37791680/scipy-optimize-minimize-slsqp-with-linear-constraints-fails

# remove pelvis (and any other?) coordinates from the optimization
cBool = [True] * nCoordinates # coordinates boolean
for i in range(nCoordinates):
	if nameCoordinates[i].startswith('pelvis'):
		cBool[i] = False

activity = np.empty((frame, nMuscles)) # muscle activity
force = np.empty((frame, nMuscles)) # muscle force

def objFun(a):  # sum of squared muscle activation
	return np.sum(a**2) # volume 

'''
momentArm is a matrix (nCoordinate, nMuscles)
moment is an array (nCoordinates)
strength is an array (nMuscles)
equality constraint: to be zero
inequality constraint: to be non-negative (greater than zero)'''

def eqConstraint(a):  # A.dot(x) - b
	# return np.sum(momentArm[n,:]*strength*a) - moment[n]
	return momentArm.dot(strength*a) - moment # np.sum(momentArm*strength*a, axis=1) - moment

init = np.zeros(nMuscles) + 0.1 # initial guess of muscle activity (0.125)
lb = np.zeros(nMuscles) # lower bound (0)
ub = np.ones(nMuscles) # upper bound (1)
constraints = {'type':'eq', 'fun': eqConstraint}
# constraints = [{'type':'eq', 'fun': lambda z,n=i: eqConstraint(z, n)} for i in range(sum(cBool))]

for i in range(frame): #frame
	print(f'Optimization ... {i+1}/{len(time)} ({round(time[i],3)})')

	moment = np.vstack(list(m.values())).T[i,cBool] # (nCoordinates)
	momentArm = MA[i,cBool,:] # (nCoordinate, nMuscles)
	strength = S[i,:] # (nMuscles)
	# strength = MIF # constant strength
	# length = FL[i,:] # (nMuscles)
	# volume = strength * length # (nMuscles)

	# ######################### scipy
	out = minimize(objFun, x0=init, method='SLSQP', bounds=Bounds(lb,ub), constraints=constraints, options={'maxiter':500}, tol=1e-08)
	print(f"\t\tfun={round(out['fun'],3)} success={out['success']}")
	if out['status'] != 0: print(f"\t\t\tmessage: {out['message']}")

	activity[i,:] = out['x']
	force[i,:] = strength * out['x']

	# ######################### cvxpy (alternative to scipy)
	# f = cp.Variable(nMuscles)
	# prob = cp.Problem(objective = cp.Minimize(cp.sum_squares(f)),
	# 				  constraints = [0<=f, f<=strength,
	# 				  				 momentArm @ f == moment])
	# prob.solve()
	# print("status:", prob.status)
	# print("optimal value", prob.value)
	# print(f.value)

	# force[i,:] = f.value
	# activity[i,:] = f.value / strength

# write muscles activity and force to mot files
head = f"static optimization\nversion=1\nnRows={activity.shape[0]}\nnColumns={1+activity.shape[1]}\ninDegrees=yes\nendheader\n" + 'time\t' + '\t'.join(nameMuscles)
np.savetxt('muscle activity.mot', np.hstack((time.reshape((-1,1)),activity)), fmt='%.6f', delimiter='\t', newline='\n', header=head, comments='')
np.savetxt('muscle force.mot', np.hstack((time.reshape((-1,1)),force)), fmt='%.6f', delimiter='\t', newline='\n', header=head, comments='')

# plt.plot(activity[:, nameMuscles.index('soleus_r')], label='soleus')
# plt.plot(activity[:, nameMuscles.index('rect_fem_r')], label='rectus femoris')
# plt.legend()
# plt.show(block=False)

plt.plot(time, force[:, nameMuscles.index('soleus_r')])
plt.show(block=False)