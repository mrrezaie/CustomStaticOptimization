# %%

from tqdm import tqdm
import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.signal import filtfilt, butter

osim.Logger.setLevel(4)

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
q = readExp('inverse_kinematics.mot', unit='radian') # angle in radian
cycle = [1.42, 2.16] # stance time

model = osim.Model('scaled.osim')
state = model.initSystem()
# print(model.getNumStateVariables())
muscles = model.getMuscles()
nameMuscles = [i.getName() for i in muscles]
nMuscles = muscles.getSize()
coordinates = model.getCoordinateSet()
nCoordinates = coordinates.getSize()
nameCoordinates = [i.getName() for i in coordinates]

# for i in range(nMuscles):
# 	muscle = muscles.get(i)
# 	# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(i))
# 	muscle.set_ignore_activation_dynamics(False) # disable activation dynamics
# 	muscle.set_ignore_tendon_compliance(False) # disable tendon compliance

state = model.initSystem()
# print(model.getNumStateVariables())
'''Rajagopal et al. 2016. Tendons were modeled as rigid when the tendon slack length was less than the muscle optimal fiber length.'''
'''when set_ignore_tendon_compliance is true, the fiber_length is omitted from state and 
the causes a change in state size'''


time = q['time']
timeBool = np.logical_and(cycle[0]<=time, time<=cycle[1])
time = time[timeBool]
fs = round(1/np.mean(np.diff(time))) # sampling frequency (Hz)
frame = len(time)

del q['time']
del m['time']

# adjust the order of moment columns to match the kinematics
for i in nameCoordinates:
	try:
		m[i] = m[i+'_moment']
		del m[i+'_moment']
	except:
		m[i] = m[i+'_force']
		del m[i+'_force']

# crop and filter the input signals
fc = 7 # cut-off frequency (Hz)
b,a = butter(4, 2*fc/fs, btype='lowpass', output='ba')
u = dict() # angular velocity (calculated using gradient)
for i in nameCoordinates:
	q[i] = filtfilt(b,a, q[i][timeBool], padlen=10)
	u[i] = filtfilt(b,a, np.gradient(q[i]), padlen=10)
	m[i] = filtfilt(b,a, m[i][timeBool], padlen=10)

# remove pelvis (and any other?) coordinates
cBool = [True] * nCoordinates # coordinates boolean
for i in range(nCoordinates):
	for j in ['pelvis', 'lumbar', 'beta', 'mtp', '_l']:
		if j in nameCoordinates[i]:
			cBool[i] = False

# https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Muscle.html#ac2ddf201cb1a2263f9a1c6baa3d7f314
MIF= np.empty(nMuscles) # muscle maximum isometric force
PAOFL = np.empty(nMuscles) # muscle pennation angle at optimal fiber length
PA = np.empty(nMuscles) # muscle pennation angle
CPA = np.empty((frame, nMuscles)) # cos of muscle pennation angle
L  = np.empty((frame, nMuscles)) # muscle length
FL = np.empty((frame, nMuscles)) # fiber length
FLT = np.empty((frame, nMuscles)) # fiber length along tendon
TL = np.empty((frame, nMuscles)) # tendon length
OFL= np.empty(nMuscles) # optimal fiber length
TSL= np.empty(nMuscles) # tendon slack length
S  = np.empty((frame, nMuscles)) # muscle strength (fiber force)
AS = np.empty((frame, nMuscles)) # muscle strength (active fiber force)
FF = np.empty((frame, nMuscles)) # fiber force
AFLM = np.empty((frame, nMuscles)) # active force length multiplier
FVM = np.empty((frame, nMuscles)) #  force velocity multiplier
AFF= np.empty((frame, nMuscles)) # active fiber force
PFF= np.empty((frame, nMuscles)) # passive fiber force
FFT= np.empty((frame, nMuscles)) # fiber force along tendon
AFFT= np.empty((frame, nMuscles)) # active fiber force along tendon
PFFT= np.empty((frame, nMuscles)) # passive fiber force along tendon
TF = np.empty((frame, nMuscles)) # tendon force
MA = np.empty((frame, sum(cBool), nMuscles)) # muscle moment arm

# store muscle parameters
# for i in tqdm(range(frame), desc='Muscle parameters'):
for i in range(frame):
	print('Muscle Parameters ... at', time[i])
	# Q = osim.Vector(); Q.resize(nCoordinates)
	# U = osim.Vector(); U.resize(nCoordinates)
	# for jj in range(nCoordinates):
	# 	Q.set(jj, q[nameCoordinates[jj]][i])
	# 	U.set(jj, u[nameCoordinates[jj]][i])
	# state.setQ(Q)
	# state.setU(U)
	for j in range(nCoordinates):
		model.updCoordinateSet().get(j).setValue(state, q[nameCoordinates[j]][i], False)
		model.updCoordinateSet().get(j).setSpeedValue(state, u[nameCoordinates[j]][i])
	model.assemble(state)

	# model.realizePosition(state)
	# model.realizeVelocity(state)
	model.realizeDynamics(state)
	# model.equilibrateMuscles(state)

	for j in range(nMuscles):
		# muscle = muscles.get(j)
		muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(j))
		# muscle.set_ignore_activation_dynamics(True) # disable activation dynamics
		# muscle.set_ignore_tendon_compliance(True) # disable tendon compliance
		muscle.setActivation(state, 1)
		muscle.computeEquilibrium(state)
		# model.equilibrateMuscles(state)

		# OFL[j]    = muscle.getOptimalFiberLength()
		# TSL[j]    = muscle.getTendonSlackLength()
		# MIF[j]    = muscle.getMaxIsometricForce()
		# PAOFL[j]  = muscle.getPennationAngleAtOptimalFiberLength()
		# TSL[j]    = muscle.getTendonSlackLength()

		# PA[j]     = muscle.getPennationAngle(state)
		# CPA[i,j]  = muscle.getCosPennationAngle(state)
		# L[i,j]    = muscle.getLength(state)
		# FL[i,j]   = muscle.getFiberLength(state)
		FLT[i,j] = muscle.getFiberLengthAlongTendon(state)
		# TL[i,j]   = muscle.getTendonLength(state)
		FF[i,j]   = muscle.getFiberForce(state)
		# AFLM[i,j] = muscle.getActiveForceLengthMultiplier(state)
		# AFVM[i,j] = muscle.getForceVelocityMultiplier(state)
		AFF[i,j]  = muscle.getActiveFiberForce(state)
		PFF[i,j]  = muscle.getPassiveFiberForce(state)
		FFT[i,j]  = muscle.getFiberForceAlongTendon(state)
		AFFT[i,j]  = muscle.getActiveFiberForceAlongTendon(state)
		PFFT[i,j]  = muscle.getPassiveFiberForceAlongTendon(state)
		# TF[i,j]   = muscle.getTendonForce(state)
		# print(muscle.getActiveFiberForce(state), muscle.getPassiveFiberForce(state))

		indx = 0
		for k in range(nCoordinates):
			if cBool[k]:
				coordinate = coordinates.get(k)
				MA[i,indx,j] = muscle.computeMomentArm(state, coordinate)
				indx += 1

# timeit muscle.getActiveFiberForce(state)
# timeit muscle.getMaxIsometricForce() * muscle.getActiveForceLengthMultiplier(state) * muscle.getForceVelocityMultiplier(state)

# plt.plot(time, MA[:, nameCoordinates.index('knee_angle_r'), nameMuscles.index('recfem_r')])
# plt.show(block=False)

# plt.plot(time, L[:, nameMuscles.index('recfem_r')], label='muscle length')
# plt.plot(time, FL[:, nameMuscles.index('recfem_r')], label='fiber length')
# plt.plot(time, TL[:, nameMuscles.index('recfem_r')], label='tendon length')
# plt.legend()
# plt.show(block=False)

# plt.plot(time, S[:, nameMuscles.index('recfem_r')], label='strength')
# plt.plot(time, FF[:, nameMuscles.index('recfem_r')], label='fiber force', marker='o')
# plt.plot(time, AFF[:, nameMuscles.index('recfem_r')], label='active fiber force')
# plt.plot(time, PFF[:, nameMuscles.index('recfem_r')], label='passive fiber force')
# plt.plot(time, TF[:, nameMuscles.index('recfem_r')], label='tendon fiber force', marker='+')
# plt.plot(time, AFF[:, nameMuscles.index('recfem_r')] + PFF[:, nameMuscles.index('recfem_r')], label='active+passive fiber force')
# plt.legend()
# plt.show(block=False)

# plt.plot(time, L[:, nameMuscles.index('recfem_r')], label='muscle length')
# plt.plot(time, FL[:, nameMuscles.index('recfem_r')], label='fiber length')
# plt.plot(time, TL[:, nameMuscles.index('recfem_r')], label='tendon length')
# plt.axhline(OFL[nameMuscles.index('recfem_r')], label='optimal fiber length', color='c')
# plt.axhline(TSL[nameMuscles.index('recfem_r')], label='tendon slack length', color='r')
# plt.legend()
# plt.show(block=False)

# plt.plot(time, FF[:, nameMuscles.index('recfem_r')], label='fiber force', marker='o')
# plt.plot(time, AFF[:, nameMuscles.index('recfem_r')], label='active fiber force')

# plt.plot(time, AFFT[:, nameMuscles.index('gaslat_r')], label='gaslat')
# plt.plot(time, AFFT[:, nameMuscles.index('gasmed_r')], label='gasmed')
# plt.legend()
# plt.show(block=False)


#####################################################################
# %% OPTIMIZATION
#####################################################################
# https://stackoverflow.com/questions/37791680/scipy-optimize-minimize-slsqp-with-linear-constraints-fails

activity = np.empty((frame, nMuscles)) # muscle activity
force = np.empty((frame, nMuscles)) # muscle force

def objFun(a):  # sum of squared muscle activation
	return np.sum(volume*(a)**3) # volume*

'''Equality constraint means that the constraint function result is to be zero whereas inequality means that it is to be non-negative. '''

def eqConstraint(a):  # A.dot(x) - b  == np.sum(A*x, axis=1) - b
	'''equality constraint options:
	momentArm.dot(a*maxIsometricForce) - moment
	momentArm.dot(a*fiberForce) - moment
	momentArm.dot(a*activeFiberForce) - moment
	momentArm.dot(a*activeFiberForce*cosPennationAngle) - moment
	momentArm.dot(a*activeFiberForce+passiveFiberForce)*cosPennationAngle - moment
	momentArm.dot(a*activeFiberForceAlongTendon+passiveFiberForceAlongTendon) - moment'''	
	return momentArm.dot(a*activeFiberForceAlongTendon) - moment

constraints = {'type':'eq', 'fun': eqConstraint}
init = np.zeros(nMuscles) + 0.125 # initial guess of muscle activity (0.125)
lb = np.zeros(nMuscles) # lower bound (0)
ub = np.ones(nMuscles) # upper bound (1)
# ub = [np.inf for _ in range(nMuscles)]

for i in range(frame): #frame
	print(f'Optimization ... {i+1}/{len(time)} ({round(time[i],3)})')
	moment = np.vstack(list(m.values())).T[i,cBool] # 1D (nCoordinates)
	momentArm = MA[i,:,:] # 2D (nCoordinate, nMuscles)
	volume = FFT[i,:] * FLT[i,:] # 1D (nMuscles)
	activeFiberForceAlongTendon = AFFT[i,:] # 1D (nMuscles)
	passiveFiberForceAlongTendon = PFFT[i,:] # 1D (nMuscles)

	# ######################### scipy
	out = minimize(objFun, x0=init, method='SLSQP', bounds=Bounds(lb,ub), constraints=constraints, options={'maxiter':500}, tol=1e-05)
	print(f"\t\tfun={round(out['fun'],3)} success={out['success']}")
	if out['status'] != 0: print(f"\t\t\tmessage: {out['message']} ({out['status']})")

	activity[i,:] = out['x']
	force[i,:] = activeFiberForceAlongTendon * out['x']

plt.figure()
plt.plot(time, activity[:, nameMuscles.index('soleus_r')], label='soleus')
plt.plot(time, activity[:, nameMuscles.index('gasmed_r')], label='gast med')
plt.plot(time, activity[:, nameMuscles.index('gaslat_r')], label='gast lat')
plt.plot(time, activity[:, nameMuscles.index('perlong_r')], label='per_long_r')
plt.legend()
plt.title('Muscle Activity')
plt.show(block=False)

plt.figure()
plt.plot(time, force[:, nameMuscles.index('soleus_r')], label='soleus')
plt.plot(time, force[:, nameMuscles.index('gasmed_r')], label='gast med')
plt.plot(time, force[:, nameMuscles.index('gaslat_r')], label='gast lat')
plt.plot(time, force[:, nameMuscles.index('perlong_r')], label='per_long_r')
plt.legend()
plt.title('Muscle Force')
plt.show(block=False)

# write muscles activity and force to sto files
head = f"static optimization\nversion=1\nnRows={activity.shape[0]}\nnColumns={1+activity.shape[1]}\ninDegrees=yes\nendheader\n" + 'time\t' + '\t'.join(nameMuscles)
np.savetxt('muscle activity.sto', np.hstack((time.reshape((-1,1)),activity)), fmt='%.6f', delimiter='\t', newline='\n', header=head, comments='')
np.savetxt('muscle force.sto', np.hstack((time.reshape((-1,1)),force)), fmt='%.6f', delimiter='\t', newline='\n', header=head, comments='')



plt.plot(AFFT[:,0])
plt.show()
###################################################################################################
###################################################################################################
###################################################################################################
# %% WIP: create state file
labels = list()
# for i in nameCoordinates:
# 	labels.append(i+'/value')
# 	labels.append(i+'/speed')	
# for i in nameMuscles:
# 	labels.append(i+'/activation')
# 	labels.append(i+'/fiber_length')
stateMat = np.empty((frame, model.getNumStateVariables()))

for i in range(model.getNumStateVariables()):
	# print(model.getStateVariableNames().get(i))
	nameState = model.getStateVariableNames().get(i)
	labels.append(nameState)
	# nameState = model.getStateVariableNames().get(i).split('/')
	if 'jointset' in nameState:

q2 = np.empty((frame,nCoordinates))
u2 = np.empty((frame,nCoordinates))

for i,ii in enumerate(nameCoordinates):
	q2[:,i] = np.rad2deg(q[ii])
	u2[:,i] = np.rad2deg(u[ii])

head = f"state\nversion=1\nnRows={len(time)}\nnColumns={1+len(labels)}\ninDegrees=yes\nendheader\n" + 'time\t' + '\t'.join(labels)
np.savetxt('state.sto', np.hstack((time.reshape((-1,1)),q2,u2,activity,FF)), 
	fmt='%.6f', delimiter='\t', newline='\n', header=head, comments='')


# report = osim.report.Report(model, 'state.sto', bilateral=True)
# report.generate()

#####################################################################
# %% test, plot muscle parameters
#####################################################################
model = osim.Model('scaled.osim')
state = model.initSystem()
muscles = model.getMuscles()
nameMuscles = [i.getName() for i in muscles]
nMuscles = muscles.getSize()
coordinates = model.getCoordinateSet()
nCoordinates = coordinates.getSize()
nameCoordinates = [i.getName() for i in coordinates]

# get the body name that muscles attached
for i in muscles:
	for j in i.getGeometryPath().getPathPointSet():
		print(i.getName(), j.getBodyName())

# get range of motion of each coordinate
# motion type:
# Rotational		1
# Translational		2
# Coupled			3
rangeCoordinates = dict()
groupMuscles = dict()
for c in coordinates:
	groupMuscles[c.getName()] = list()
	# print(c.getName(), c.getMotionType())
	if c.getMotionType() == 1: # rotational
		rangeCoordinates[c.getName()] = np.rad2deg([c.getRangeMin(), c.getRangeMax()]).tolist()
	else:
		rangeCoordinates[c.getName()] = np.array([c.getRangeMin(), c.getRangeMax()]).tolist()
	# rangeCoordinates[c.getName()].append(c.getMotionType()) 

# 	rangeCoordinates[c.getName()].append(np.mean((c.getRangeMin(), c.getRangeMax())))
# 	model.getCoordinateSet().get(c.getName()).setValue(state, c.getDefaultValue(), False)
# model.assemble(state)

# for i in range(10):

# for m in model.getMuscles():
# 	for c in model.getCoordinateSet():
# 		# print(m.computeMomentArm(state, c))
# 		# break
# 		if abs(m.computeMomentArm(state, c)) > 1e-8:
# 			groupMuscles[c.getName()].append(m.getName())

n = 25
FF = np.empty(n)
AFF = np.empty(n)
PFF = np.empty(n)
TF = np.empty(n)
q = np.linspace(-40, 30, n)
muscle = model.getMuscles().get('soleus_r')
muscle.setActivation(state, 1)

for i in range(n):

	model.updCoordinateSet().get('ankle_angle_r').setValue(state, np.deg2rad(q[i]))
	model.updCoordinateSet().get('ankle_angle_r').setSpeedValue(state, 0)
	model.realizeDynamics(state)

	# muscle.set_ignore_activation_dynamics(True) # disable activation dynamics
	# muscle.set_ignore_tendon_compliance(True) # disable tendon compliance

	muscle.computeEquilibrium(state)
	
	FF[i] = muscle.getFiberForce(state)
	AFF[i] = muscle.getActiveFiberForce(state)
	PFF[i] = muscle.getPassiveFiberForce(state)
	TF[i] = muscle.getTendonForce(state)

plt.plot(q, FF, label='total fiber force')
plt.plot(q, AFF, label='active fiber force')
plt.plot(q, PFF, label='passive fiber force')
plt.plot(q, TF, label='tendon force')
plt.legend()
plt.xlabel('ankle angle (deg)')
plt.ylabel('force (N)')
plt.title('Soleus r')
plt.show(block=False)