import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, Bounds
from time import time
import opensim as osim
osim.Logger.setLevel(4)

cycle = [0.83, 1.58] # stance time

model = osim.Model('scaled.osim')
state = model.initSystem()

nMuscles = model.getMuscles().getSize()

MIF	= np.empty(nMuscles) # muscle maximum isometric force
OFL	= np.empty(nMuscles) # optimal fiber length
ML  = np.empty(nMuscles) # muscle length
# PA	= np.empty(nMuscles) # muscle pennation angle
# TSL	= np.empty(nMuscles) # tendon slack length

for i,muscle in enumerate(model.updMuscles()):
	# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(i))
	muscle.set_ignore_activation_dynamics(False) # activation dynamics (have no impact)
	muscle.set_ignore_tendon_compliance(False) # compliant tendon
	MIF[i] = muscle.getMaxIsometricForce()
	OFL[i] = muscle.getOptimalFiberLength()
	ML[i]  = muscle.getLength(state)
	# PA[i]     = muscle.getPennationAngle(state)
	# TSL[i]    = muscle.getTendonSlackLength()

state = model.initSystem()
'''Rajagopal et al. 2016. Tendons were modeled as rigid when the tendon slack length was less than the muscle optimal fiber length.'''
'''when set_ignore_tendon_compliance is true, the fiber_length is omitted from state and causes a change in state size'''

# get coordinates' value, speed and moment from IK and ID files
q = osim.TimeSeriesTable('inverse_kinematics.mot')
osim.TableUtilities().filterLowpass(q, 7, padData=True)
model.getSimbodyEngine().convertDegreesToRadians(q)
q.trim(cycle[0], cycle[1]) # remove padding
t = q.getIndependentColumn() # time

# calculate speed
GCVS = osim.GCVSplineSet(q)
firstDerivative = osim.StdVectorInt(); firstDerivative.push_back(0)
u = osim.TimeSeriesTable(osim.StdVectorDouble(t))
for i in q.getColumnLabels():
	speed = [GCVS.get(i).calcDerivative(firstDerivative,  osim.Vector(1,j)) for j in t]
	u.appendColumn(i, osim.Vector(speed))
u.addTableMetaDataString('inDegrees', 'no')
u.addTableMetaDataString('nColumns', str(u.getNumColumns()))
u.addTableMetaDataString('nRows', str(u.getNumRows()))

# coordinates generalized force
m = osim.TimeSeriesTable('inverse_dynamics.sto')
osim.TableUtilities().filterLowpass(m, 14, padData=True)
m.trim(cycle[0], cycle[1])

for i in q.getColumnLabels():
	if m.hasColumn(i+'_moment'):
		name = i+'_moment'
	elif m.hasColumn(i+'_force'):
		name = i+'_force'
	values = m.getDependentColumn(name).to_numpy()
	m.appendColumn(i, osim.Vector(values))
	m.removeColumn(name)

mMat = m.getMatrix()
# osim.STOFileAdapter().write(q, 'q.sto')
# osim.STOFileAdapter().write(u, 'u.sto')
# osim.STOFileAdapter().write(m, 'm.sto')

# remove pelvis (and any other?) coordinates
ok = [True] * model.getCoordinateSet().getSize() # coordinates boolean
for i,coordinate in enumerate(model.getCoordinateSet()):
	for j in ['pelvis', 'lumbar', 'beta', 'subtalar', 'mtp']:
		if j in coordinate.getName():
			ok[i] = False


'''Equality constraint means that the constraint function result is to be zero 
whereas inequality means that it is to be non-negative.'''

# PCSA = max isometric force / 60
# muscle volume = muscle length * PCSA
# muscle stress = muscle force / PCSA
# contractile element = activity * max isometric force * active force-length multiplier * force-velocity multiplier
# parallel elasic element = max isometric force * passive force-length multiplier
# fiber force along tendon = (contractile element + parallel elasic element) * cos pennation angle

def objFun(a):  # sum of squared muscle activation [volume weighted, power==3]
	# return np.sum((a)**2)
	return np.sum(volume*length*(a)**2) # volume*length*

def eqConstraint(a):  # A.dot(x)-b  == np.sum(A*x,axis=1)-b
	# return momentArm.dot(a*MIF) - moment # classic static optimization
	return momentArm.dot(a*activeElement + passiveElement) - moment

# def eqConstraint2(a):  # gast activation constraint or EMG constraint
# 	return a[nameMuscles.index('gasmed_r')] - a[nameMuscles.index('gaslat_r')]

PCSA = MIF / 60 # Rajagopal et al. (2016) (N/cm^2)
volume = PCSA * OFL
length = ML

constraints = ({'type':'eq', 'fun':eqConstraint})
init = [0.1 for _ in range(nMuscles)] # initial guess of muscle activity (0.1)
lb   = [0   for _ in range(nMuscles)] # lower bound (0)
ub   = [1   for _ in range(nMuscles)] # max activity >= 1

activity = osim.TimeSeriesTable()
activity.addTableMetaDataString('inDegrees', 'no')
activity.setColumnLabels(osim.StdVectorString([joint.getName() for joint in model.getMuscles()]))
force = activity.clone()

ts = time()
for i,ii in enumerate(t):
	state.setTime(ii)

	# update coordinates' values
	for coordinate in model.updCoordinateSet():
		name = coordinate.getName()
		coordinate.setValue(state, q.getDependentColumn(name)[i], False)
		coordinate.setSpeedValue(state, u.getDependentColumn(name)[i])
		
	model.assemble(state)
	# model.realizePosition(state)
	model.realizeVelocity(state)
	# model.realizeDynamics(state)
	# model.realizeAcceleration(state)
	# model.equilibrateMuscles(state)

	# muscle parameters at each time frame
	L   = np.empty(nMuscles) # muscle length
	CPA = np.empty(nMuscles) # cos Pennation angle
	FLM	= np.empty(nMuscles) # active force length multiplier
	FVM	= np.empty(nMuscles) # force velocity multiplier
	PFM = np.empty(nMuscles) # passive force multiplier
	MA 	= np.zeros((sum(ok), nMuscles)) # muscle moment arm (for specific coordinates only)

	for j,muscle in enumerate(model.getMuscles()):
		muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
		muscle.setActivation(state, 1)
		muscle.computeEquilibrium(state)

		L[j]   = muscle.getLength(state)
		CPA[j] = muscle.getCosPennationAngle(state)
		FLM[j] = muscle.getActiveForceLengthMultiplier(state)
		FVM[j] = muscle.getForceVelocityMultiplier(state)
		PFM[j] = muscle.getPassiveForceMultiplier(state)

		indx = 0
		for k,coordinate in enumerate(model.getCoordinateSet()):
			if ok[k]: # increases the speed
				MA[indx,j] = muscle.computeMomentArm(state, coordinate)
				# if abs(temp) > 1e-6:
				# 	MA[indx,j] = temp
				indx += 1

	# moment = m.getRowAtIndex(i).to_numpy()[ok] # THERE IS A BUG HERE
	moment = np.array([mMat.getElt(i,j) for j in range(len(ok))])[ok] # 1D (nCoordinates)
	momentArm = MA # 2D (nCoordinate, nMuscles)
	activeElement  = MIF*FLM*CPA # along tendon 1D (nMuscles) [FVM in compliant tendon is 1 (automatically excluded)]
	passiveElement = MIF*PFM*CPA # along tendon 1D (nMuscles)

	out = minimize(objFun, x0=init, method='SLSQP', bounds=Bounds(lb,ub), constraints=constraints, options={'maxiter':500}, tol=1e-06)
	init = out['x']
	activity.appendRow(ii, osim.RowVector(out['x']))
	force.appendRow(ii, osim.RowVector(activeElement * out['x'] + passiveElement))

	print(f"Optimization ... {(i+1):0>3d}/{len(t):0>3d} ({round(ii,3):.3f}) success={out['success']} fun={round(out['fun'],3)}")
	if out['status'] != 0: 
		print(f"\t\t\tmessage: {out['message']} ({out['status']})")

print(f'Optimization ... finished in {time()-ts:.2f} s')

# write muscles activity and force to sto files
activity.addTableMetaDataString('nColumns', str(activity.getNumColumns()))
activity.addTableMetaDataString('nRows', str(activity.getNumRows()))
force.addTableMetaDataString('nColumns', str(force.getNumColumns()))
force.addTableMetaDataString('nRows', str(force.getNumRows()))
osim.STOFileAdapter().write(activity, 'activity.sto')
osim.STOFileAdapter().write(force, 'force.sto')


plt.figure()
# plt.plot(t, activity.getDependentColumn('soleus_r').to_numpy(), label='soleus_r')
# plt.plot(t, activity.getDependentColumn('gasmed_r').to_numpy(), label='gast med_r')
# plt.plot(t, activity.getDependentColumn('gaslat_r').to_numpy(), label='gast lat_r')
# plt.plot(t, activity.getDependentColumn('perlong_r').to_numpy(), label='per_long_r')
plt.plot(t, activity.getDependentColumn('soleus_l').to_numpy(), label='soleus_l')
plt.plot(t, activity.getDependentColumn('gasmed_l').to_numpy(), label='gast med_l')
plt.plot(t, activity.getDependentColumn('gaslat_l').to_numpy(), label='gast lat_l')
plt.plot(t, activity.getDependentColumn('perlong_l').to_numpy(), label='per_long_l')
plt.legend()
plt.title('Muscle Activity (weighted cost function)')
plt.xlabel('Time (s)')
plt.savefig('activity.png')
plt.show(block=False)

plt.figure()
# plt.plot(t, force.getDependentColumn('soleus_r').to_numpy(), label='soleus_r')
# plt.plot(t, force.getDependentColumn('gasmed_r').to_numpy(), label='gast med_r')
# plt.plot(t, force.getDependentColumn('gaslat_r').to_numpy(), label='gast lat_r')
# plt.plot(t, force.getDependentColumn('perlong_r').to_numpy(), label='per_long_r')
plt.plot(t, force.getDependentColumn('soleus_l').to_numpy(), label='soleus_l')
plt.plot(t, force.getDependentColumn('gasmed_l').to_numpy(), label='gast med_l')
plt.plot(t, force.getDependentColumn('gaslat_l').to_numpy(), label='gast lat_l')
plt.plot(t, force.getDependentColumn('perlong_l').to_numpy(), label='per_long_l')
plt.legend()
plt.title('Muscle Force')
plt.show(block=False)
