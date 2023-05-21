modelName = 'input/scaled.osim'
IKName    = 'input/inverse_kinematics.mot'
IDName    = 'input/inverse_dynamics.sto'
GRFName   = 'input/grf_walk.mot'
ExtLName  = 'input/grf_walk.xml'

exclude = ['knee_angle_l_beta', 'knee_angle_l_beta', \
           'subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l']

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, Bounds
from time import time
import opensim as osim
osim.Logger.setLevel(4)

cycle = [0.86, 1.57] # stance time Rajagopal
# cycle = [0.6, 1.4] # stance time Gait2392

model = osim.Model(modelName)
state = model.initSystem()

nCoordinates    = model.getCoordinateSet().getSize()
nameCoordinates = [coordinate.getName() for coordinate in model.getCoordinateSet()]
nMuscles        = model.getMuscles().getSize()
nameMuscles     = [muscle.getName() for muscle in model.getMuscles()]
nameJoints      = [joint.getName() for joint in model.getJointSet()]

########## find muscles actuate coordinates
'''test three ranges [min, inter, and max] for each coordinate to see 
if there is any change at muscles length with a threshold of 0.1 mm 
(sum of absolute differences)'''

# coordinate = model.getCoordinateSet().get('knee_angle_r')
coordinateMuscle = dict()
for coordinate in model.getCoordinateSet():
	# criteria to include only free coordinates
	c1 = coordinate.get_locked()==False # unlocked
	c2 = coordinate.getMotionType()!=3  # not coupled
	c3 = coordinate.getJoint().getName() not in exclude # not excluded
	if (c1 and c2 and c3):
		# muscles length in default coordinate pose
		length0 = [muscle.getLength(state) for muscle in model.getMuscles()]

		r0 = coordinate.getDefaultValue()
		r1 = coordinate.getRangeMin() # min range
		r2 = coordinate.getRangeMax() # max range
		r3 = (r1+r2)/2       # intermediate range
		length = list()
		for j in [r1,r2,r3]:
			coordinate.setValue(state, j)
			model.realizePosition(state)
			length.append([muscle.getLength(state) for muscle in model.getMuscles()])
		# changes in muscle length (mm)
		dl = 1000 * (np.array(length) - length0) # 2D (3,nMuscles)
		ok = np.sum(np.abs(dl), axis=0)>1e-1 # sum of absolute difference
		coordinateMuscle[coordinate.getName()] = np.array(nameMuscles)[ok].tolist()
		coordinate.setValue(state, r0) # back to default 
	else:
		coordinateMuscle[coordinate.getName()] = []

# 'knee_angle_r': ['bflh_r', 'bfsh_r', 'gaslat_r', 'gasmed_r', 'grac_r', 'recfem_r', 'sart_r', 
                 # 'semimem_r', 'semiten_r', 'tfl_r', 'vasint_r', 'vaslat_r', 'vasmed_r']

muscleCoordinate  = dict()
empty = list() # empty or excluded coordinates
for coordinate,muscles in coordinateMuscle.items():
	if len(muscles)>0:
		for muscle in muscles: # each muscle
			if muscle not in muscleCoordinate.keys():
				muscleCoordinate[muscle] = list()
			muscleCoordinate[muscle].append(coordinate)
	else:
		empty.append(coordinate)
# 'gaslat_l': ['knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l']

# boolean to exclude specific coordinates
ok = [True] * nCoordinates # coordinates boolean
for i,coordinate in enumerate(nameCoordinates):
	if coordinate in empty:
		ok[i] = False


MIF = np.empty(nMuscles) # maximum isometric force
OFL = np.empty(nMuscles) # optimal fiber length
TSL = np.empty(nMuscles) # tendon slack length
OPA = np.empty(nMuscles) # pennation angle at optimal fiber length

for i,muscle in enumerate(model.updMuscles()):
	# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(i))

	MIF[i] = muscle.getMaxIsometricForce()
	OFL[i] = muscle.getOptimalFiberLength()
	TSL[i] = muscle.getTendonSlackLength()
	OPA[i] = muscle.getPennationAngleAtOptimalFiberLength()

	muscle.set_ignore_activation_dynamics(False) # activation dynamics (have no impact)

	if muscle.getTendonSlackLength() < muscle.getOptimalFiberLength():
		muscle.set_ignore_tendon_compliance(True) # rigid tendon
		print('rigid     tendon:', muscle.getName())
	else:
		muscle.set_ignore_tendon_compliance(False) # compliant tendon
		print('compliant tendon:', muscle.getName())

state = model.initSystem()
'''Rajagopal et al. 2016. Tendons were modeled as rigid when the 
tendon slack length was less than the muscle optimal fiber length.

In rigid tendons, there is no fiber_length in state.

in Millard2012EquilibriumMuscle:
ignoreTendonCompliance     computeEquilibrium     ForceVelocityMultiplier
        False                     False                     0
        False                     True                      1
        True                      False                     values
        True                      True                      values

# PCSA = max isometric force / 60
# muscle volume = muscle length * PCSA
# muscle stress = muscle force / PCSA
# contractile element = activity * max isometric force * 
                        active force-length multiplier * force-velocity multiplier
# parallel elasic element = max isometric force * passive force-length multiplier
# fiber force along tendon = (contractile element + parallel elasic element) * cos pennation angle
'''

########## Coordinates' values
q = osim.TimeSeriesTable(IKName)
osim.TableUtilities().filterLowpass(q, 7, padData=True)
model.getSimbodyEngine().convertDegreesToRadians(q)
q.trim(cycle[0], cycle[1]) # remove padding
t = q.getIndependentColumn() # time

########## calculate speed
GCVS = osim.GCVSplineSet(q)
# GCVS.evaluate(column,derivative,time)
d1 = osim.StdVectorInt(); d1.push_back(0) # first derivative
u = osim.TimeSeriesTable(t) # StdVectorDouble
for i in q.getColumnLabels():
	speed = [GCVS.get(i).calcDerivative(d1, osim.Vector(1,j)) for j in t]
	u.appendColumn(i, osim.Vector(speed))
u.addTableMetaDataString('inDegrees', 'no')
u.addTableMetaDataString('nColumns', str(u.getNumColumns()))
u.addTableMetaDataString('nRows', str(u.getNumRows()))

########## coordinates generalized force
m = osim.TimeSeriesTable(IDName)
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

# osim.STOFileAdapter().write(q, 'q.sto')
# osim.STOFileAdapter().write(u, 'u.sto')
# osim.STOFileAdapter().write(m, 'm.sto')

########## Opimization parameters
def objFun(a): # sum of weighted squared muscle activation
	return np.sum(weight*(a)**2)

'''Equality constraint means that the constraint function result is to be zero 
whereas inequality means that it is to be non-negative.'''

def eqConstraint(a):  # A.dot(x)-b  == np.sum(A*x,axis=1)-b
	# return momentArm.dot(a*MIF) - moment # classic static optimization
	return momentArm.dot(a*activeElement + passiveElement) - moment

# def eqConstraint2(a):  # gast activation constraint or EMG constraint
# 	return a[nameMuscles.index('gasmed_r')] - a[nameMuscles.index('gaslat_r')]


# weighting
PCSA   = MIF / 60     # specific tension used by Rajagopal et al. (2016) (N/cm^2)
volume = PCSA * OFL   # muscle volume
length = OFL*np.cos(OPA) + TSL    # muscle length
ratio  = OFL*np.cos(OPA) / length # fiber to muscle-tendon length ratio
# weight = volume * length
# weight = PCSA / ratio
weight = volume / ratio
# weight = 1 / ratio

# store weights as a CSV file
head = ['muscles','MIF','PCSA','OFL','TSL','LENGTH','VOLUME','1/RATIO','VOLUME/RATIO']
import csv
with open('output/weight.csv', mode='w', newline='') as f:
	writer = csv.writer(f)
	writer.writerows([head])
	out = [nameMuscles, np.round(MIF,3), np.round(PCSA,3), np.round(OFL,3), \
	       np.round(TSL,3), np.round(length,3), np.round(volume,3), np.round(1/ratio,3), \
	       np.round(volume/ratio,3)]
	writer.writerows(zip(*out))

# bounds, constraints and initial values
constraints = ({'type':'eq', 'fun':eqConstraint})
init = [0.1 for _ in range(nMuscles)] # initial guess of muscle activity (0.1)
lb   = [0.0 for _ in range(nMuscles)] # lower bound (0)
ub   = [1.0 for _ in range(nMuscles)] # max activity >= 1

########## Output variables
activity = osim.TimeSeriesTable()
activity.setColumnLabels(nameMuscles) # StdVectorString
force = activity.clone()

reaction = osim.TimeSeriesTableVec3()
reaction.setColumnLabels(nameJoints) # StdVectorString
ground = model.getGround()

########## Add external load file to the model
GRF = osim.Storage(GRFName)
for i in osim.ForceSet(ExtLName):
	exForce = osim.ExternalForce.safeDownCast(i)
	exForce.setDataSource(GRF)
	model.getForceSet().cloneAndAppend(exForce)

state  = model.initSystem()

########## Main loop
ts = time()
for i,ii in enumerate(t):
	state.setTime(ii)

	##### Update coordinates' values and speeds
	value = osim.RowVector(q.getRowAtIndex(i))
	speed = osim.RowVector(u.getRowAtIndex(i))
	for j,coordinate in enumerate(model.updCoordinateSet()):
		coordinate.setValue(state, value[j], False)
		coordinate.setSpeedValue(state, speed[j])
		
	model.assemble(state)
	# model.realizePosition(state)
	model.realizeVelocity(state)
	# model.realizeDynamics(state)
	# model.realizeAcceleration(state)
	# model.equilibrateMuscles(state)

	##### Get muscle parameters at each time frame
	# L   = np.empty(nMuscles) # muscle length
	CPA = np.empty(nMuscles) # cos Pennation angle
	FLM = np.empty(nMuscles) # active force length multiplier
	PFM = np.empty(nMuscles) # passive force multiplier
	FVM = np.empty(nMuscles) # force velocity multiplier
	MA  = np.zeros((nCoordinates, nMuscles)) # force velocity multiplier

	for j,muscle in enumerate(model.getMuscles()):
		# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
		muscle.setActivation(state, 1)
		muscle.computeEquilibrium(state)

		# L[j]   = muscle.getLength(state)
		CPA[j] = muscle.getCosPennationAngle(state)
		FLM[j] = muscle.getActiveForceLengthMultiplier(state)
		PFM[j] = muscle.getPassiveForceMultiplier(state)
		FVM[j] = muscle.getForceVelocityMultiplier(state)

		for coordinate in muscleCoordinate[muscle.getName()]:
			indx = nameCoordinates.index(coordinate)
			MA[indx,j] = muscle.computeMomentArm(state, model.getCoordinateSet().get(coordinate))

	##### Optimization
	# moment = m.getRowAtIndex(i).to_numpy()[ok] # THERE IS A BUG HERE
	moment = osim.RowVector(m.getRowAtIndex(i)).to_numpy()[ok] # 1D (nCoordinates)
	momentArm = MA[ok,:] # 2D (nCoordinate, nMuscles)
	activeElement  = MIF*FLM*FVM*CPA # along tendon, 1D (nMuscles)
	passiveElement = MIF*PFM*CPA     # along tendon, 1D (nMuscles)

	out = minimize(objFun, x0=init, method='SLSQP', bounds=Bounds(lb,ub), \
				   constraints=constraints, options={'maxiter':500}, tol=1e-06)
	init = out['x']
	activity.appendRow(ii, osim.RowVector(out['x']))
	force.appendRow(ii, osim.RowVector(activeElement * out['x'] + passiveElement))

	print(f'Optimization ... {(i+1):0>3d}/{len(t):0>3d} ({round(ii,3):.3f})', \
	 	  f"success={out['success']} fun={round(out['fun'],3)}")

	if out['status'] != 0: 
		print(f"\t\t\tmessage: {out['message']} ({out['status']})")

	##### Joint Reaction Analysis
	for j,muscle in enumerate(model.getMuscles()):
		# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
		muscle.setActivation(state, out['x'][j])

	model.equilibrateMuscles(state)
	model.realizeAcceleration(state)

	# vec3 = osim.RowVectorVec3(model.getJointSet().getSize())
	temp = list()
	for j,joint in enumerate(model.getJointSet()):
		reactionGround = joint.calcReactionOnChildExpressedInGround(state)
		temp.append(ground.expressVectorInAnotherFrame(state, reactionGround.get(1), joint.getChildFrame()))
		# vec3[j] = ground.expressVectorInAnotherFrame(state, reactionGround.get(1), joint.getChildFrame())
	reaction.appendRow(ii, osim.RowVectorVec3(temp))


print(f'Optimization ... finished in {time()-ts:.2f} s')



########## Inter-segmental forces and moments
# model2 = model.clone()
# model2.getMuscles().clearAndDestroy()
# model2.printToXML('new.osim')
# # model2.getForceSet(osim.ForceSet())

# for coordinate in model2.getCoordinateSet():
# if coordinate.get_locked()==False and coordinate.getMotionType()!=3:
# 	name = coordinate.getName()

# 	##### add coordinate actuator
# 	CA = osim.CoordinateActuator()
# 	CA.setName(name+'_actuator')
# 	CA.setCoordinate(coordinate)
# 	CA.setMinControl(-float('inf'))
# 	CA.setMaxControl(float('inf'))
# 	CA.setOptimalForce(1)
# 	model2.addForce(CA)

# 	##### add controller
# 	const = osim.Constant(0)
# 	const.setName(name+'_const')
# 	PC = osim.PrescribedController()

# 	PC.setName(name+'_controller')
# 	PC.addActuator(CA)
# 	PC.prescribeControlForActuator(0,const)
# 	model2.addController(PC)

# state2 = model2.initSystem()
# for i,ii in enumerate(t):
# 	state2.setTime(ii)

# 	##### Update coordinates' values and speeds
# 	value = osim.RowVector(q.getRowAtIndex(i))
# 	speed = osim.RowVector(u.getRowAtIndex(i))
# 	for j,coordinate in enumerate(model2.updCoordinateSet()):
# 		coordinate.setValue(state, value[j], False)
# 		coordinate.setSpeedValue(state, speed[j])

########## write output to sto files
reaction = reaction.flatten(['_x','_y','_z'])
for table in [reaction,activity,force]:
	table.addTableMetaDataString('inDegrees', 'no')
	table.addTableMetaDataString('nColumns', str(table.getNumColumns()))
	table.addTableMetaDataString('nRows',    str(force.getNumRows()))

osim.STOFileAdapter().write(reaction, 'output/jointReaction.sto')
osim.STOFileAdapter().write(activity, 'output/activity.sto')
osim.STOFileAdapter().write(force,    'output/force.sto')




plt.figure()
# plt.plot(t, activity.getDependentColumn('soleus_r').to_numpy(), label='soleus_r')
# plt.plot(t, activity.getDependentColumn('gasmed_r').to_numpy(), label='gast med_r')
# plt.plot(t, activity.getDependentColumn('gaslat_r').to_numpy(), label='gast lat_r')
# plt.plot(t, activity.getDependentColumn('perlong_r').to_numpy(), label='per_long_r')
plt.plot(t, activity.getDependentColumn('soleus_l').to_numpy(), label='soleus_l')
plt.plot(t, activity.getDependentColumn('gasmed_l').to_numpy(), label='gast med_l')
plt.plot(t, activity.getDependentColumn('gaslat_l').to_numpy(), label='gast lat_l')
plt.plot(t, activity.getDependentColumn('perlong_l').to_numpy(), label='per_long_l')
plt.plot(t, activity.getDependentColumn('tfl_l').to_numpy(), label='tfl_l')
plt.legend()
plt.title('Muscle Activity (weighted cost function)')
plt.xlabel('Time (s)')
# plt.savefig('output/activity.png')
plt.show(block=False)


plt.figure()
plt.plot(t, -1*reaction.getDependentColumn('walker_knee_l_y').to_numpy(), label='typical')
plt.show(block=False)


# %%
plt.close('all')
typical  = osim.TimeSeriesTable('output/jointReaction_t.sto')
volume = osim.TimeSeriesTable('output/jointReaction_v.sto')
volLength = osim.TimeSeriesTable('output/jointReaction_vl.sto')
pcsaRatio = osim.TimeSeriesTable('output/jointReaction_pr.sto')
volRatio = osim.TimeSeriesTable('output/jointReaction_vr.sto')
ratio = osim.TimeSeriesTable('output/jointReaction_r.sto')

t = typical.getIndependentColumn()

plt.close('all')
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(13,4.5), tight_layout=True, sharey=False, sharex=True)
plt.suptitle('Joint Contact Force')

ax1.plot(t, -1*typical.getDependentColumn('hip_l_y').to_numpy(), label='typical')
ax1.plot(t, -1*volume.getDependentColumn('hip_l_y').to_numpy(), label='volume')
# ax1.plot(t, -1*volLength.getDependentColumn('hip_l_y').to_numpy(), label='volume*length')
# ax1.plot(t, -1*pcsaRatio.getDependentColumn('hip_l_y').to_numpy(), label='PCSA/ratio')
ax1.plot(t, -1*ratio.getDependentColumn('hip_l_y').to_numpy(), label='1/ratio')
ax1.plot(t, -1*volRatio.getDependentColumn('hip_l_y').to_numpy(), label='volume/ratio')
ax1.set_title('Hip Joint')
ax1.set_xlabel('Time (s)')
ax1.legend()

ax2.plot(t, -1*typical.getDependentColumn('walker_knee_l_y').to_numpy(), label='typical')
ax2.plot(t, -1*volume.getDependentColumn('walker_knee_l_y').to_numpy(), label='volume')
# ax2.plot(t, -1*volLength.getDependentColumn('walker_knee_l_y').to_numpy(), label='volume*length')
# ax2.plot(t, -1*pcsaRatio.getDependentColumn('walker_knee_l_y').to_numpy(), label='PCSA/ratio')
ax2.plot(t, -1*ratio.getDependentColumn('walker_knee_l_y').to_numpy(), label='1/ratio')
ax2.plot(t, -1*volRatio.getDependentColumn('walker_knee_l_y').to_numpy(), label='volume/ratio')
ax2.set_title('Knee Joint')
ax2.set_xlabel('Time (s)')
ax2.legend()

ax3.plot(t, -1*typical.getDependentColumn('ankle_l_y').to_numpy(), label='typical')
ax3.plot(t, -1*volume.getDependentColumn('ankle_l_y').to_numpy(), label='volume')
# ax3.plot(t, -1*volLength.getDependentColumn('ankle_l_y').to_numpy(), label='volume*length')
# ax3.plot(t, -1*pcsaRatio.getDependentColumn('ankle_l_y').to_numpy(), label='PCSA/ratio')
ax3.plot(t, -1*ratio.getDependentColumn('ankle_l_y').to_numpy(), label='1/ratio')
ax3.plot(t, -1*volRatio.getDependentColumn('ankle_l_y').to_numpy(), label='volume/ratio')
ax3.set_title('Ankle Joint')
ax3.set_xlabel('Time (s)')
ax3.legend()

plt.savefig('output/KJCF.png', dpi=300)
# plt.show(block=False)





# %%
typical  = osim.TimeSeriesTable('output/activity_t.sto')
volume = osim.TimeSeriesTable('output/activity_v.sto')
volLength = osim.TimeSeriesTable('output/activity_vl.sto')
pcsaRatio = osim.TimeSeriesTable('output/activity_pr.sto')
volRatio = osim.TimeSeriesTable('output/activity_vr.sto')
ratio = osim.TimeSeriesTable('output/activity_r.sto')

t = typical.getIndependentColumn()

plt.close('all')
# ((ax1,ax2),(ax3,ax4))
fig, (ax1,ax2,ax5,ax6) = plt.subplots(1,4, figsize=(13,4.5), tight_layout=True, sharey=True, sharex=True)
plt.suptitle('Muscle activity')

ax1.plot(t, typical.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
ax1.plot(t, typical.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
ax1.plot(t, typical.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
ax1.plot(t, typical.getDependentColumn('perlong_l').to_numpy(), label='per long l')
ax1.plot(t, typical.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
ax1.set_title('Typical SOpt')
ax1.set_xlabel('Time (s)')
ax1.legend()

ax2.plot(t, volume.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
ax2.plot(t, volume.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
ax2.plot(t, volume.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
ax2.plot(t, volume.getDependentColumn('perlong_l').to_numpy(), label='per long l')
ax2.plot(t, volume.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
ax2.set_title('Volume-weighted SOpt')
ax2.set_xlabel('Time (s)')
ax2.legend()

# ax3.plot(t, volLength.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
# ax3.plot(t, volLength.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
# ax3.plot(t, volLength.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
# ax3.plot(t, volLength.getDependentColumn('perlong_l').to_numpy(), label='per long l')
# ax3.plot(t, volLength.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
# ax3.set_title('Volume*length-weighted SOpt')
# ax3.set_xlabel('Time (s)')
# ax3.legend()

# ax4.plot(t, pcsaRatio.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
# ax4.plot(t, pcsaRatio.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
# ax4.plot(t, pcsaRatio.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
# ax4.plot(t, pcsaRatio.getDependentColumn('perlong_l').to_numpy(), label='per long l')
# ax4.plot(t, pcsaRatio.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
# ax4.set_title('PCSA/ratio-weighted SOpt')
# ax4.set_xlabel('Time (s)')
# ax4.legend()

ax5.plot(t, ratio.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
ax5.plot(t, ratio.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
ax5.plot(t, ratio.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
ax5.plot(t, ratio.getDependentColumn('perlong_l').to_numpy(), label='per long l')
ax5.plot(t, ratio.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
ax5.set_title('1/ratio-weighted SOpt')
ax5.set_xlabel('Time (s)')
ax5.legend()


ax6.plot(t, volRatio.getDependentColumn('soleus_l').to_numpy(), label='soleus l')
ax6.plot(t, volRatio.getDependentColumn('gasmed_l').to_numpy(), label='gast med l')
ax6.plot(t, volRatio.getDependentColumn('gaslat_l').to_numpy(), label='gast lat l')
ax6.plot(t, volRatio.getDependentColumn('perlong_l').to_numpy(), label='per long l')
ax6.plot(t, volRatio.getDependentColumn('tfl_l').to_numpy(), label='tfl l')
ax6.set_title('volume/ratio-weighted SOpt')
ax6.set_xlabel('Time (s)')
ax6.legend()

plt.savefig('output/activity.png', dpi=300)
# plt.show(block=False)