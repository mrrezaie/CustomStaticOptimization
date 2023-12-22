# # Gait2392 model
# modelName = 'input/Gait2392_Simbody/scaled.osim'
# IKName    = 'input/Gait2392_Simbody/inverse_kinematics.mot'
# IDName    = 'input/Gait2392_Simbody/inverse_dynamics.sto'
# GRFName   = 'input/Gait2392_Simbody/subject01_walk1_grf.mot'
# ExtLName  = 'input/Gait2392_Simbody/subject01_walk1_grf.xml'
# geometry  = 'input/Gait2392_Simbody/Geometry'
# cycle = [0.6, 1.4] # stance time Gait2392
# # cycle = [0.0,2.5]

# Rajagopal model
modelName = 'input/scaled.osim'
IKName    = 'input/inverse_kinematics.mot'
IDName    = 'input/inverse_dynamics.sto'
GRFName   = 'input/grf_walk.mot'
ExtLName  = 'input/grf_walk.xml'
geometry  = 'input/Geometry'
# cycle = [0.86, 1.57] # left stance time Rajagopal
# cycle = [0.24, 1.4] # first right stride time Rajagopal
cycle = [1.4, 2.12] # second right stance time Rajagopal
weight = 85*9.81

# # LaiUhlrich2020 model
# modelName = 'input2/static_model.osim'
# IKName    = 'input2/walk_ik.mot'
# IDName    = 'input2/walk_id.sto'
# GRFName   = 'input2/walk_forces.mot'
# ExtLName  = 'input2/walk_grf.xml'
# geometry  = 'input2/Geometry'
# EMG       = 'input2/walk_emg.sto'
# cycle = [0, 1.26] # stance time Rajagopal
# weight = 50*9.81

# name 
exclude = ['subtalar_angle_r', 'subtalar_angle_l']

# minMax, momentMatch, accelerationMatch
criteria = 'momentMatch'

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from time import time as absoluteTime
import opensim as osim

# # Off Critical Error Warn Info Debug Trace
# osim.Logger.setLevel(4)
osim.Logger.setLevelString('Error')
# osim.Logger.removeFileSink()
# osim.Logger.addFileSink('loggg.log')

osim.ModelVisualizer.addDirToGeometrySearchPaths(geometry)

model = osim.Model(modelName)
state = model.initSystem()

# # name of all state variables
# for i in range(model.getNumStateVariables()):
# 	print(model.getStateVariableNames().get(i))

# get coordinates in multibody tree order
coordinateOrder = list()
for coordinate in model.getCoordinateSet():
	BI = coordinate.getBodyIndex()
	MI = coordinate.getMobilizerQIndex()
	coordinateOrder.append([BI, MI, coordinate])
multibodyOrder = [i[2] for i in sorted(coordinateOrder)]

# for i in model.getCoordinatesInMultibodyTreeOrder():
# 	coordinate = osim.Coordinate.safeDownCast(i) # rises error
# 	i.getName() # interrupes the Python session
# 	print(i)

nCoordinates     = model.getCoordinateSet().getSize()
nameCoordinates  = [coordinate.getName() for coordinate in model.getCoordinateSet()]
nameCoordinatesM = [coordinate.getName() for coordinate in multibodyOrder]
nMuscles         = model.getMuscles().getSize()
nameMuscles      = [muscle.getName() for muscle in model.getMuscles()]
nameJoints       = [joint.getName() for joint in model.getJointSet()]


########## find muscles spanning each coordinate
'''test three ranges [min, inter, and max] for each coordinate to see 
if there is any change at muscles length with a threshold of 0.1 mm 
(sum of absolute differences)'''

# coordinate = model.getCoordinateSet().get('knee_angle_r')
coordinateMuscles = dict()
unfree = list()
for coordinate in multibodyOrder:
	cName = coordinate.getName()
	# criteria to include only free coordinates
	c1 = not coordinate.get_locked()==True  # unlocked
	c2 = not coordinate.getMotionType()==3  # not coupled
	# c3 = not cName in exclude # not excluded

	if (c1 and c2):

		# print(cName)
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
		coordinateMuscles[cName] = np.array(nameMuscles)[ok].tolist()
		coordinate.setValue(state, r0) # back to default 
	else:
		# coordinateMuscles[cName] = []
		unfree.append(cName)

# example:
# knee_angle_r: ['bflh_r', 'bfsh_r', 'gaslat_r', 'gasmed_r', 'grac_r', 'recfem_r', 'sart_r', 
                 # 'semimem_r', 'semiten_r', 'tfl_r', 'vasint_r', 'vaslat_r', 'vasmed_r']


########## find coordinates actuated by each muscle
muscleCoordinates  = dict()
empty = list() # empty or excluded coordinates
for cName,musclesName in coordinateMuscles.items():
	if musclesName:
		for mName in musclesName: # each muscle
			if mName not in muscleCoordinates.keys():
				muscleCoordinates[mName] = list()
			if cName not in exclude:
				muscleCoordinates[mName].append(cName)
	else:
		empty.append(cName)

# example:
# gaslat_l: ['knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l']

# exclude: list of free coordinates that must be excluded from moment table
# unfree : list of either locked or coupled coordinates
# empty  : list of coordinates without any muscle

# boolean to include only specific coordinates
indxCoordinates = list()
for i,cName in enumerate(nameCoordinatesM):
	c1 = not cName in exclude
	c2 = not cName in unfree
	c3 = not cName in empty
	if c1 and c2 and c3:
		indxCoordinates.append(i)
		print('include', cName)
	else:
		print('exclude', cName)
print()

########## Get initial muscles properties
MIF = np.empty(nMuscles) # maximum isometric force
OFL = np.empty(nMuscles) # optimal fiber length
TSL = np.empty(nMuscles) # tendon slack length
OPA = np.empty(nMuscles) # pennation angle at optimal fiber length

for i,muscle in enumerate(model.getMuscles()):
	# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
	# muscle.setMaxIsometricForce( 0.5* muscle.getMaxIsometricForce())

	MIF[i] = muscle.getMaxIsometricForce()
	OFL[i] = muscle.getOptimalFiberLength()
	TSL[i] = muscle.getTendonSlackLength()
	OPA[i] = muscle.getPennationAngleAtOptimalFiberLength()

	muscle.set_ignore_activation_dynamics(False) # activation dynamics (have no impact)
	muscle.set_ignore_tendon_compliance(False)   # compliant tendon
	# muscle.set_ignore_tendon_compliance(True)    # rigid tendon

	if muscle.getTendonSlackLength() < muscle.getOptimalFiberLength():
		muscle.set_ignore_tendon_compliance(True) # rigid tendon
		print('r i g i d tendon:', muscle.getName())
	else:
		muscle.set_ignore_tendon_compliance(False) # compliant tendon
		print('compliant tendon:', muscle.getName())
print()

state = model.initSystem() # the size is subject to the tendon models


########## coordinates value
IKFile = osim.TimeSeriesTable(IKName)

osim.TableUtilities().filterLowpass(IKFile, 7, padData=True) # Butterworthlow pass filter (3rd order)
IKFile.trim(cycle[0], cycle[1]) # remove padding

if IKFile.getTableMetaDataString('inDegrees') == 'yes':
	model.getSimbodyEngine().convertDegreesToRadians(IKFile) # convert from degrees to radians
	print('Coordinates were converted to Radians\n')

times = IKFile.getIndependentColumn() # time

q = osim.TimeSeriesTable(times)
# change order of q table to multibody tree order
for cName in nameCoordinatesM:
	if IKFile.hasColumn(cName):
		column = IKFile.getDependentColumn(cName) #.to_numpy() VectorView returns wrong values
		q.appendColumn(cName, (column)) #osim.Vector 
	else:
		print(f'{cName} does not exist in {IKName} file')

del IKFile, column


########## calculate speeds and accelerations
GCVS = osim.GCVSplineSet(q) # degree=5

u     = q.clone() # osim.TimeSeriesTable(times)
u_dot = q.clone()

# d1 = osim.StdVectorInt(); d1.push_back(0) # first derivative
# d2 = osim.StdVectorInt(); d2.push_back(0); d2.push_back(0) # second derivative

for i,label in enumerate(q.getColumnLabels()): # already in multibody tree order

	# speedColumn = [GCVS.evaluate(i, 1, time) for time in times] # first  derivative
	# accelColumn = [GCVS.evaluate(i, 2, time) for time in times] # second derivative
	# speedColumn = [GCVS.get(label).calcDerivative(d1, osim.Vector(1,time)) for time in times]
	# accelColumn = [GCVS.get(label).calcDerivative(d2, osim.Vector(1,time)) for time in times]

	speedColumn = u.updDependentColumn(label)
	accelColumn = u_dot.updDependentColumn(label)

	for j, time in enumerate(times):
		speedColumn[j] = GCVS.evaluate(i, 1, time) # first  derivative
		accelColumn[j] = GCVS.evaluate(i, 2, time) # second derivative


########## generalized forces
IDFile = osim.TimeSeriesTable(IDName)

osim.TableUtilities().filterLowpass(IDFile, 14, padData=True)
IDFile.trim(cycle[0], cycle[1])

tau = osim.TimeSeriesTable(times)

for cName in nameCoordinatesM:

	if IDFile.hasColumn(cName+'_moment'):
		name = cName+'_moment'
	elif IDFile.hasColumn(cName+'_force'):
		name = cName+'_force'

	# print(name, '\t', cName)
	column = IDFile.getDependentColumn(name) # .to_numpy()
	tau.appendColumn(cName, (column)) #osim.Vector

del IDFile, column

# update tables' metadata
for table in [q,u,u_dot,tau]:
	table.addTableMetaDataString('inDegrees', 'no')
	table.addTableMetaDataString('nColumns', str(table.getNumColumns()))
	table.addTableMetaDataString('nRows',    str(table.getNumRows()))


########## Add external load file to the model
GRF = osim.Storage(GRFName)
for i in osim.ForceSet(ExtLName):
	exForce = osim.ExternalForce.safeDownCast(i)
	exForce.setDataSource(GRF)
	model.getForceSet().cloneAndAppend(exForce)


########## Add actuators to coordinates without muscle

nActuators = 0
indxActuators = list() # index of coordinates with actuator

for i,coordinate in enumerate(multibodyOrder):

	cName = coordinate.getName()
	c1 = cName in empty
	c2 = not cName in exclude
	c3 = not cName in unfree

	if c1 and c2 and c3:

		print(f'Actuator for {cName}')
		indxActuators.append(i)

		# coordinate actuator
		actuator = osim.CoordinateActuator()
		actuator.setName(cName+'_actuator')
		actuator.setCoordinate(coordinate)
		actuator.setMinControl(-np.inf)
		actuator.setMaxControl(+np.inf)
		actuator.setOptimalForce(1) # activation == force
		model.addForce(actuator)

		# # prescribe controller
		# PC = osim.PrescribedController()
		# PC.setName(cName+'_controller')
		# PC.addActuator(actuator)
		# const = osim.Constant(0)
		# const.setName(cName+'_const')
		# PC.prescribeControlForActuator(0,const)
		# model.addController(PC)

		nActuators += 1

print(f'Overall {nActuators} coordinate actuators\n')

state  = model.initSystem()
assert model.getNumControls() == (nMuscles+nActuators)


########## Output variables
activity = osim.TimeSeriesTable()
activity.setColumnLabels(nameMuscles) # StdVectorString
force = activity.clone()
fiberLength = activity.clone()
reaction = osim.TimeSeriesTableVec3()
reaction.setColumnLabels(nameJoints) # StdVectorString
ground = model.getGround()


########## Opimization parameters

if criteria == 'momentMatch':

	def objFun(a):

		# mActivation = a[:nMuscles]
		# aActivation = a[-nActuators:]
		# global musclesMomentW, musclesMoment

		musclesForce = a * activeElement + passiveElement
		musclesMoment  = momentArm * (musclesForce)
		momentError = np.sqrt( (musclesMoment.sum(axis=1) - moment)**2 ).sum()

		# musclesMomentW = np.sqrt(np.sum(musclesMoment**2, axis=0))
		musclesMomentW = np.abs(musclesMoment).sum(axis=0)/10 # better than the former
		# musclesMomentW = np.mean(np.abs(musclesMoment), axis=0)
		# musclesMomentW = np.sum(np.abs(musclesMoment), axis=0) / np.count_nonzero(momentArm, axis=0)/10 # interesting
		# musclesMomentW = np.sum(np.abs(musclesMoment), axis=0)**(1/np.count_nonzero(momentArm, axis=0)) 

		# MTU weighting
		PCSA   = MIF / 60     # specific tension used by Rajagopal et al. (2016) (N/cm^2)
		volume = PCSA * OFL   # muscle volume
		length = OFL*np.cos(OPA) + TSL    # muscle length, 
		fiberR = 1 / (OFL*np.cos(OPA) / length) # fiber to muscle-tendon length ratio
		tenR   = TSL / length

		# print('recfem', musclesMomentW[nameMuscles.index('recfem_r')].round(2))
		# print('sart', musclesMomentW[nameMuscles.index('sart_r')].round(2))
		# print('psoas', musclesMomentW[nameMuscles.index('psoas_r')].round(2))
		# print('tfl', musclesMomentW[nameMuscles.index('tfl_r')].round(2))

		# np.sum( (musclesMomentJ - moment)**2 ) 
		# return np.sum(a**2) + np.abs(musclesMoment).sum() # the lowest JCF
		return np.sum( musclesMomentW * a**2) # + np.sum(aActivation**2)


	def eqConstraint(a):  # A.dot(x)-b  == np.sum(A*x,axis=1)-b
		# mActivation = a[:nMuscles]
		# aActivation = a[-nActuators:]
		musclesForce = a * activeElement + passiveElement
		musclesMoment  = momentArm.dot(musclesForce)
		# actuatorsForce = np.zeros_like(musclesMoment)
		# actuatorsForce[indxActuators] = aActivation

		# return (musclesMoment + actuatorsForce) - moment
		return musclesMoment - moment

	# def eqConstraint2(a):  # gast activation constraint or EMG constraint
	# 	return a[nameMuscles.index('gasmed_r')] - a[nameMuscles.index('gaslat_r')]

	# bounds, constraints and initial values
	init = [0.1 for _ in range(nMuscles)] # +nActuators)] # initial guess of muscle activity (0.1)
	lb = [0. for _ in range(nMuscles)] # + [-np.inf for _ in range(nActuators)]
	ub = [1. for _ in range(nMuscles)] # + [+np.inf for _ in range(nActuators)]

	# constraints = ({'type':'eq', 'fun':eqConstraint}) # linear equality constraint
	constraints = NonlinearConstraint(eqConstraint, lb=0, ub=0) # nonlinear equality constraint


if criteria == 'minMax': # minmax critera (Rasmussen2001)

	# the last element is beta
	def objFun(a):
		beta = a[-1]
		return beta # minimize beta

	def eqConstraint(a): # must be zero
		activation = a[:-1]
		return momentArm.dot(activation*activeElement + passiveElement) - moment

	def ineqConstraint(a): # must be non-negative
		beta = a[-1]
		activation = a[:-1]
		return beta - activation  # activations less than beta

	constraints = ({'type':'ineq', 'fun':ineqConstraint}, # linear inequality constraint
				   {'type':  'eq', 'fun':  eqConstraint}) # linear   equality constraint

	init = [0.1 for _ in range(nMuscles+1)]
	lb   = 0.
	ub   = 1.


if criteria == 'accelerationMatch':


	def objFun(a):

		# mActivation = a[:nMuscles]
		# aActivation  = a[-nActuators:]	

		return np.sum(a**2) # + np.sum(mActivation**2)


	def eqConstraint(a): # must be zero

		global predictedUDot

		# a = out['x']

		# musclesActivation = a[:nMuscles]
		# actuatorsControl  = a[-nControls:]

		# model.setControls(state, osim.Vector(a))

		for i,muscle in enumerate(model.getMuscles()):
			muscle.setActivation(state, a[i])

		model.equilibrateMuscles(state)

		# for i,controller in enumerate(model.getControllerSet()):
		# 	print(i, controller.getName())
		# 	PC    = osim.PrescribedController.safeDownCast(controller)
		# 	const = osim.Constant.safeDownCast(PC.get_ControlFunctions(0).get(0))
		# 	const.setValue(actuatorsControl[i])

		# ii = 0
		# for i,actuator in enumerate(model.getActuators()):
		# 	if actuator.getName().endswith('_actuator'):
		# 		# print(actuator.getName())
		# 		actuator = osim.CoordinateActuator().safeDownCast(actuator)
		# 		actuator.setOverrideActuation(state, actuatorsControl[ii])
		# 		ii += 1



		model.realizeAcceleration(state)

		# # name of all state speed variables [in coordinate set order]
		# for i in range(model.getNumStateVariables()):
		# 	name = model.getStateVariableNames().get(i)
		# 	print(name)

		# predictedUDot = list()
		# for i in multibodyOrder:
		# 	nameSpeed = i.getAbsolutePathString() + '/speed'
		# 	if '_beta' not in nameSpeed or 'mtp_angle' not in nameSpeed:
		# 		speed = model.getStateVariableDerivativeValue(state, nameSpeed)
		# 		predictedUDot.append(speed)

		# accelOK = len(nameCoordinatesM) * [True]
		# for i,coordinate in enumerate(multibodyOrder):
		# 	if coordinate.get_locked()==True or coordinate.getMotionType()==3 or \
		# 		'pelvis' in coordinate.getName() or 'lumbar' in coordinate.getName():
		# 		accelOK[i] = False

		predictedUDot = state.getUDot().to_numpy()[indxCoordinates]

		return predictedUDot - experimentalUDot


	init = [0.1 for _ in range(nMuscles)]
	lb   = [0.0 for _ in range(nMuscles)] # + [-1. for _ in range(nControls)]
	ub   = [1.0 for _ in range(nMuscles)]

	constraints = ({'type':'eq', 'fun':eqConstraint}) # linear equality constraint
	# constraints = NonlinearConstraint(eqConstraint, lb=0, ub=0) # nonlinear equality constraint


state  = model.initSystem()
timeStart = absoluteTime()


########## Main optimization loop
print('Optimization ... started')
for ti,time in enumerate(times):
	
	state.setTime(time)

	##### Update coordinates' values and speeds
	for coordinate in multibodyOrder:
		cName = coordinate.getName()
		coordinate.setValue(state, q.getDependentColumn(cName)[ti], enforceContraints=False)
		coordinate.setSpeedValue(state, u.getDependentColumn(cName)[ti])

	model.assemble(state)

	# model.realizePosition(state)
	model.realizeVelocity(state)
	# model.realizeDynamics(state)
	# model.realizeAcceleration(state)
	# model.equilibrateMuscles(state)

	# state.getUDot().to_numpy()
	# state.getU().to_numpy()
	# state.getQ().to_numpy()
	# a.getRowAtIndex(0).to_numpy()
	# u.getRowAtIndex(0).to_numpy()
	# q.getRowAtIndex(0).to_numpy()

	##### Get muscle parameters at each time frame
	# L   = np.empty(nMuscles) # muscle length
	CPA = np.empty(nMuscles) # cos Pennation angle
	FLM = np.empty(nMuscles) # active force length multiplier
	PFM = np.empty(nMuscles) # passive force multiplier
	FVM = np.empty(nMuscles) # force velocity multiplier
	MA  = np.zeros((nCoordinates, nMuscles)) # force velocity multiplier
	FL  = np.empty(nMuscles) # fiber length

	for j,muscle in enumerate(model.getMuscles()):
		# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
		mName = muscle.getName()
		muscle.setActivation(state, 1)
		muscle.computeEquilibrium(state)

		# L[j]   = muscle.getLength(state)
		CPA[j] = muscle.getCosPennationAngle(state)
		FLM[j] = muscle.getActiveForceLengthMultiplier(state)
		PFM[j] = muscle.getPassiveForceMultiplier(state)
		FVM[j] = muscle.getForceVelocityMultiplier(state)
		FL[j]  = muscle.getFiberLength(state)

		for cName in muscleCoordinates[mName]:
			indx = nameCoordinatesM.index(cName)
			coordinate = model.getCoordinateSet().get(cName)
			MA[indx,j] = muscle.computeMomentArm(state, coordinate)

	fiberLength.appendRow(time, osim.RowVector(FL))

	##### Optimization
	moment = tau.getRowAtIndex(ti).to_numpy()[indxCoordinates] # 1D (nCoordinates) in CoordinateSet order
	momentArm = MA[indxCoordinates,:] # 2D (nCoordinate, nMuscles)

	# in case of tendon elasticity and fiber equilibrium, FVM is already one
	activeElement  = MIF*FLM*FVM*CPA # along tendon, 1D (nMuscles)
	passiveElement = MIF*PFM*CPA     # along tendon, 1D (nMuscles)

	experimentalUDot = u_dot.getRowAtIndex(ti).to_numpy()[indxCoordinates]

	out = minimize(objFun, x0=init, method='SLSQP', bounds=Bounds(lb,ub), constraints=constraints, 
					options={'maxiter':200}, tol=1e-6)
	init = out['x']

	if criteria == 'minMax':
		mActivation = out['x'][:-1]
	else:
		mActivation = out['x']

	activity.appendRow(time, osim.RowVector(mActivation))
	force.appendRow(time, osim.RowVector(activeElement * mActivation + passiveElement))

	print(f'Optimization ... {(ti+1):0>3d}/{len(times):0>3d} ({round(time,3):.3f})', \
	 	  f"success={out['success']} fun={round(out['fun'],3)}")#, musclesMomentW.sum().round(3))

	if out['status'] != 0: 
		print(f"\t\t\tmessage: {out['message']} ({out['status']})")


	##### Joint Reaction Analysis
	for i,muscle in enumerate(model.getMuscles()):
		muscle.setActivation(state, mActivation[i])

	# controls = list() # mActivation.tolist() + 
	# for controller in model.getControllerSet():
	# 	cName = controller.getName()[:-11]
	# 	values = aActivation # m.getDependentColumn(cName)[ti]
	# 	controls.append(values)
	# 	# print(cName)
	# 	PC    = osim.PrescribedController.safeDownCast(controller)
	# 	const = osim.Constant.safeDownCast(PC.get_ControlFunctions(0).get(0))
	# 	const.setValue(values)

	aActivation = tau.getRowAtIndex(ti).to_numpy()[indxActuators]

	model.setControls(state, osim.Vector(mActivation.tolist() + aActivation.tolist()))
	model.equilibrateMuscles(state)
	model.realizeAcceleration(state)

	row = list()
	for j,joint in enumerate(model.getJointSet()):
		reactionGround = joint.calcReactionOnChildExpressedInGround(state)
		reactionForce  = reactionGround.get(1) # 0==moment, 1==force
		jointChildBody = joint.getChildFrame().findBaseFrame() # body frame not joint frame
		row.append(ground.expressVectorInAnotherFrame(state, reactionForce, jointChildBody))
	reaction.appendRow(time, osim.RowVectorVec3(row))

	# break

print(f'Optimization ... finished in {absoluteTime()-timeStart:.2f} s')

# print(np.round(predictedUDot, 3))
# print(np.round(experimentalUDot, 3))


########## write output to sto files
reaction = reaction.flatten(['_x','_y','_z'])

stateAll = osim.TimeSeriesTable(times)

for i in multibodyOrder:
	stateAll.appendColumn(i.getAbsolutePathString()+'/value', q.getDependentColumn(i.getName()))
	# stateAll.appendColumn(i.getAbsolutePathString()+'/speed', u_dot.getDependentColumn(i.getName()))
for i in model.getMuscles():
	stateAll.appendColumn(i.getAbsolutePathString()+'/fiber_length', fiberLength.getDependentColumn(i.getName()))
	stateAll.appendColumn(i.getAbsolutePathString()+'/activation', activity.getDependentColumn(i.getName()))

for table in [reaction,activity,force,stateAll]:
	table.addTableMetaDataString('inDegrees', 'no')
	table.addTableMetaDataString('nColumns', str(table.getNumColumns()))
	table.addTableMetaDataString('nRows',    str(table.getNumRows()))

# # # osim.STOFileAdapter().write(reaction, 'output/jointReaction.sto')
# # # osim.STOFileAdapter().write(activity, 'output/activity.sto')
# # # osim.STOFileAdapter().write(force,    'output/force.sto')
# osim.STOFileAdapter().write(stateAll,    'output/state.sto')


# This table should contain both coordinates value and speeds and muscle activations. 
# Note that muscle excitations (i.e., columns labeled like '/forceset/soleus_r') 
# will not visualize, because they are not states in the model. 
# If you're constructing a table and adding the muscle activations that you want to visualize, 
# make sure they have the correct column name (i.e., '/forceset/soleus_r/activation').



# plt.close('all')
# plt.figure(figsize=(8,4), layout="constrained")
# plt.suptitle('ID vs. muscle moment', fontsize=20)
# for i,j in enumerate(np.array(nameCoordinates)[ok]):
# 	ax = plt.subplot(2,5,i+1)
# 	ax.plot(m.getDependentColumn(j))
# 	ax.plot(momenMuscle.getDependentColumn(j), linestyle='--')
# 	ax.set_title(j)
# 	ax.yaxis.set_tick_params(labelsize=7)
# 	if i==4:
# 		ax.legend(['ID', 'Muscles'], prop={'size': 7})
# plt.show(block=False)







# ########## Statistics
# ########## Compare EMG and muscles activity (cross-correlation)

# def interp(data, N=101):
#     x = np.arange(len(data))
#     xp = np.linspace(0, len(data), N)
#     return np.interp(xp, x, data)
    
# emg = osim.TimeSeriesTable(EMG)

# # from scipy.stats import pearsonr

# print('\nCross-correlation with EMG:')
# for label in ['tibant_r', 'soleus_r', 'gasmed_r', 'vasmed_r', 'recfem_r', 'semiten_r']:

# 	muscleEMG = interp(emg.getDependentColumn(label).to_numpy(), len(t))
# 	muscleSO  = activity.getDependentColumn(label).to_numpy()
# 	# normalize the input signals
# 	muscleEMG /= np.linalg.norm(muscleEMG)
# 	muscleSO  /= np.linalg.norm(muscleSO)

# 	# plt.figure()
# 	# plt.plot(muscleSO, label='SO')
# 	# plt.plot(muscleEMG, label='EMG')
# 	# plt.legend()

# 	# correlation (pearson or cross-correlation), the later handles shiftings
# 	# pearCorr = pearsonr(muscleEMG, muscleSO)
# 	crossCorr = np.correlate(muscleEMG, muscleSO, mode='full')
# 	crossCorrMax = crossCorr.max().round(3) # the maximum 
# 	print(label, crossCorrMax)

# 	# # plt.plot(crossCor)
# 	# plt.title(f'{label}: {crossCorrMax}')
# 	# plt.show(block=False)	


########## extract the second joint contact force peak
print('\nSecond joint contact force peak:')
for i in ['ankle_r_y', 'walker_knee_r_y', 'hip_r_y']:
	signal = -1*reaction.getDependentColumn(i).to_numpy()/ (weight)
	print(i, np.max(signal[40:80]).round(2))


########## extract synergy for each muscle group
print("\nMuscle recruitment (synergy vector or weight):")

from sklearn.decomposition import NMF

groupMuscles = dict()
for i in range(model.getForceSet().getNumGroups()):
	nameGroup = model.getForceSet().getGroup(i).getName()
	groupMuscles[nameGroup] = list()
	for j in range(model.getForceSet().getGroup(i).getMembers().getSize()):
		nameMember = model.getForceSet().getGroup(i).getMembers().get(j).getName()
		groupMuscles[nameGroup].append(nameMember)
		# print(nameGroup, nameMember)


# (key.startswith('knee') or key.startswith('ankle')) and 
for key,items in groupMuscles.items():
	if key.endswith('_r'): # and not 'rot' in key and not 'verter' in key
		# print(key, items)
		data = [activity.getDependentColumn(i).to_numpy() for i in items]
		modelNMF = NMF(n_components=1, init='random', random_state=0, max_iter=5000)
		W = modelNMF.fit_transform(data)
		# H = modelNMF.components_
		# plt.plot(H[0])
		# plt.plot(np.dot(W,H).T)
		# plt.plot(np.transpose(data), linestyle='--')
		# plt.show(block=False)

		print(key, np.std(W.sum(axis=1), ddof=1).round(3))



########## plot

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 7 
mpl.rcParams['ytick.labelsize'] = 7 

# plt.close('all')
_, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(2.5,7.25), layout="constrained", sharex=True)


for i in ['soleus_r','gasmed_r','gaslat_r','perlong_r','tibpost_r','perbrev_r']:
# for i in ['soleus_r','med_gas_r','lat_gas_r','per_long_r','tib_ant_r','tfl_r']:
	if i=='perlong_r':
		ax1.plot(times, activity.getDependentColumn(i).to_numpy(), label=i, linestyle='--')
	elif i=='perbrev_r':
		ax1.plot(times, activity.getDependentColumn(i).to_numpy(), label=i, linestyle='-.')
	elif i=='tibpost_r':
		ax1.plot(times, activity.getDependentColumn(i).to_numpy(), label=i, linestyle='dotted')
	else:
		ax1.plot(times, activity.getDependentColumn(i).to_numpy(), label=i)
ax1.set_title('Ankle Plantarflexors')
ax1.set_ylabel('Activation', fontsize=7)
# ax1.set_ylim(-0.01,0.65)
ax1.legend(prop={'size': 6})
# ax1.set_xlabel('Time (s)')

for i in ['glmin1_r','glmin2_r','glmin3_r','glmed1_r','glmed2_r','glmed3_r']:
# for i in ['glut_min1_r','glut_min2_r','glut_min3_r','glut_med1_r','glut_med2_r','glut_med3_r','rect_fem_r']:
	ax2.plot(times, activity.getDependentColumn(i).to_numpy(), label=i)
ax2.set_title('Hip Abductors')
ax2.set_ylabel('Activation', fontsize=7)
# ax2.set_ylim(-0.01,0.9)
ax2.legend(prop={'size': 6})

for i in ['iliacus_r','psoas_r','tfl_r','sart_r','recfem_r']:
# for i in ['glut_min1_r','glut_min2_r','glut_min3_r','glut_med1_r','glut_med2_r','glut_med3_r','rect_fem_r']:
	ax3.plot(times, activity.getDependentColumn(i).to_numpy(), label=i)
ax3.set_title('Hip Flexors')
ax3.set_ylabel('Activation', fontsize=7)
# ax3.set_ylim(-0.01,0.85)
ax3.legend(prop={'size': 6})

ax4.plot(times, -1*reaction.getDependentColumn('hip_r_y').to_numpy() / (weight), label='HJCF')
ax4.plot(times, -1*reaction.getDependentColumn('walker_knee_r_y').to_numpy() / (weight), label='KJCF')
ax4.plot(times, -1*reaction.getDependentColumn('ankle_r_y').to_numpy() / (weight), label='AJCF')
ax4.set_title('Joints Contact Force')
ax4.set_xlabel('Stance Time (s)', fontsize=7)
ax4.set_ylabel('Force (N/BW)', fontsize=7)
ax4.set_ylim(-0.01,6.5)
ax4.legend(prop={'size': 6})

plt.savefig('plot.png', dpi=500)
plt.show(block=False)



# viz = osim.VisualizerUtilities()
# # viz.showModel(model)
# viz.showMotion(model, stateAll)



# stiffness2 = MIF / length
# MTUweight = np.zeros(nMuscles)
# MTUweight = np.ones(nMuscles) * -1
# MTUweight = volume * length
# MTUweight = PCSA / ratio
# MTUweight = volume / ratio
# MTUweight = 1 / ratio
# MTUweight = TSL / ratio
# MTUweight = (MIF * OFL)
# MTUweight = volume
# MTUweight = np.ones(nMuscles)
# MTUweight = (MIF/max(MIF)) * 5 * (OFL/max(OFL))
# MTUweight = (MIF/100) * (TSL-OFL) # so interesting
# MTUweight = np.abs((MIF) * (TSL-OFL)) # WOW, but why?
# MTUweight = (MIF) / (1000 * OFL**2 * np.sin(OPA) * np.cos(OPA))
# MTUweight = (MIF/max(MIF)) * (OFL/max(OFL)) / (length/max(length))
# MTUweight = MIF * OFL / length / 1000 # very bad
# MTUweight = (MIF/max(MIF)) / (OFL/max(OFL))
# MTUweight = (MIF/max(MIF)) * (length/max(length)) / (OFL/max(OFL)) # interesting
# MTUweight = (MIF/10000) / ratio # so interesting
# MTUweight = volume * TSL # bad for Gmin
# MTUweight = volume * tenR
# MTUweight = (PCSA/10) * tenR
# MTUweight = PCSA * ratio
# MTUweight = tenR # good but too much KJCF
# MTUweight = TSL
# MTUweight = OFL
# MTUweight = 1 / OFL
# MTUweight = OFL * TSL
# MTUweight = OFL * np.sin(OPA) * TSL
# MTUweight = TSL / OFL # good one particularly with p3
# MTUweight = TSL / OFL*np.cos(OPA) # good one
# MTUweight = (volume * TSL) / (OFL*np.cos(OPA))
# MTUweight = PCSA * TSL / np.cos(OPA)
# MTUweight = PCSA * TSL / length
# MTUweight = PCSA
# MTUweight = PCSA / OFL
# MTUweight = PCSA * TSL / OFL # bad in Gmin
# MTUweight = PCSA * TSL
# MTUweight = (np.sqrt(PCSA)*TSL)
# MTUweight = np.sqrt(PCSA) * TSL / OFL
# MTUweight = np.sqrt(PCSA) * TSL / length
