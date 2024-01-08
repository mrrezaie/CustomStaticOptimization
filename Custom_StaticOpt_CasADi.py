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

FC = 7

import numpy as np
import matplotlib.pyplot as plt
import os
import casadi
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
			coordinate.setValue(state, j, enforceContraints=False)
			model.assemble(state)
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
include = list()
for i,cName in enumerate(nameCoordinatesM):
	c1 = not cName in exclude
	c2 = not cName in unfree
	c3 = not cName in empty
	if c1 and c2 and c3:
		indxCoordinates.append(i)
		include.append(cName)
	# 	print('include', cName)
	# else:
	# 	print('exclude', cName)

print(f"Excluded coordinates: \n\t{' '.join(exclude)}\n")
print(f"Unfree coordinates: \n\t{' '.join(unfree)}\n")
print(f"No muscles coordinates: \n\t{' '.join(empty)}\n")
print(f"Included coordinates: \n\t{' '.join(include)}\n")

########## Get initial muscles properties
MIF = np.empty(nMuscles) # maximum isometric force
OFL = np.empty(nMuscles) # optimal fiber length
TSL = np.empty(nMuscles) # tendon slack length
OPA = np.empty(nMuscles) # pennation angle at optimal fiber length

rigidTendon, compliantTendon = list(), list()
for mi,muscle in enumerate(model.getMuscles()):
	# muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
	# muscle.setMaxIsometricForce( 0.5* muscle.getMaxIsometricForce())
	mName = muscle.getName()

	MIF[mi] = muscle.getMaxIsometricForce()
	OFL[mi] = muscle.getOptimalFiberLength()
	TSL[mi] = muscle.getTendonSlackLength()
	OPA[mi] = muscle.getPennationAngleAtOptimalFiberLength()

	muscle.set_ignore_activation_dynamics(False) # activation dynamics (have no impact)
	muscle.set_ignore_tendon_compliance(False)   # compliant tendon
	# muscle.set_ignore_tendon_compliance(True)    # rigid tendon

	if muscle.getTendonSlackLength() < muscle.getOptimalFiberLength():
		muscle.set_ignore_tendon_compliance(True) # rigid tendon
		rigidTendon.append(mName)
		# print('r i g i d tendon:', muscle.getName())
	else:
		muscle.set_ignore_tendon_compliance(False) # compliant tendon
		compliantTendon.append(mName)
		# print('compliant tendon:', muscle.getName())

print(f"Rigid Tendons: \n\t{' '.join(rigidTendon)}\n")
print(f"Compliant Tendons: \n\t{' '.join(compliantTendon)}\n")

state = model.initSystem() # the size is subject to the tendon models


########## read Ik and ID files (coordinates value and generalized forces)

# read IK and ID files
IKFile = osim.TimeSeriesTable(IKName)
IDFile = osim.TimeSeriesTable(IDName)

# process the tables
for table in [IKFile,IDFile]:
	timeColumn = table.getIndependentColumn() # time
	osim.TableUtilities().filterLowpass(table, FC, padData=True) # Butterworthlow pass filter (3rd order)
	table.trim(timeColumn[0], timeColumn[-1]) # remove padding

# convert IK degrees to radians
if IKFile.getTableMetaDataString('inDegrees') == 'yes':
	model.getSimbodyEngine().convertDegreesToRadians(IKFile) # convert from degrees to radians
	print('Coordinates were converted to Radians\n')

# generate times
fs = round(1/np.diff(timeColumn).mean())
dt = 1/fs
nTimes = round((cycle[1]-cycle[0]) * fs) + 1
times = np.linspace(cycle[0], cycle[1], nTimes)


########## calculate speeds and accelerations
q,u,u_dot,tau = [osim.TimeSeriesTable(times) for _ in range(4)]

# GCVSplineSet helps fix time irregularity and inconsistency
IKGCVSS = osim.GCVSplineSet(IKFile, [], 5, 0) # degree=5
IDGCVSS = osim.GCVSplineSet(IDFile, [], 5, 0) # degree=5
# d2 = osim.StdVectorInt(); d2.push_back(0); d2.push_back(0) # second derivative

for cName in nameCoordinatesM: # in multibody tree order

	GCVS = IKGCVSS.get(cName)
	q.appendColumn(cName, osim.Vector([GCVS.calcValue(osim.Vector(1,time)) for time in times]) )
	u.appendColumn(cName, osim.Vector([GCVS.calcDerivative([0], osim.Vector(1,time)) for time in times]) )
	u_dot.appendColumn(cName, osim.Vector([GCVS.calcDerivative([0,0], osim.Vector(1,time)) for time in times]) )

	if IDFile.hasColumn(cName+'_moment'):
		cNameID = cName+'_moment'
	elif IDFile.hasColumn(cName+'_force'):
		cNameID = cName+'_force'	
	GCVS = IDGCVSS.get(cNameID)
	tau.appendColumn(cName, osim.Vector([GCVS.calcValue(osim.Vector(1,time)) for time in times]) )

del IDFile, IKFile, timeColumn, IKGCVSS, IDGCVSS, GCVS





########## get muscle properties at each time step 
# muscle-tendon length
# cosine pennation angle
# active force length multiplier
# passive force multiplier
# force velocity multiplier
# fiber length
# moment arm

MTL,CPA,FLM,PFM,FVM,ML = [osim.TimeSeriesTable() for _ in range(6)]
for i in [MTL,CPA,FLM,PFM,FVM,ML]:
	i.setColumnLabels(nameMuscles)

MA = dict()
for cName in include: # included coordinates
	MA[cName] = osim.TimeSeriesTable()
	# MA[cName].setColumnLabels(coordinateMuscles[cName])
	MA[cName].setColumnLabels(nameMuscles)

timeStart = absoluteTime()

for ti,time in enumerate(times):
	# print(f'Muscle Parameters ... {(ti+1):0>3d}/{len(times):0>3d} ({round(time,3):.3f})')
	state.setTime(time)

	##### Update coordinates' values and speeds
	for coordinate in multibodyOrder:
		cName = coordinate.getName()
		coordinate.setValue(state, q.getDependentColumn(cName)[ti], enforceContraints=False)
		coordinate.setSpeedValue(state, u.getDependentColumn(cName)[ti])

	model.assemble(state)

	# model.realizePosition(state)
	model.realizeVelocity(state)

	# _MTL,_CPA,_FLM,_PFM,_FVM,_ML = [list() for _ in range(6)]

	for muscle in model.getMuscles():
		muscle.setActivation(state, 1)
		# muscle.computeEquilibrium(state)

	# 	_MTL.append(muscle.getLength(state))
	# 	_CPA.append(muscle.getCosPennationAngle(state))
	# 	_FLM.append(muscle.getActiveForceLengthMultiplier(state))
	# 	_PFM.append(muscle.getPassiveForceMultiplier(state))
	# 	_FVM.append(muscle.getForceVelocityMultiplier(state))
	# 	_ML.append( muscle.getFiberLength(state))

	# for table,temp in zip([MTL,CPA,FLM,PFM,FVM,ML], [_MTL,_CPA,_FLM,_PFM,_FVM,_ML]):
	# 	table.appendRow(time, osim.RowVector(temp) )

	model.equilibrateMuscles(state)

	MTL.appendRow(time, osim.RowVector( [muscle.getLength(state) for muscle in model.getMuscles()] ))
	CPA.appendRow(time, osim.RowVector( [muscle.getCosPennationAngle(state) for muscle in model.getMuscles()] ))
	FLM.appendRow(time, osim.RowVector( [muscle.getActiveForceLengthMultiplier(state) for muscle in model.getMuscles()] ))
	PFM.appendRow(time, osim.RowVector( [muscle.getPassiveForceMultiplier(state) for muscle in model.getMuscles()] ))
	FVM.appendRow(time, osim.RowVector( [muscle.getForceVelocityMultiplier(state) for muscle in model.getMuscles()] ))
	ML.appendRow( time, osim.RowVector( [muscle.getFiberLength(state) for muscle in model.getMuscles()] ))

	# for cName in include:
	# 	coordinate = model.getCoordinateSet().get(cName)
	# 	row = list()
	# 	for mName in coordinateMuscles[cName]:
	# 		muscle = model.getMuscles().get(mName)
	# 		row.append(muscle.computeMomentArm(state, coordinate))
	# 	MA[cName].appendRow(time, osim.RowVector(row))

	for cName in include:
		coordinate = model.getCoordinateSet().get(cName)
		row = np.zeros(nMuscles)
		for mi, mName in enumerate(nameMuscles):
			if mName in coordinateMuscles[cName]:
				muscle = model.getMuscles().get(mName)
				row[mi] = muscle.computeMomentArm(state, coordinate)
		MA[cName].appendRow(time, osim.RowVector(row))

print(f'Muscles parameters extraction ... finished in {absoluteTime()-timeStart:.2f} s\n')


# smoothing
for table in [MTL,CPA,FLM,PFM,FVM,ML] + list(MA.values()):
	timeColumn = table.getIndependentColumn() # time
	osim.TableUtilities().filterLowpass(table, FC, padData=True) # Butterworthlow pass filter (3rd order)
	table.trim(timeColumn[0], timeColumn[-1]) # remove padding

# # muscle-tendon velocity
# MTV = osim.TimeSeriesTable(times)
# GCVSS = osim.GCVSplineSet(MTL, [], 5, 0) # degree=5
# for mName in nameMuscles: # in multibody tree order
# 	GCVS = GCVSS.get(mName)
# 	MTV.appendColumn(mName, osim.Vector([GCVS.calcDerivative([0], osim.Vector(1,time)) for time in times]) )

# update tables' metadata
for table in [q,u,u_dot,tau] + [MTL,MTL,CPA,FLM,PFM,FVM,ML] + list(MA.values()):
	table.addTableMetaDataString('inDegrees', 'no')
	table.addTableMetaDataString('nColumns', str(table.getNumColumns()))
	table.addTableMetaDataString('nRows',    str(table.getNumRows()))

# for cName,table in MA.items():
# 	# print(key)
# 	plt.figure(cName)
# 	plt.plot(table.getMatrix().to_numpy())
# 	plt.legend(table.getColumnLabels())
# 	plt.show(block=False)

# for mName in FLM.getColumnLabels():
# 	plt.figure(mName)
# 	plt.plot(FLM.getDependentColumn(mName))
# 	plt.show(block=False)

# for cName in include:
# 	plt.figure(cName)
# 	plt.plot(q.getDependentColumn(cName))
# 	plt.plot(u.getDependentColumn(cName))
# 	plt.plot(u_dot.getDependentColumn(cName))
# 	plt.show(block=False)


########## Add external load file to the model
# for joint contact force analysis
GRF = osim.Storage(GRFName)
for i in osim.ForceSet(ExtLName):
	exForce = osim.ExternalForce.safeDownCast(i)
	exForce.setDataSource(GRF)
	model.getForceSet().cloneAndAppend(exForce)


########## Add actuators to coordinates without muscle

nActuators = 0
indxActuators = list() # index of coordinates with actuator

# for i,coordinate in enumerate(multibodyOrder):
# 	cName = coordinate.getName()
# 	c1 = cName in empty
# 	c2 = not cName in exclude
# 	c3 = not cName in unfree
# 	if c1 and c2 and c3:

for ci,cName in enumerate(empty):

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

print(f"{nActuators} coordinate actuators for \n\t{' '.join(empty)}\n\n")

state  = model.initSystem()
assert model.getNumControls() == (nMuscles+nActuators)


########## optimization
activeElement  = MIF*FLM.getMatrix().to_numpy() \
					*FVM.getMatrix().to_numpy() \
					*CPA.getMatrix().to_numpy()
passiveElement = MIF*PFM.getMatrix().to_numpy() \
					*CPA.getMatrix().to_numpy()

SOpt = casadi.Opti()
x = SOpt.variable(nTimes,nMuscles)
SOpt.set_initial(x, 0.1)
SOpt.subject_to( SOpt.bounded(0,x,1) ) # bounds

strength = x * activeElement + passiveElement

# equality constraint
# calculate joint moment differences for each coordinate
momentWeigth = np.empty((nTimes,nMuscles))
for ci,cName in enumerate(include):
	_moment = tau.getDependentColumn(cName).to_numpy()
	_momentArm = MA[cName].getMatrix().to_numpy()
	SOpt.subject_to( casadi.sum2(_momentArm*strength) == _moment )
	momentWeigth = np.add(momentWeigth, np.abs(_momentArm)*strength)

PCSA = MIF/60
volumeWeigth = np.repeat([PCSA*OFL], nTimes, axis=0) # (nTimes,nMuscles)

obj = x**2 * volumeWeigth
# obj = x**2 * momentWeigth

SOpt.minimize( casadi.sum1(casadi.sum2(obj)) )
# SOpt.minimize( casadi.sumsqr(x) )


# optimizer settings are based on MuscleRedundancySolver
p_opts = {"expand":True}
s_opts = {"max_iter":1000, 'linear_solver':'mumps', 'tol':1e-6, 'nlp_scaling_method':'gradient-based'}
SOpt.solver('ipopt',p_opts,s_opts)
solver = SOpt.solve()
output = solver.value(x)
# output.shape

print(f'\nMin activation {np.min(output).round(2)}')
print(f'Max activation {np.max(output).round(2)}\n')

# fix round-off error
if (output<0).any():
	output[output<0]=0
if (output>1).any():
	output[output>1]=1

########## Output variables
activity,force,stateData = [osim.TimeSeriesTable(times) for _ in range(3)]

for mi,mName in enumerate(nameMuscles):
	activity.appendColumn(mName, osim.Vector(output[:,mi]))
	force.appendColumn(   mName, osim.Vector(output[:,mi] * activeElement[:,mi] + passiveElement[:,mi]))

for i in multibodyOrder:
	stateData.appendColumn(i.getAbsolutePathString()+'/value', q.getDependentColumn(i.getName()))
	stateData.appendColumn(i.getAbsolutePathString()+'/speed', u_dot.getDependentColumn(i.getName()))
for i in model.getMuscles():
	stateData.appendColumn(i.getAbsolutePathString()+'/fiber_length', ML.getDependentColumn(i.getName()))
	stateData.appendColumn(i.getAbsolutePathString()+'/activation', activity.getDependentColumn(i.getName()))



######### Joint Reaction Analysis

state  = model.initSystem()

reaction = osim.TimeSeriesTableVec3()
reaction.setColumnLabels(nameJoints) # StdVectorString
ground = model.getGround()

timeStart = absoluteTime()
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

	for muscle in model.getMuscles():
		mName = muscle.getName()
		muscle.setActivation(state, activity.getDependentColumn(mName)[ti])

	# controls = list() # mActivation.tolist() + 
	# for controller in model.getControllerSet():
	# 	cName = controller.getName()[:-11]
	# 	values = aActivation # m.getDependentColumn(cName)[ti]
	# 	controls.append(values)
	# 	# print(cName)
	# 	PC    = osim.PrescribedController.safeDownCast(controller)
	# 	const = osim.Constant.safeDownCast(PC.get_ControlFunctions(0).get(0))
	# 	const.setValue(values)

	# aActivation = tau.getRowAtIndex(ti).to_numpy()[indxActuators]
	mActivation = [activity.getDependentColumn(mName)[ti] for mName in nameMuscles]
	aActivation = [tau.getDependentColumn(cName)[ti] for cName in empty]

	model.setControls(state, osim.Vector(mActivation + aActivation))
	model.equilibrateMuscles(state)
	model.realizeAcceleration(state)

	row = list()
	for j,joint in enumerate(model.getJointSet()):
		reactionGround = joint.calcReactionOnChildExpressedInGround(state)
		reactionForce  = reactionGround.get(1) # 0==moment, 1==force
		jointChildBody = joint.getChildFrame().findBaseFrame() # body frame not joint frame
		row.append(ground.expressVectorInAnotherFrame(state, reactionForce, jointChildBody))
	reaction.appendRow(time, osim.RowVectorVec3(row))

print(f'Joint contact force analysis ... finished in {absoluteTime()-timeStart:.2f} s\n')


########## write output to sto files
reaction = reaction.flatten(['_x','_y','_z'])

for table in [reaction,activity,force,stateData]:
	table.addTableMetaDataString('inDegrees', 'no')
	table.addTableMetaDataString('nColumns', str(table.getNumColumns()))
	table.addTableMetaDataString('nRows',    str(table.getNumRows()))

# # # osim.STOFileAdapter().write(reaction, 'output/jointReaction.sto')
# # # osim.STOFileAdapter().write(activity, 'output/activity.sto')
# # # osim.STOFileAdapter().write(force,    'output/force.sto')
# osim.STOFileAdapter().write(stateData,    'output/state.sto')


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
print('Second joint contact force peak:')
for i in ['ankle_r_y', 'walker_knee_r_y', 'hip_r_y']:
	signal = -1*reaction.getDependentColumn(i).to_numpy()/ (weight)
	print('\t',i, np.max(signal[40:80]).round(2))


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

		print('\t',key, np.std(W.sum(axis=1), ddof=1).round(3))



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
# viz.showMotion(model, stateData)
