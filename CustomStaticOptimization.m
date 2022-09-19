%% CustomStaticOptimization
close all; clear all; clc;

import org.opensim.modeling.*;

% Load data into MATLAB arrays.
q = readExp('Kinematics_q.sto');
u = readExp('Kinematics_u.sto');
m = readExp('inverse_dynamics.sto');
model = Model('subject01_simbody.osim');
state = model.initSystem();
muscles = model.updMuscles();
nMuscles = muscles.getSize();
nameMuscles = {};
for i=1:nMuscles
    nameMuscles(i) = muscles.get(i-1).getName();
end

coordinates = model.getCoordinateSet();
nCoordinates = coordinates.getSize();
nameCoordinates = {};
for i=1:nCoordinates
    nameCoordinates(i) = coordinates.get(i-1).getName();
end

time = q.time;
timeBool = find((0.62<=time) & (time<=1.404));
time = time(timeBool);
frame = length(time);
q = rmfield(q, 'time');
u = rmfield(u, 'time');
m = rmfield(m, 'time');

q2 = zeros(frame, nCoordinates);
u2 = zeros(frame, nCoordinates);
m2 = zeros(frame, nCoordinates);

for i=1:nCoordinates
    q.(nameCoordinates{i}) = q.(nameCoordinates{i})(timeBool);
    u.(nameCoordinates{i}) = u.(nameCoordinates{i})(timeBool);
    q2(:,i) = deg2rad(q.(nameCoordinates{i}));
    u2(:,i) = deg2rad(u.(nameCoordinates{i}));
    try
        m.(nameCoordinates{i}) = m.([nameCoordinates{i},'_force'])(timeBool);
        m = rmfield(m , [nameCoordinates{i},'_force']);
        m2(:,i) = m.(nameCoordinates{i});
    catch
        m.(nameCoordinates{i}) = m.([nameCoordinates{i},'_moment'])(timeBool);
        m = rmfield(m , [nameCoordinates{i},'_moment']);
        m2(:,i) = m.(nameCoordinates{i});
    end
end


MIF= zeros(1,nMuscles); % muscle maximum isometric force
L  = zeros(frame, nMuscles); % muscle length
FL = zeros(frame, nMuscles); % fiber length
TL = zeros(frame, nMuscles); % tendon length
OFL= zeros(1,nMuscles); % optimal fiber length
TSL= zeros(1,nMuscles); % tendon slack length
S  = zeros(frame, nMuscles); % muscle strength
FF = zeros(frame, nMuscles); % fiber force
AFF= zeros(frame, nMuscles); % active fiber force
PFF= zeros(frame, nMuscles); % passive fiber force
TF = zeros(frame, nMuscles); % tendon force
MA = zeros(frame, nCoordinates, nMuscles); % muscle moment arm

% store muscle parameters
for i=1:frame
	fprintf('Muscle Parameters ... at %f\n', time(i))
    
	Q = Vector(); Q.resize(nCoordinates);
	U = Vector(); U.resize(nCoordinates);
    for j=1:nCoordinates
		Q.set(j-1, q2(i,j));
		U.set(j-1, u2(i,j));
    end
	state.setQ(Q);
	state.setU(U);
    
%     model.realizePosition(state);
%     model.realizeVelocity(state);
%     model.equilibrateMuscles(state);
    model.realizeDynamics(state);

    for j=1:nMuscles
        muscle = Millard2012EquilibriumMuscle.safeDownCast(muscles.get(j-1));
        muscle.set_ignore_activation_dynamics(true); % disable activation dynamics
        muscle.set_ignore_tendon_compliance(true); % disable tendon compliance
%         
%         muscle.computeFiberEquilibrium(state);
%         muscle.computeEquilibrium(state);
%         muscle.computeActuation(state);
        model.equilibrateMuscles(state);

        L(i,j)  = muscle.getLength(state);
        FL(i,j) = muscle.getFiberLength(state);
        TL(i,j) = muscle.getTendonLength(state);
        OFL(j)  = muscle.getOptimalFiberLength();
        TSL(j)  = muscle.getTendonSlackLength();
        TL(i,j) = muscle.getTendonLength(state);
        MIF(j)  = muscle.getMaxIsometricForce();
        FF(i,j) = muscle.getFiberForce(state);
        AFF(i,j)= muscle.getActiveFiberForce(state);
        PFF(i,j)= muscle.getPassiveFiberForce(state);
        TF(i,j) = muscle.getTendonForce(state);

        muscle.setActivation(state, 1);
        S(i,j) = muscle.getFiberForce(state);

        for k=1:nCoordinates
			coordinate = coordinates.get(k-1);
			coordinate.setValue(state, q2(i,k))
			MA(i,k,j) = muscle.computeMomentArm(state, coordinate);
        end
    end
end

% plot(S(:,34))
% 
% plot(MA(:,10,28))

%% Static Optimization.
options_sqp = optimoptions('fmincon','Display','notify-detailed', ...
     'TolCon',1e-4,'TolFun',1e-12,'TolX',1e-8,'MaxFunEvals',20000,...
     'MaxIter',5000,'Algorithm','sqp');
options_ip = optimoptions('fmincon','Display','notify-detailed', ...
     'TolCon',1e-4,'TolFun',1e-12,'TolX',1e-8,'MaxFunEvals',1000000,...
     'MaxIter',10000,'Algorithm','interior-point');

activity = zeros(1, nMuscles);
force = zeros(1, nMuscles);
init = 0.1 * ones(1, nMuscles); % initial guess of activation
lb = zeros(1, nMuscles); % lower bound (min muscle activity = 0)
ub =  ones(1, nMuscles); % upper bound (max muscle activity = 1)
costFunc = @(a) sum(a.^2); % sum of squared muscle activity
A = []; % Linear inequality constraints (A=matrix and b=array)
b = [];
nonlcon = []; % Nonlinear constraints (function)

for i = 1:frame
    fprintf('Optimizing...time step %i/%i \n',  i, frame);

	% Linear equality constraints (Aeq=matrix and beq=array)
    Aeq = squeeze(MA(i,:,:)).*S(i,:);   % moment arm * strength
    beq = m2(i,:)';

    a = fmincon(costFunc, init, A, b, Aeq, beq, lb, ub, nonlcon, options_sqp);
%     init = a;

	activity(i,:) = a;
    force(i,:) = S(i,:) .* a;
    
end

% plot(activity(:,34))
plot(force(:,34))

%%
function struct = readExp(file)
    % function [data, label] = readExp(file)
    %Read OpenSim STO and MOT files Or any other format that 
    %the headers are separated from labels and data by 'endheader' line.
    %can also return struct'''

    i = 0;
    while true
        try 
            data = dlmread(file, '\t', i, 0);
            row = i;
            break
        catch
            i = i+1;
        end
    end
    % sprintf('Label row number is: %d', row)

    fid = fopen(file);
    for r=1:(row-1)
        fgetl(fid);
    end
    label = fgetl(fid);
    fclose(fid);
    label = strsplit(label, '\t');

    for i = 1:length(label)
        struct.(char(label(i))) = data(:,i);
    end
end

