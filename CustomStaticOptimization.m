%% CustomStaticOptimization
close all; clear all; clc;
import org.opensim.modeling.*;

cycle = [1.42, 2.16]; % desired time
% Load data as struct
q = readExp('_Kinematics_q.sto');
u = readExp('_Kinematics_u.sto');
m = readExp('inverse_dynamics.sto');

model = Model('scaled.osim');
state = model.initSystem();
% get muscles
muscles = model.getMuscles();
nMuscles = muscles.getSize();
nameMuscles = {};
for i=1:nMuscles
    nameMuscles(i) = muscles.get(i-1).getName();
end

% get coordinates
coordinates = model.getCoordinateSet();
nCoordinates = coordinates.getSize();
nameCoordinates = {};
for i=1:nCoordinates
    nameCoordinates(i) = coordinates.get(i-1).getName();
end

time = q.time;
timeBool = find((cycle(1)<=time) & (time<=cycle(2)));
time = time(timeBool);
fs = round(1/mean(diff(time))); % sampling frequency
frame = length(time);

q = rmfield(q, 'time');
u = rmfield(u, 'time');
m = rmfield(m, 'time');

% convert input data as MATLAB array
q2 = zeros(frame, nCoordinates);
u2 = zeros(frame, nCoordinates);
m2 = zeros(frame, nCoordinates);

fc = 7; % cut-off frequency
[b,a] = butter(4, 2*fc/fs, 'low');

% crop, filter and re-order the input files
for i=1:nCoordinates
    q.(nameCoordinates{i}) = q.(nameCoordinates{i})(timeBool);
    u.(nameCoordinates{i}) = u.(nameCoordinates{i})(timeBool);
    q2(:,i) = filtfilt(b,a, deg2rad(q.(nameCoordinates{i})));
    u2(:,i) = filtfilt(b,a, deg2rad(u.(nameCoordinates{i})));
    try
        m.(nameCoordinates{i}) = m.([nameCoordinates{i},'_force'])(timeBool);
        m = rmfield(m , [nameCoordinates{i},'_force']);
        m2(:,i) = filtfilt(b,a, m.(nameCoordinates{i}));
    catch
        m.(nameCoordinates{i}) = m.([nameCoordinates{i},'_moment'])(timeBool);
        m = rmfield(m , [nameCoordinates{i},'_moment']);
        m2(:,i) = filtfilt(b,a, m.(nameCoordinates{i}));
    end
end

% name of coordinates to exclude
exclude = {'pelvis', 'lumbar', 'beta', 'mtp', '_l'};
cBool = true(1,nCoordinates);
for i=1:nCoordinates
    for j=1:length(exclude) % exclude coordinates
        if contains(nameCoordinates{i}, exclude{j})
            cBool(i) = false;
        end
    end
end

% retrieve muscle parameters
MIF= zeros(1, nMuscles); % muscle maximum isometric force
FL = zeros(frame, nMuscles); % fiber length
FF = zeros(frame, nMuscles); % fiber force
MA = zeros(frame, nCoordinates, nMuscles); % muscle moment arm

tic
for i=1:frame
	fprintf('Muscle Parameters ... at %f\n', time(i))
    
    for j=1:nCoordinates
        model.updCoordinateSet().get(j-1).setValue(state, q2(i,j), false)
        model.updCoordinateSet().get(j-1).setSpeedValue(state, u2(i,j))
    end
    model.assemble(state)
    model.realizeVelocity(state);
%     model.realizeDynamics(state);
%     model.equilibrateMuscles(state);
    
    
    for j=1:nMuscles
        muscle = muscles.get(j-1);
%         muscle = Millard2012EquilibriumMuscle.safeDownCast(muscles.get(j-1));
%         muscle.set_ignore_tendon_compliance(true); % disable tendon compliance
%         muscle.set_ignore_activation_dynamics(true); % disable activation dynamics

        muscle.setActivation(state, 1); % muscle is fully activated
        muscle.computeEquilibrium(state);
        
        FL(i,j) = muscle.getFiberLength(state);
        FF(i,j) = muscle.getFiberForce(state);
        MIF(j) = muscle.getMaxIsometricForce();

        for k=1:nCoordinates
            if cBool(k) % I hope this increases the speed
                coordinate = model.getCoordinateSet().get(k-1);
                MA(i,k,j) = muscle.computeMomentArm(state, coordinate);
            end
        end
        
    end
end
toc

V = MIF .* FL; % muscle volume = strength * fiber length

% plot(time, MA(:,10, 30))
% figure()
% plot(time, FL(:, 30))
% figure()
% plot(time, FF(:, 30))
%% Static Optimization.

options_sqp = optimoptions('fmincon','Display','notify-detailed', ...
     'TolCon',1e-4,'TolFun',1e-12,'TolX',1e-8,'MaxFunEvals',20000,...
     'MaxIter',500,'Algorithm','sqp');

activity = zeros(1, nMuscles);
force = zeros(1, nMuscles);
init = 0.1 * ones(1, nMuscles); % initial guess of activation
lb = zeros(1, nMuscles) + 1e-3; % lower bound (min muscle activity ~= 0)
ub = ones(1, nMuscles) * inf;   % upper bound (max muscle activity >= 1)
A=[]; b=[]; % Linear inequality constraints (A=matrix and b=array)
nonlcon = []; % Nonlinear constraints (function)

tic
for i = 1:frame
    fprintf('Optimizing...time step %i/%i \n',  i, frame);
    
    volume = V(i,:); % muscle volume
	% Linear equality constraints (Aeq=matrix and beq=array)
    Aeq = squeeze(MA(i,cBool,:)) .* MIF; % moment arm * strength
    beq = m2(i,cBool); % joint moment

    % minimize sum of volume weighted squared muscle activity
    a = fmincon(@(a) sum(volume.*(a).^2), init, A, b, Aeq, beq, lb, ub, nonlcon, options_sqp);

	activity(i,:) = a;
    force(i,:) = MIF .* a; % muscle strength * muscle activity
end
toc

% plot(activity(:,34))
figure()
plot(time, activity(:,34), 'LineWidth',2)
hold on
plot(time, activity(:,40), 'LineWidth',2)
hold on
plot(time, activity(:,13), 'LineWidth',2)
hold on
plot(time, activity(:,14), 'LineWidth',2)
legend('soleus', 'peroneus longus', 'gast lat', 'gast med')
ylabel('activity')
xlabel('stance (time)')
% ylim([0,1])
% figure()
% plot(time, force(:,34), 'LineWidth',2)
% hold on
% plot(time, force(:,40), 'LineWidth',2)
% hold on
% plot(time, force(:,13), 'LineWidth',2)
% hold on
% plot(time, force(:,14), 'LineWidth',2)
% legend('soleus', 'peroneus longus', 'gast lat', 'gast med')
% ylabel('force')
% xlabel('stance (time)')


%%

function structData = readExp(file)
    % function [data, label] = readExp(file)
    %Read OpenSim STO and MOT files Or any other format that 
    %the headers are separated from labels and data by 'endheader' line.
    %can also return struct'''
    
    if ~exist(file)
        error('%s file does not exist', file)
    end
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
        structData.(char(label(i))) = data(:,i);
    end
end