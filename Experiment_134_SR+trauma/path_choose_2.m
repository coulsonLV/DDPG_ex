function [sys,x0,str,ts,simStateCompliance] = path_choose_2(t,x,u,flag)

%
% The following outlines the general structure of an S-function.
%
switch flag
    
    %%%%%%%%%%%%%%%%%%
    % Initialization %
    %%%%%%%%%%%%%%%%%%
    case 0
        [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes;
        
        %%%%%%%%%%%%%%%
        % Derivatives %
        %%%%%%%%%%%%%%%
    case 1
        sys=mdlDerivatives(t,x,u);
        
        %%%%%%%%%%
        % Update %
        %%%%%%%%%%
    case 2
        sys=mdlUpdate(t,x,u);
        
        %%%%%%%%%%%
        % Outputs %
        %%%%%%%%%%%
    case 3
        sys=mdlOutputs(t,x,u);
        
        %%%%%%%%%%%%%%%%%%%%%%%
        % GetTimeOfNextVarHit %
        %%%%%%%%%%%%%%%%%%%%%%%
    case 4
        sys=mdlGetTimeOfNextVarHit(t,x,u);
        
        %%%%%%%%%%%%%
        % Terminate %
        %%%%%%%%%%%%%
    case 9
        sys=mdlTerminate(t,x,u);
        
        %%%%%%%%%%%%%%%%%%%%
        % Unexpected flags %
        %%%%%%%%%%%%%%%%%%%%
    otherwise
        DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
        
end


function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes


sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 1;
sizes.NumOutputs     = 15;
sizes.NumInputs      = 18;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);


x0  = [0];


str = [];


ts  = [0 0];


simStateCompliance = 'UnknownSimState';


function sys=mdlDerivatives(~,~,~)

sys = [];


function sys=mdlUpdate(~,~,~)

sys = [];


function sys=mdlOutputs(~,~,u)

vx = u(1);
action = u(2);
row = u(3);
drow = u(4);
x = u(5);
y = u(6);
x1 = u(7);
y1 = u(8);
t  = u(9);
k_sum = u(10);
a = u(11);
b = u(12);
c = u(13);
d = u(14);
e = u(15);
g = u(16);
sf= u(17);
pf= u(18);
s0= x-x1;
p0= y-y1;
t1 = mod(t,1);

if t1==0
pf = action -  y1;
sf = vx*3+s0;
X = [1,  s0,  s0^2,   s0^3,   s0^4,   s0^5;
     1,  sf,  sf^2,   sf^3,   sf^4,   sf^5;
     0,   1,  2*s0, 3*s0^2, 4*s0^3, 5*s0^4;
     0,   1,  2*sf, 3*sf^2, 4*sf^3, 5*sf^4;
     0,   0,     2,   6*s0,12*s0^2, 20*s0^3;
     0,   0,     2,   6*sf,12*sf^2, 20*sf^3  ];
Y = [ p0;   pf ;  row ;  0  ; drow ;  0 ];
            A1 = X\Y;
            a = A1(6);
            b = A1(5);
            c = A1(4);
            d = A1(3);
            e = A1(2);
            g = A1(1);
%       for s1=s0:1:sf
%           p1= a*s1^5 + b*s1^4 + c*s1^3 + d*s1^2+e*s1+g;
%       plot(s1,p1,'b.')
%       end
end  
    k00=0;
    for i= 1:6
        s = s0 + (0.2*i-0.2)*vx;
        if s>=sf
           p=pf;
           dp=0;
           ddp = 0;
        else
        p = a*s^5 + b*s^4 + c*s^3 + d*s^2+e*s+g;
        dp = 5*a*s^4 + 4*b*s^3 + 3*c*s^2+2*d*s+e;
        ddp = 20*a*s^3 + 12*b*s^2+6*c*s+2*d;
        end
        A = sqrt(dp^2 + (1 - p*k00)^2);
        B = sign(1 - p*k00);
        k(i,1) = B/A*(k00 + ((1 - p*k00)*ddp + k00*dp^2)/(A^2));
       
    end
    k_sum = 0.3*k(2,1)+0.2*k(3,1)+0.1*k(4,1)+0.1*k(5,1)+0.3*k(6,1);
    raw0 = k(1,1);
    raw1 = k(2,1);
    raw2 = k(3,1);
    raw3 = k(4,1);
    raw4 = k(5,1);
    raw5 = k(6,1);
     
    




  
sys = [a,b,c,d,e,g,k_sum,raw0,raw1,raw2,raw3,raw4,raw5,sf,pf];


function sys=mdlGetTimeOfNextVarHit(t,~,~)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;


function sys=mdlTerminate(~,~,~)

sys = [];

