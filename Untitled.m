clear all;
close all;
clc

n_input=4;%input('no of inputs');

n_op=3;

n_hid=4;%input('enter the no of hidden layer nodes');

w_ih=rand(n_input,n_hid)

w_bhid=rand(n_hid,1)

w_ho=rand(n_hid,n_op)

% w_bout=rand

eta=1;

n_pattern=3;


load fisheriris


%define training set

in=[];

for i=1:40
    
     temp=[meas(i,:);meas(50+i,:);meas(100+i,:)]';
     
     in=[in temp];

end

% coding (+1/-1) of 3 classes

a = [0.1 0.1 0.9]';
b = [0.1 0.9 0.1]';
c = [0.9 0.1 0.1]';

% define targets
    
temp = [repmat(a,1,1) repmat(b,1,1) repmat(c,1,1)];

desired_out=repmat(temp,[1 40]);

% in=[.1 .1 .9 .9;.1 .9 .1 .9]

% desired_out=[.1 .9 .9 .1]

iteration=10000;


error=zeros(n_op,iteration);

% keyboard

for iter=1:iteration
    
    for j=1:size(in,2)
        
        %estimated output
        
        op_w=in(:,j)'*w_ih;
        
        op_sig=1./(1+exp(-(op_w+w_bhid')));
        
        out=1./(1+exp(-(op_sig*w_ho)));
        
        e=desired_out(:,j)'-out;
        
        delta=(out.*(1-out)).*e;
       
        %hidden layer weights updation
        
        w_ho=w_ho+eta*op_sig'*delta;
        
        
        
        
%         w_bout=w_bout+2*delta;

        delta_hid=op_sig'.*(1-op_sig)'.*(w_ho*delta');
        
        %input layer weight updations
        
        w_ih=w_ih+eta*(in(:,j)*delta_hid');  
        
        w_bhid=w_bhid+2*delta_hid;
        
    end
    
    iter
    
    error(:,iter)=e;
    
end



sse=sum((error(:,1:iter).^2),1);

plot(sse);
title('error square plot for xor gate training');
xlabel('no of iterations');
ylabel('error.^2');

keyboard

%%%%%%%testing%%%%%%%%%


in1 = [meas(41:50,:);meas(91:100,:);meas(141:150,:)]'

out = [];

% in=[.1 .1 .9 .9;.1 .9 .1 .9];

 for i=1:size(in1,2)
% display('testing for input combination:')
%     in1
%     
    
    op_w=in1(:,i)'*w_ih;
    op_sig=1./(1+exp(-(op_w+w_bhid')));
    out(:,i)=(1./(1+exp(-(op_sig*w_ho))))';
    
 end

 out