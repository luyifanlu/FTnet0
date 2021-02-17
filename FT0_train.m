function [W,V,error_train,Memory,Partial_W,Partial_V] = FT0_train(setting,order_FT0,XTrain,YTrain,input_dim,output_dim,alpha,beta,eta,timestep,iteration)

% Decide on task types
if setting == 'p'    % prediction 
    disp('FT0 prediction step 2');
end
if setting == 'c'    % classification, only depends on the activation of the final layer
    disp('FT0 classification step 2')
end

order = order_FT0;

% initialize neural network weights
W = 2*rand(output_dim,input_dim) - 1;   
V = 2*rand(output_dim,output_dim) - 1;    

W_update = zeros(size(W));
V_update = zeros(size(V));

% initialize memory units
memory_value = 2*rand(1,output_dim)-1;

% initialize partial values
partialW = cell(output_dim,1);
partialV = cell(output_dim,1);
for jj = 1:output_dim
    partialW{jj,1} = zeros(output_dim,input_dim); % partial M_jj / partial W
    partialV{jj,1} = zeros(output_dim,output_dim);   % partial M_jj / partial V
end

%% Train a FT0
[n,~] = size(XTrain);
sampling = n * iteration;  % the number of total samples, including iteration
for kk = 1:sampling
    sample = mod( kk,n ); 
    if sample == 0
        sample = n;
    end
    if sample - timestep >= 0
        X = XTrain( sample-timestep+1:sample,: );
        Y = YTrain( sample-timestep+1:sample,: );
    else
        X = XTrain( 1:sample,: );
        Y = YTrain( 1:sample,: );
    end
    
    pW = partialW; 
    pV = partialV;
    overallError = 0; 
    out_diff = []; 
    for time = 1:length(Y) 
        % feedforward
        input = X(time,:); 
        in(1,:) = alpha*input*W' - beta*memory_value(end,:)*V;
        in(2,:) = beta*input*W' + alpha*memory_value(end,:)*V';
        [feed,der] = complexReLU(in,order);
        out = feed(1,:);
        out_ad = der;
        memory = feed(2,:);
        
        memory_value = [memory_value;memory];
        prev_memory = memory_value(end-1,:);
        [pW,pV] = partial_forward(pW,pV,V,alpha,beta,memory,input,prev_memory,der,time);
        
        error_out = Y(time,:) - out;
        out_diff = [out_diff; error_out .* out_ad];
        overallError = overallError + 0.5*sum(error_out.^2); 
    end
    
    for time = 1:length(Y)
        position = length(Y)-time; 
        input = X(end-position,:); 
        prev_memory = memory_value(end-position-1,:);
        hid_diff = out_diff(end-position,:); 

        % error at W and V = time direction
        % partial memory causes errors
        [W_real,W_imag,V_real,V_imag] = partial_update(input,prev_memory,out,pW,pV,V,position ) ;
        % time direction causes errors
        delta_W = hid_diff' .* ( alpha * W_real - beta * W_imag );
        delta_V = hid_diff' .* ( - beta * ( V_real + V_imag ) );
        
        % update at time = position
        W_update = W_update + delta_W;
        V_update = V_update + delta_V;
    end
    % update parameters
    W = W + W_update * eta;
    V = V + V_update * eta;
    
    W_update = W_update * 0;
    V_update = V_update * 0;
    
    error_train(kk) = overallError;
    clear overallError;
end
Memory = memory_value(end,:);
Partial_W = pW{:,end};
Partial_V = pV{:,end};
end

%% Sub-Codings
function [pW,pV] = partial_forward(pW,pV,V,alpha,beta,memory,prev_stimulus,prev_memory,der,time)
for ii = 1:length(memory)
            temp1 = zeros( length(memory),length(prev_stimulus) );
            temp2 = zeros( length(memory),length(memory) );
            for cc = 1:length(memory)
                temp1 = temp1 + V(ii,cc) * pW{ii,time};
                temp2 = temp2 + V(ii,cc) * pV{ii,time};
            end
            add = zeros( size(temp1) );
            add(ii,:) = prev_stimulus;
            pW{ii,time+1} = der(ii)*[ beta*add + alpha*temp1 ];
            add = zeros( size(temp2) );
            add(ii,ii) = prev_memory(ii);
            pV{ii,time+1} = der(ii)*alpha*[ repmat(prev_memory,length(memory),1)+ temp2 ];
end
end

function [W_real,W_imag,V_real,V_imag] = partial_update(prev_stimulus,prev_memory,stimulus,pW,pV,V,position )      
% error at W and V = time direction ( partial memory )
W_real = repmat( prev_stimulus,length(stimulus),1 );
V_real = repmat( prev_memory,length(stimulus),1 );
        W_imag = zeros(length(stimulus),length(prev_stimulus));
        V_imag = zeros(length(stimulus),length(stimulus));
        for ii = 1:length(stimulus)
            temp1 = zeros(1,length(prev_stimulus));
            temp2 = zeros(1,length(stimulus));
            for cc = 1:length(stimulus)
                W_imag(ii,:) = temp1 + V(ii,cc) * pW{cc,end-position-1}(ii,:);
                V_imag(ii,:) = temp2 + V(ii,cc) * pV{cc,end-position-1}(ii,:);
            end
        end
end