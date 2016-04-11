function varargout = CustomerFeature_GUI(varargin)
% CUSTOMERFEATURE_GUI MATLAB code for CustomerFeature_GUI.fig

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CustomerFeature_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CustomerFeature_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
end
function CustomerFeature_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CustomerFeature_GUI (see VARARGIN)

% Choose default command line output for CustomerFeature_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CustomerFeature_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);
end
function varargout = CustomerFeature_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end
function popupmenu4_Callback(hObject, eventdata, handles)

fea=get(handles.popupmenu4,'Value');
if fea==1
    set(handles.edit6,'Visible','off');
    set(handles.text18,'Visible','off');
    guidata(hObject,handles);
else
    set(handles.edit6,'Visible','on');
    set(handles.text18,'Visible','on');
    guidata(hObject,handles);
end
end
function popupmenu5_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu5
fea=get(handles.popupmenu5,'Value');
if fea==1
    set(handles.edit7,'Visible','off');
    set(handles.text20,'Visible','off');
    set(handles.popupmenu6,'Visible','off');
    set(handles.text26,'Visible','off');
    guidata(hObject,handles);
else
    set(handles.edit7,'Visible','on');
    set(handles.text20,'Visible','on');
    set(handles.popupmenu6,'Visible','on');
    set(handles.text26,'Visible','on');
    guidata(hObject,handles);
end
end
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Kag_predataGUI(hObject,handles);
end
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2
fea=get(handles.popupmenu2,'Value');
if fea==1
    p.feature_select=0;
    set(handles.edit3,'Visible','off');
    set(handles.text9,'Visible','off');
elseif fea==2
    set(handles.text9,'Visible','on');
    set(handles.edit3,'Visible','on');
    guidata(hObject,handles);
    %p.feature_select=str2double(get(handles.edit3,'string'));
end

end
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load data.mat;
guidata(hObject,handles);
outcsv(handles.p,out);
end
function popupmenu6_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu6 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu6
fea=get(handles.popupmenu6,'Value');
if fea==1||fea==2
    set(handles.edit8,'Visible','off');
    set(handles.text25,'Visible','off');
    guidata(hObject,handles);
else
    set(handles.edit8,'Visible','on');
    set(handles.text25,'Visible','on');
    guidata(hObject,handles);
end
end

function Kag_predataGUI(hObject,handles)

p=pset(hObject,handles);handles.p=p;

eval(['load(''' p.predata '/train.mat'')']);
eval(['load(''' p.predata '/test.mat'')']);

set(handles.pushbutton2,'enable','on');
set(handles.edit4,'enable','on');
set(handles.uitable2,'enable','off');
set(handles.uitable3,'enable','off');
guidata(hObject,handles);
eval([  'set(handles.pushbutton2,''String'',''' num2str(0) '/' num2str(p.crossvaliN) ''');'  ]);
guidata(hObject,handles);
pause(0.1);

[proference_table,feature_table,out]=feature_selection_run(p,hObject,handles,train_data,test_data,train_dataID,test_dataID,train_out);

set(handles.uitable2,'enable','on');
set(handles.uitable3,'enable','on');
guidata(hObject,handles);
eval([  'set(handles.uitable2,''data'',proference_table);'  ]);
eval([  'set(handles.uitable3,''data'',feature_table);'  ]);
set(handles.pushbutton2,'string','create csv file');
guidata(hObject,handles);

% out sumitted csv file
save('data.mat','p','out','train_data','train_dataID','train_out','test_data','test_dataID','proference_table','feature_table');
end
function p=pset(hObject,handles)

p.predata='pre_data';
guidata(hObject,handles);
p.pre_data=get(handles.popupmenu1,'Value');
p.crossvaliN=str2double(get(handles.edit2,'string'));
p.fea1=get(handles.popupmenu2,'Value');

p.pre2_thred =0.8;
p.classifier=get(handles.popupmenu3,'Value');
p.out_name='trainedClassifier';

p.feature_filter=get(handles.popupmenu4,'Value');
if p.feature_filter==1
    p.filter_number=0;
else
    p.feature_number=str2num(get(handles.edit6,'string'));
end

p.feature_wrapper=get(handles.popupmenu5,'Value');
if p.feature_wrapper==1
    p.wrapper_number=0;
else
    p.wrapper_criteria=get(handles.popupmenu6,'Value');
    if p.wrapper_criteria==3||p.wrapper_criteria==4
        p.wrapper_base=str2num(get(handles.edit8,'string'));
    else
        p.wrapper_base=[];
    end
    p.wrapper_number=str2num(get(handles.edit7,'string'));
end

end

% Load data and pre-data
function [out,AUC_mean,acc]=load_train(p,handles,train_data,test_data,train_dataID,test_dataID,train_out,feature_select)

%eval(['load(''' p.predata '/train.mat'')']);
%eval(['load(''' p.predata '/test.mat'')']);

% Select features
if feature_select~=0
    train_data=train_data(:,feature_select);
    test_data=test_data(:,feature_select);
    train_dataID=train_dataID(:,feature_select);
    test_dataID=test_dataID(:,feature_select);
else
end

% cut datasets into 2 part(validation)
AUC=zeros(1,p.crossvaliN);out_matrix=zeros(size(test_data,1),p.crossvaliN);acc_matrix=zeros(1,p.crossvaliN);
for vai=1:p.crossvaliN
    [AUC(1,vai),out_matrix(:,vai),classifier{vai},acc_matrix(1,vai)]=vali_data(p,vai,train_data,test_data,train_out);
end
out=(mean(out_matrix'))';
AUC_mean=mean(AUC);AUC_show=AUC';acc=mean(acc_matrix);

end
function [out1,out2,out3]=pre_data(data1,data2,data3,p)

switch p.pre_data
    case 2
        [out1,out2,out3] = maxmin(data1,data2,data3,p);
    case 3
        [out1,out2,out3] = thred(data1,data2,data3,p);
    case 1
        out1=data1;out2=data2;out3=data3;
end
    function [out1,out2,out3]=maxmin(data1,data2,data3,p)
       out1=zeros(size(data1));out2=zeros(size(data2));out3=zeros(size(data3));
       data1(data1<-100000)=0;data2(data2<-100000)=0;data3(data3<-100000)=0;
       maxmin=[max(data1);min(data1)];       
       for i=1:size(data1,2)
           if isempty(find(data1(:,i)))==1
           else
               out1(:,i)=(data1(:,i)-maxmin(2,i))/(maxmin(1,i)-maxmin(2,i));
           end
          if isempty(find(data2(:,i)))==1
           else
               out2(:,i)=(data2(:,i)-maxmin(2,i))/(maxmin(1,i)-maxmin(2,i));
          end
          if isempty(find(data3(:,i)))==1
           else
               out3(:,i)=(data3(:,i)-maxmin(2,i))/(maxmin(1,i)-maxmin(2,i));
           end
       end
    end
    function [out1,out2,out3] = thred(data1,data2,data3,p)
        out1=zeros(size(data1));out2=zeros(size(data2));out3=zeros(size(data3));
        for i=1:size(data1,2)
            if isempty(find(data1(:,i)))==1
            else                
               for j=1:size(data1(:,i),1)
                  if data1(j,i)> max(data1(:,i))*p.pre2_thred
                      out1(j,i)=1;                         
                  end
               end
            end 
            if isempty(find(data2(:,i)))==1
            else                
               for j=1:size(data2(:,i),1)
                  if data2(j,i)> max(data2(:,i))*p.pre2_thred
                      out2(j,i)=1;                         
                  end
               end
            end
            if isempty(find(data3(:,i)))==1
            else                
               for j=1:size(data3(:,i),1)
                  if data3(j,i)> max(data3(:,i))*p.pre2_thred
                      out3(j,i)=1;                         
                  end
               end
            end
        end
    end
end
function [AUC,out,classifier,acc]=vali_data(p,vai,train_data,test_data,train_out)

    % training and valida dataset
    valiN=fix(size(train_data,1)/p.crossvaliN);div_train1=[];
    if vai==p.crossvaliN
        div_train2=(vai-1)*valiN+1:size(train_data,1);
    else
        div_train2=(vai-1)*valiN+1:vai*valiN;
    end
    for t=1:size(train_data,1)
        if isempty(find(div_train2==t))==1
            div_train1=[div_train1 t];
        end
    end
    pre_train1_data=train_data(div_train1,:);pre_train2_data=train_data(div_train2,:);
    train1_out=train_out(div_train1,:);train2_out=train_out(div_train2,:);

    % pre-process data methods
    [train1_data,train2_data,test_data]=pre_data(pre_train1_data,pre_train2_data,test_data,p);
    [AUC,out,classifier,acc]=Kag_compareGUI(p,train1_data,train1_out,train2_data,train2_out,test_data);
end
function [AUC,out,classifier,acc]=Kag_compareGUI(p,train1_data,train1_out,train2_data,train2_out,test_data)

% Different classifier

switch p.classifier
    case 1
        classifier = fitcnb(train1_data,train1_out); % multiclass naive Bayes model
        mod_cp=1;
    case 2
        classifier = fitctree(train1_data,train1_out); % decision tree 
        mod_cp=1;
    case 3
        classifier = fitcdiscr(train1_data,train1_out); % discriminant analysis classifier
        mod_cp=1;
    case 4
        classifier = fitcknn(train1_data,train1_out); % k-nearest neighbor classifier
        mod_cp=1;
    case 5
        classifier = fitcsvm(train1_data,train1_out); % binary support vector machine classifier
        mod_cp=1;
    case 6
        classifier = fitensemble(train1_data,train1_out,'RUSBoost',100,'Tree'); % Ensemble
        mod_cp=1;
    case 7
        setdemorandstream(391418381)                  % Business Preferred Network (BPN)
        net = patternnet(50);net.trainParam.showWindow = false;
        [net,~] = train(net,train1_data',train1_out');
        mod_cp=2;classifier=net;
    case 8                                             % Radial basis network(RBN)
        eg=0.02;sc = 100;   %eg: sum-squared error goal, sc: spread constant
        net = newrb(train1_data',train1_out',eg,sc);
        mod_cp=2;classifier=net;
    case 9                                             % Adaptive neuro-fuzzy inference system(ANFIS)
        numMFs = 5;mfType = 'gbellmf';
        in_fis = genfis1([train1_data,train1_out],numMFs,mfType);
        epoch_n = 20;dispOpt = zeros(1,4);
        out_fis = anfis([train1_data,train1_out],in_fis,epoch_n,dispOpt);
        train2_model=evalfis(train2_data,out_fis);
        out=evalfis(test_data,out_fis);
        mod_cp=3;classifier=out_fis;
        [~,~,~,AUC] = perfcurve(train2_out,train2_model,'1');
        acc=1-sum(abs(train2_out-train2_model))/size(train2_out,1);
end
    if mod_cp==1
        train2_model = predict(classifier,train2_data);
        out = predict(classifier,test_data);
        [~,~,~,AUC] = perfcurve(train2_out,train2_model,'1');
        acc=1-sum(abs(train2_out-train2_model))/size(train2_out,1);
    elseif mod_cp==2
        train2_model = (net(train2_data'))';out = (net(test_data'))';
        [~,~,~,AUC] = perfcurve(train2_out,train2_model,'1');
        acc=1-sum(abs(train2_out-train2_model))/size(train2_out,1);
    end
end
function outcsv(p,out)
p.csv_name=[p.out_name '_result.csv'];eval(['load(''' p.predata '/ID.mat'')']);

p.x={'ID','TARGET'};
fid=fopen(p.csv_name,'wt');
fprintf(fid, '%s,', p.x{1,1:end-1}) ;
fprintf(fid, '%s\n', p.x{1,end}) ;
fclose(fid) ;
dlmwrite(p.csv_name,[ID out],'delimiter',',','-append','precision', 6);
end
function [proference_table,feature_table,out]=feature_selection_run(p,hObject,handles,train_data,test_data,train_dataID,test_dataID,train_out)
tic;
%step1: feature pool
if p.fea1==1
    p.feature_select=1:size(train_data,2);
elseif p.fea1==2
    p.feature_select=str2num(get(handles.edit3,'string'));
end

%step2: Feature selection - Filter
if p.feature_filter==1
    %no filter
elseif p.feature_filter==2
    %p.feature_select=f(p.feature_select,p.filter_number)    PCA filter method
elseif p.feature_filter==2
    %p.feature_select=f(p.feature_select,p.filter_number)    FA filter method
end

%step3: Feature selection - Wrapper
% p.feature_wrapper: methods of feature selection wrapper methods
% p.wrapper_maxnumber: 0 ir 1: feature combination lists of N or 1,2---N
% p.wrapper_number: 1,2....N   number of features in combinations.

if p.feature_wrapper==1
    feature_list{1}=p.feature_select;
elseif p.feature_wrapper==2
    feature_list=makelist(p.feature_select,p.wrapper_number,p.wrapper_criteria,p.wrapper_base);
elseif p.feature_wrapper==3
    %feature_list=f(p.feature_select,p.wrapper_number)    PCA filter method
elseif p.feature_wrapper==4
    %feature_list=f(p.feature_select,p.wrapper_number)    PCA filter method
elseif p.feature_wrapper==5
    %feature_list=f(p.feature_select,p.wrapper_number)    PCA filter method
elseif p.feature_wrapper==6
    %feature_list=f(p.feature_select,p.wrapper_number)    PCA filter method
end

% Estimate running circle
n_run_max=0;n_run=0;out=[];agut=0;acct=0;
for f1=1:size(feature_list,2)
    n_run_max=n_run_max+size(feature_list{f1},1);
end
    set(handles.edit4,'String',num2str(0));
    eval([  'set(handles.pushbutton2,''String'',''' num2str(0) '/' num2str(n_run_max) ''');'  ]);
    guidata(hObject,handles);
    pause(0.1);

% Run classification according to p.feature_list
for f1=1:size(feature_list,2)
    tablen=size(feature_list{f1},1);
    resulttable_temp{f1}=zeros(tablen,2);
    for f2=1:tablen
        [out_new,resulttable_temp{f1}(f2,2),resulttable_temp{f1}(f2,1)]=load_train(p,handles,train_data,test_data,train_dataID,test_dataID,train_out,feature_list{f1}(f2,:));
        if resulttable_temp{f1}(f2,2)>=agut
            if resulttable_temp{f1}(f2,1)>=acct
                out=out_new;
            end
        end
        n_run=n_run+1;
        eval([  'set(handles.pushbutton2,''String'',''' num2str(n_run) '/' num2str(n_run_max) ''');'  ]);
        tt=toc;
        set(handles.edit4,'String',num2str(tt));
        guidata(hObject,handles);
        pause(0.1);
    end
end

% proference_table and feature_table
max_feature=size(feature_list{1,size(feature_list,2)},2);
feature_table=[];proference_table=[];
for f1=1:size(feature_list,2)
    if f1~=size(feature_list,2)
        feature_list{f1}=[feature_list{f1} zeros(size(feature_list{f1},1),max_feature-size(feature_list{f1},2))];
    end
    feature_table=[feature_table;feature_list{f1}];
    proference_table=[proference_table;resulttable_temp{f1}];
end
A=[proference_table feature_table];
B=sortrows(A,-1);C=sortrows(B,-2);
proference_table=C(:,1:2);feature_table=C(:,3:end);
end
function list=makelist(feature_select,wrapper_number,wrapper_criteria,wrapper_base)
if wrapper_criteria==1
    wrapper_number=min(max(round(wrapper_number),1),size(feature_select,2));
    list{1} = nchoosek(feature_select,wrapper_number);
elseif wrapper_criteria==2
    wrapper_number=min(max(round(wrapper_number),1),size(feature_select,2));
    for wi=1:wrapper_number
        list{wi} = nchoosek(feature_select,wi);
    end
elseif wrapper_criteria==3
    A=feature_select(ismember(feature_select,wrapper_base)==0);
    wrapper_number=min(max(round(wrapper_number),1),size(A,2));
    A_matrix = nchoosek(A,wrapper_number);
    list{1} = [repmat(wrapper_base,size(A_matrix,1),1) A_matrix];
elseif wrapper_criteria==4
    A=feature_select(ismember(feature_select,wrapper_base)==0);
    wrapper_number=min(max(round(wrapper_number),1),size(A,2));
    for wi=1:wrapper_number
        A_matrix = nchoosek(A,wi);
        list{wi} = [repmat(wrapper_base,size(A_matrix,1),1) A_matrix];
    end
end

function [M, I] = permn(V, N, K)

narginchk(2,3) ;

if fix(N) ~= N || N < 0 || numel(N) ~= 1 ;
    error('permn:negativeN','Second argument should be a positive integer') ;
end
nV = numel(V) ;

if nargin==2, % PERMN(V,N) - return all permutations
    
    if nV==0 || N == 0,
        M = zeros(nV,N) ;
        I = zeros(nV,N) ;
        
    elseif N == 1,
        % return column vectors
        M = V(:) ;
        I = (1:nV).' ;
    else
        % this is faster than the math trick used for the call with three
        % arguments.
        [Y{N:-1:1}] = ndgrid(1:nV) ;
        I = reshape(cat(N+1,Y{:}),[],N) ;
        % I = local_allcomb(1:nV, N) ;
        M = V(I) ;
    end
else % PERMN(V,N,K) - return a subset of all permutations
    nK = numel(K) ;
    if nV == 0 || N == 0 || nK == 0
        M = zeros(numel(K), N) ;
        I = zeros(numel(K), N) ;
    elseif nK < 1 || any(K<1) || any(K ~= fix(K))
        error('permn:InvalidIndex','Third argument should contain positive integers.') ;
    else
        
        V = reshape(V,1,[]) ; % v1.1 make input a row vector
        nV = numel(V) ;
        Npos = nV^N ;
        if any(K > Npos)
            warning('permn:IndexOverflow', ...
                'Values of K exceeding the total number of combinations are saturated.')
            K = min(K, Npos) ;
        end
             
        % The engine is based on version 3.2 of COMBN  with the correction
        % suggested by Roger Stafford. This approaches uses a single matrix
        % multiplication.
        B = nV.^(1-N:0) ;
        I = ((K(:)-.5) * B) ; % matrix multiplication
        I = rem(floor(I),nV) + 1 ;
        M = V(I) ;
    end
end

end
end

% --- Executes when entered data in editable cell(s) in uitable2.
function uitable2_CellEditCallback(hObject, eventdata, handles)
guidata(hObject,handles);
end
function uitable3_CellEditCallback(hObject, eventdata, handles)
guidata(hObject,handles);
end
function edit5_Callback(hObject, eventdata, handles)
end
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit6_Callback(hObject, eventdata, handles)
end
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function popupmenu5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double
end
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
end
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3
end
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double
end
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double
end
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
end
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double
guidata(hObject,handles);
end
function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double
guidata(hObject,handles);
end
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
