
load("data_spin_w_special.mat");
roll_spin_w_pid_special (1,:) = episode_state(1,:,4);
pitch_spin_w_pid_special (1,:) = episode_state(1,:,5);
airspeed_spin_w_pid_special (1,:) = episode_state(1,:,6);
altitude_spin_w_pid_special (1,:) = episode_state(1,:,7);

load("data_stall_w_special.mat");
roll_stall_w_pid_special (1,:) = episode_state(1,:,4);
pitch_stall_w_pid_special (1,:) = episode_state(1,:,5);
airspeed_stall_w_pid_special (1,:) = episode_state(1,:,6);
altitude_stall_w_pid_special (1,:) = episode_state(1,:,7);

load("test_stall_w_special.mat");
roll_stall_w_test_special (1,:) = episode_state(1,:,4);
pitch_stall_w_test_special (1,:) = episode_state(1,:,5);
airspeed_stall_w_test_special (1,:) = episode_state(1,:,6);
altitude_stall_w_test_special (1,:) = episode_state(1,:,7);

load("test_spin_w_special.mat");
roll_spin_w_test_special (1,:) = episode_state(1,:,4);
pitch_spin_w_test_special (1,:) = episode_state(1,:,5);
airspeed_spin_w_test_special (1,:) = episode_state(1,:,6);
altitude_spin_w_test_special (1,:) = episode_state(1,:,7);
% 
% 
% %%%%%%%%%%%%%%%%%%% 计算高度损失 %%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%% spin 50epis %%%%%%%%%%%%%%%%%%%
% load('pid_spin_nw_50.mat')
% for i = 1:50
%     roll_spin_nw_pid(i,:) = episode_state(i,:,4);
%     pitch_spin_nw_pid(i,:) = episode_state(i,:,5);
%     yaw_spin_nw_pid(i,:)= episode_state(i,:,6);
%     altitude_spin_nw_pid(i,:)= episode_state(i,:,7);
%     airspeed_spin_nw_pid(i,:) = episode_state(i,:,8);
% end
% 
% 
% load('pid_spin_w_50.mat')
% for i = 1:50
%     roll_spin_w_pid(i,:) = episode_state(i,:,4);
%     pitch_spin_w_pid(i,:) = episode_state(i,:,5);
%     yaw_spin_w_pid(i,:)= episode_state(i,:,6);
%     altitude_spin_w_pid(i,:)= episode_state(i,:,7);
%     airspeed_spin_w_pid(i,:) = episode_state(i,:,8);
% end
% load('TD3_test_spin_nw_50.mat')
% for i = 1:50
%     roll_spin_nw_test(i,:) = episode_state(i,:,4);
%     pitch_spin_nw_test(i,:) = episode_state(i,:,5);
%     yaw_spin_nw_test(i,:)= episode_state(i,:,6);
%     altitude_spin_nw_test(i,:)= episode_state(i,:,7);
%     airspeed_spin_nw_test(i,:) = episode_state(i,:,8);
% end
% 
% 
% load('TD3_test_spin_w_50.mat')
% for i = 1:50
%     roll_spin_w_test(i,:) = episode_state(i,:,4);
%     pitch_spin_w_test(i,:) = episode_state(i,:,5);
%     yaw_spin_w_test(i,:)= episode_state(i,:,6);
%     altitude_spin_w_test(i,:)= episode_state(i,:,7);
%     airspeed_spin_w_test(i,:) = episode_state(i,:,8);
% end
% 
% 
% 
% altitude_spin_nw_pid_lost_ave = mean(2500.-altitude_spin_nw_pid(:,500));
% 
% altitude_spin_w_pid_lost_ave = mean(2500.-altitude_spin_w_pid(:,500));
% 
% altitude_spin_nw_test_lost_ave = mean(2500.-altitude_spin_nw_test(:,500));
% 
% altitude_spin_w_test_lost_ave = mean(2500.-altitude_spin_w_test(:,500));
% 




%%%%%%%%%%%%%%%%%%%%%%%%%%% STALL 50epis%%%%%%%%%%%%%%%%%%



% 
% load('pid_stall_nw_50.mat')
% for i = 1:50
%     roll_stall_nw_pid(i,:) = episode_state(i,:,4);
%     pitch_stall_nw_pid(i,:) = episode_state(i,:,5);
%     yaw_stall_nw_pid(i,:)= episode_state(i,:,6);
%     altitude_stall_nw_pid(i,:)= episode_state(i,:,7);
%     airspeed_stall_nw_pid(i,:) = episode_state(i,:,8);
% end

% 
% load('pid_stall_w_50.mat')
% for i = 1:50
%     roll_stall_w_pid(i,:) = episode_state(i,:,4);
%     pitch_stall_w_pid(i,:) = episode_state(i,:,5);
%     yaw_stall_w_pid(i,:)= episode_state(i,:,6);
%     altitude_stall_w_pid(i,:)= episode_state(i,:,7);
%     airspeed_stall_w_pid(i,:) = episode_state(i,:,8);
% end
% 
% 
% load('TD3_test_stall_nw_50.mat')
% for i = 1:50
%     roll_stall_nw_test(i,:) = episode_state(i,:,4);
%     pitch_stall_nw_test(i,:) = episode_state(i,:,5);
%     yaw_stall_nw_test(i,:)= episode_state(i,:,6);
%     altitude_stall_nw_test(i,:)= episode_state(i,:,7);
%     airspeed_stall_nw_test(i,:) = episode_state(i,:,8);
% end
% 
% 
% load('TD3_test_stall_w_50.mat')
% for i = 1:50
%     roll_stall_w_test(i,:) = episode_state(i,:,4);
%     pitch_stall_w_test(i,:) = episode_state(i,:,5);
%     yaw_stall_w_test(i,:)= episode_state(i,:,6);
%     altitude_stall_w_test(i,:)= episode_state(i,:,7);
%     airspeed_stall_w_test(i,:) = episode_state(i,:,8);
% end










% roll = episode_state{1,1}(:,4);
% pitch = episode_state{1,1}(:,5);
% yaw = episode_state{1,1}(:,6);
% altitude = episode_state{1,1}(:,7);
% airspeed = episode_state{1,1}(:,8);
wucha_po =zeros(1,500);
wucha_ne = zeros(1,500);
wucha_po (1,:)=15;
wucha_ne(1,:)= -15;
load('data_spin_nw.mat')
for i = 1:10
    roll_spin_nw_pid(i,:) = episode_state(i,:,4);
    pitch_spin_nw_pid(i,:) = episode_state(i,:,5);
    yaw_spin_nw_pid(i,:)= episode_state(i,:,6);
    altitude_spin_nw_pid(i,:)= episode_state(i,:,7);
    airspeed_spin_nw_pid(i,:) = episode_state(i,:,8);
end


load('data_spin_w.mat')
for i = 1:10
    roll_spin_w_pid(i,:) = episode_state(i,:,4);
    pitch_spin_w_pid(i,:) = episode_state(i,:,5);
    yaw_spin_w_pid(i,:)= episode_state(i,:,6);
    altitude_spin_w_pid(i,:)= episode_state(i,:,7);
    airspeed_spin_w_pid(i,:) = episode_state(i,:,8);
end


load('data_stall_nw.mat')
for i = 1:10
    roll_stall_nw_pid(i,:) = episode_state(i,:,4);
    pitch_stall_nw_pid(i,:) = episode_state(i,:,5);
    yaw_stall_nw_pid(i,:)= episode_state(i,:,6);
    altitude_stall_nw_pid(i,:)= episode_state(i,:,7);
    airspeed_stall_nw_pid(i,:) = episode_state(i,:,8);
end


load('data_stall_w.mat')
for i = 1:10
    roll_stall_w_pid(i,:) = episode_state(i,:,4);
    pitch_stall_w_pid(i,:) = episode_state(i,:,5);
    yaw_stall_w_pid(i,:)= episode_state(i,:,6);
    altitude_stall_w_pid(i,:)= episode_state(i,:,7);
    airspeed_stall_w_pid(i,:) = episode_state(i,:,8);
end

load('train_test_stall_w_1')
for i=1:10
%     COUNT = num2str(i);
%     nam1 = 'train_test_stall_w_';
%     nam2 = num2str(i);
%     nam3 = '.mat';
%     filename = [nam1, nam2, nam3];
%     load(filename);
%     
%     roll_stall_w_test(i,:) = episode_state(1,:,4);
%     pitch_stall_w_test(i,:) = episode_state(1,:,5);
%     yaw_stall_w_test(i,:)= episode_state(1,:,6);
%     altitude_stall_w_test(i,:)= episode_state(1,:,7);
%     airspeed_stall_w_test(i,:) = episode_state(1,:,8);
    roll_stall_w_test(i,:) = episode_state(i,:,4);
    pitch_stall_w_test(i,:) = episode_state(i,:,5);
    yaw_stall_w_test(i,:)= episode_state(i,:,6);
    altitude_stall_w_test(i,:)= episode_state(i,:,7);
    airspeed_stall_w_test(i,:) = episode_state(i,:,8);
    
end
% load('train_test_stall_nw_1');
for i=1:10
    COUNT = num2str(i);
    nam1 = 'data_test_stall_nw_';
    nam2 = num2str(i);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    
    roll_stall_nw_test(i,:) = episode_state(1,:,4);
    pitch_stall_nw_test(i,:) = episode_state(1,:,5);
    yaw_stall_nw_test(i,:)= episode_state(1,:,6);
    altitude_stall_nw_test(i,:)= episode_state(1,:,7);
    airspeed_stall_nw_test(i,:) = episode_state(1,:,8);

%     roll_stall_nw_test(i,:) = episode_state(i,:,4);
%     pitch_stall_nw_test(i,:) = episode_state(i,:,5);
%     yaw_stall_nw_test(i,:)= episode_state(i,:,6);
%     altitude_stall_nw_test(i,:)= episode_state(i,:,7);
%     airspeed_stall_nw_test(i,:) = episode_state(i,:,8);
    
end

load('train_test_spin_w_1');
for i=1:10
%     COUNT = num2str(i);
%     nam1 = 'train_test_spin_w_';
%     nam2 = num2str(i);
%     nam3 = '.mat';
%     filename = [nam1, nam2, nam3];
%     load(filename);
    
%     roll_spin_w_test(i,:) = episode_state(1,:,4);
%     pitch_spin_w_test(i,:) = episode_state(1,:,5);
%     yaw_spin_w_test(i,:)= episode_state(1,:,6);
%     altitude_spin_w_test(i,:)= episode_state(1,:,7);
%     airspeed_spin_w_test(i,:) = episode_state(1,:,8);
    roll_spin_w_test(i,:) = episode_state(i,:,4);
    pitch_spin_w_test(i,:) = episode_state(i,:,5);
    yaw_spin_w_test(i,:)= episode_state(i,:,6);
    altitude_spin_w_test(i,:)= episode_state(i,:,7);
    airspeed_spin_w_test(i,:) = episode_state(i,:,8);
    
end

% load('train_test_spin_nw_1');
for i=1:10
    COUNT = num2str(i);
    nam1 = 'data_test_spin_w_';
    nam2 = num2str(i);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    
    roll_spin_nw_test(i,:) = episode_state(1,:,4);
    pitch_spin_nw_test(i,:) = episode_state(1,:,5);
    yaw_spin_nw_test(i,:)= episode_state(1,:,6);
    altitude_spin_nw_test(i,:)= episode_state(1,:,7);
    airspeed_spin_nw_test(i,:) = episode_state(1,:,8);
%     roll_spin_nw_test(i,:) = episode_state(i,:,4);
%     pitch_spin_nw_test(i,:) = episode_state(i,:,5);
%     yaw_spin_nw_test(i,:)= episode_state(i,:,6);
%     altitude_spin_nw_test(i,:)= episode_state(i,:,7);
%     airspeed_spin_nw_test(i,:) = episode_state(i,:,8);
    
end


load('TD3_w.mat');
for j = 1:10
   pitch_spin_w_test(j,:) = episode_state(j,1:500,5);
    altitude_spin_w_test(j,:) = episode_state(j,1:500,7);
end
% 
load('TD3_nw.mat');
for j = 1:10
     pitch_spin_nw_test(j,:) = episode_state(j,1:500,5);
   altitude_spin_nw_test(j,:) = episode_state(j,1:500,7);
end

load('pid_w.mat');
for j = 1:10
    pitch_spin_w_pid(j,:) = episode_state(j,1:500,5);
   altitude_spin_w_pid(j,:) = episode_state(j,1:500,7);
end
% 
load('pid_nw.mat');
for j = 1:10
    pitch_spin_nw_pid(j,:) = episode_state(j,1:500,5);
   altitude_spin_nw_pid(j,:) = episode_state(j,1:500,7);
end


step = 1:500;
% roll_conf = [roll+3 roll(end:-1:1)-3];
% pitch_conf = [pitch+3 pitch(end:-1:1)-3]
% step_conf = [step step(end:-1:1)];
figure(1)

for j=1:10
%     plot(step,altitude_spin_nw_test(j,:));hold on;
%     plot(step,altitude_spin_w_test(j,:));hold on;
%     plot(step,altitude_spin_nw_pid(j,:));hold on;
%     plot(step,altitude_spin_w_pid(j,:));hold on;   
    
end

%%%%%%%%%%%%%%%%%%%%%%% SPIN %%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%  roll spin chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,roll_spin_nw_pid(8,:));
% plot(step,roll_spin_w_pid(8,:));%第八组 有风尾旋pid
% plot(step,roll_spin_w_test(5,:));%第5组 有风尾旋test
% plot(step,roll_spin_nw_test(10,:));%第十组 无风尾旋test
grid on;
legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');


%%%%%%%%%%%%%%%%%%%%%  pitch spin chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,pitch_spin_nw_pid(8,:));
%plot(step,pitch_spin_w_pid(8,:));%第八组 有风尾旋pid
% plot(step,pitch_spin_w_test(7,:));%第八组 有风尾旋test
% plot(step,pitch_spin_nw_test(2,:));%第2组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');




%%%%%%%%%%%%%%%%%%%%%  airspeed spin chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,airspeed_spin_nw_pid(8,:));hold on
%  plot(step,airspeed_spin_w_pid(4,:));hold on%第4组 有风尾旋pid
% plot(step,airspeed_spin_w_test(2,:));%第2组 有风尾旋test
% plot(step,airspeed_spin_nw_test(10,:));%第10组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');


%%%%%%%%%%%%%%%%%%%%%  altitude spin chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,altitude_spin_nw_pid(8,:));
% plot(step,altitude_spin_w_pid(7,:));%第7组 有风尾旋pid
% plot(step,altitude_spin_w_test(6,:));%第6组 有风尾旋test
% plot(step,altitude_spin_nw_test(7,:));%第3组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');
% 





%%%%%%%%%%%%%%%%%%%%%  STALL %%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%  roll stall chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,roll_stall_nw_pid(9,:));hold on%第9组 无风stall pid
% plot(step,roll_stall_w_pid(7,:)); hold on%第7组 有风stallpid
% plot(step,roll_stall_w_test(1,:)); hold on %第 1 组 有风尾旋test
% plot(step,roll_stall_nw_test(1,:));%第3组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');


%%%%%%%%%%%%%%%%%%%%%  pitch stall chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,pitch_stall_nw_pid(10,:));hold on%第10组
% plot(step,pitch_stall_w_pid(7,:));hold on%第7组 有风尾旋pid
% plot(step,pitch_stall_w_test(9,:));%第9组 有风尾旋test
% plot(step,pitch_stall_nw_test(1,:));%第1组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');




%%%%%%%%%%%%%%%%%%%%%  airspeed stall chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,airspeed_stall_nw_pid(10,:));hold on%第10组
%  plot(step,airspeed_stall_w_pid(3,:));hold on%第3组 有风尾旋pid
% plot(step,airspeed_stall_w_test(9,:));%第9组 有风尾旋test
% plot(step,airspeed_stall_nw_test(7,:));%第7组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');


%%%%%%%%%%%%%%%%%%%%%  altitude stall chose%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,altitude_stall_nw_pid(10,:));hold on%第10组
% plot(step,altitude_stall_w_pid(7,:));%第7组 有风尾旋pid
% plot(step,altitude_stall_w_test(2,:));%第2组 有风尾旋test
% plot(step,altitude_stall_nw_test(3,:));%第3组 无风尾旋test
% grid on;
% legend('data1','data2','data3','data4','data5','data6','data7','data8','data9','data10');






figure(2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  roll spin %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% plot(step,roll_spin_nw_pid(8,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,roll_spin_w_pid(3,:),'--','linewidth',5,'Color','#EDB120');hold on;%第5组 有风尾旋pid
% % plot(step,roll_spin_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第5组 有风尾旋pid
% plot(step,roll_spin_nw_test(10,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,roll_spin_w_test(6,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% grid on;


%%%%%%%%%%%%%%%%%%%%%%%pitch spin%%%%%%%%%%%%%%%%%%%%%%%%
% 
plot(step,pitch_spin_nw_pid(8,:),'--','linewidth',5,'Color','#0072BD');hold on;
plot(step,pitch_spin_w_pid(10,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% plot(step,pitch_spin_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid

plot(step,pitch_spin_nw_test(7,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
plot(step,pitch_spin_w_test(7,:),'linewidth',5,'Color','#A2142F');hold on;%第7组 有风尾旋test
grid on;



%%%%%%%%%%%%%%%%%%%%%%%airspeed spin%%%%%%%%%%%%%%%%%%%%%%%%
% 
% airspeed_spin_nw_pid(8,1)= airspeed_spin_nw_pid(8,2);
% airspeed_spin_w_pid(5,1) = airspeed_spin_w_pid(5,2);
% airspeed_spin_nw_test(10,1)= airspeed_spin_nw_test(10,2);
% airspeed_spin_w_test(6,1)= airspeed_spin_w_test(6,2);
% airspeed_spin_w_pid_special(1,1) = airspeed_spin_w_pid_special(1,2);
% plot(step,airspeed_spin_nw_pid(8,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,airspeed_spin_w_pid(5,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% % plot(step,airspeed_spin_w_pid_special(1,:),':','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% 
% plot(step,airspeed_spin_nw_test(10,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,airspeed_spin_w_test(6,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% grid on;



%%%%%%%%%%%%%%%%%%%%%%%altitude spin%%%%%%%%%%%%%%%%%%%%%%%%

plot(step,altitude_spin_nw_pid(8,:),'--','linewidth',5,'Color','#0072BD');hold on;
plot(step,altitude_spin_w_pid(10,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% plot(step,altitude_spin_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid

plot(step,altitude_spin_nw_test(7,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
plot(step,altitude_spin_w_test(7,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
grid on;
altitude_spin_nw_pid_lost_ave = mean(2500.- altitude_spin_nw_pid(8,:));
altitude_spin_w_pid_lost_ave = mean(2500.- altitude_spin_w_pid(7,:));
altitude_spin_nw_test_lost_ave = mean(2500.- altitude_spin_nw_test(3,:));
altitude_spin_w_test_lost_ave = mean(2500.- altitude_spin_w_test(6,:));

% 







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  roll stall %%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,roll_stall_nw_pid(9,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,roll_stall_w_pid(10,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% % plot(step,roll_stall_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% 
% plot(step,roll_stall_nw_test(1,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,roll_stall_w_test(10,:),'linewidth',5,'Color','#A2142F');hold on;%第4组 有风尾旋test
% grid on;


%%%%%%%%%%%%%%%%%%%%%%%pitch stall%%%%%%%%%%%%%%%%%%%%%%%%


% plot(step,pitch_stall_nw_pid(10,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,pitch_stall_w_pid(7,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% % plot(step,pitch_stall_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% plot(step,pitch_stall_nw_test(1,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,pitch_stall_w_test(9,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% % plot(step,pitch_stall_w_test_special(1,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% 
% grid on;



%%%%%%%%%%%%%%%%%%%%%%%airspeed stall%%%%%%%%%%%%%%%%%%%%%%%%

% airspeed_stall_nw_pid(8,1)= airspeed_stall_nw_pid(8,2);
% airspeed_stall_w_pid(1,1) = airspeed_stall_w_pid(1,2);
% airspeed_stall_nw_test(7,1)= airspeed_stall_nw_test(7,2);
% airspeed_stall_w_test(5,1)= airspeed_stall_w_test(5,2);
% plot(step,airspeed_stall_nw_pid(10,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,airspeed_stall_w_pid(10,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% % plot(step,airspeed_stall_w_pid_special(1,:),':','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% plot(step,airspeed_stall_nw_test(7,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,airspeed_stall_w_test(1,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% grid on;
% 


%%%%%%%%%%%%%%%%%%%%%%%altitude stall%%%%%%%%%%%%%%%%%%%%%%%%

% plot(step,altitude_stall_nw_pid(10,:),'--','linewidth',5,'Color','#0072BD');hold on;
% plot(step,altitude_stall_w_pid(7,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% % plot(step,altitude_stall_w_pid_special(1,:),'--','linewidth',5,'Color','#EDB120');hold on;%第八组 有风尾旋pid
% 
% plot(step,altitude_stall_nw_test(3,:),'linewidth',5,'Color','#77AC30');hold on;%第十组 无风尾旋test
% plot(step,altitude_stall_w_test(9,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% % plot(step,altitude_stall_w_test_special(1,:),'linewidth',5,'Color','#A2142F');hold on;%第八组 有风尾旋test
% 
% grid on;










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plot(step,wucha_po,'--','Color','k','linewidth',3);hold on;
plot(step,wucha_ne,'--','Color','k','linewidth',3);hold on;
set(gca,'FontName','Times New Roman','FontSize',45);
% 
% 
hl21 = xlabel('Step','FontName','Times New Roman','FontSize',50);
set(hl21);
% hl22 = ylabel('Roll(deg)','FontName','Times New Roman','FontSize',50,'Rotation',90);
hl22 = ylabel('Pitch(deg)','FontName','Times New Roman','FontSize',50,'Rotation',90);
% hl22 = ylabel('Airspeed(m/s)','FontName','Times New Roman','FontSize',50,'Rotation',90);
% hl22 = ylabel('Altitude(m)','FontName','Times New Roman','FontSize',50,'Rotation',90);

set(hl22);
% axis([0 500 -240 180])
% set(gca,'ytick',[-240 -210 -180 -150 -120 -90  -60 -30  0   30  60  90  120 150  180 210]);
axis([0 500 -80 40])
% axis([0 500 -100 140])
% axis([0 500 -20 100])
% axis([0 500 1900 2600])
% set(gca,'ytick',[ -100 -80 -60 -40  -20  0   20  40  60  80 100]);
set(gca,'ytick',[-100 -80 -60 -40  -20  0   20  40  60  80 100 120 140]);
% set(gca,'ytick',[1900 2000 2100 2200 2300 2400 2500 2600]);

% txt1 = '$$10$$';
% hl2=text(200,20,txt1,'FontSize',45);
% set(hl2,'interpreter','latex')
% txt2 = '$$-10$$';
% h22=text(200,-20,txt1,'FontSize',45);
% set(h22,'interpreter','latex')


% 
% hl20 = legend('PID in calm wind','PID in 15 kts wind','TD3 in calm wind','TD3 in 15 kts wind','Target region','FontName','Times New Roman','FontSize',45);
hl20 = legend('PID in calm wind','PID in 15 kts wind','TD3 in calm wind','TD3 in 15 kts wind','Target region','FontName','Times New Roman','FontSize',45);
% hl20 = legend('PID in calm wind','PID in 15 kts wind','TD3 in calm wind','TD3 in 15 kts wind','Airspeed envelope','FontName','Times New Roman','FontSize',45);
% hl20 = legend('PID in calm wind','PID in 15 kts wind','TD3 in calm wind','TD3 in 15 kts wind','Initial altitude','FontName','Times New Roman','FontSize',45);

set(hl20,'Box','on');
% set(hl20,'interpreter','latex');
% 
% 
% 
% figure(2)

% pr = fill(step_conf,roll_conf,'red')
% pr.FaceColor = [1,0.8,0.8];
% pr.EdgeColor = 'none';
% pp = fill(step_conf,pitch_conf,'red')
% pp.FaceColor = [1,0.8,0.8];
% pp.EdgeColor = 'none';
% hold on

% 
% figure(3)
% plot(step,altitude);grid on
% title('Altitude');xlabel('steps');ylabel('altitude(m)');
% legend('altitude')
% 
% figure(4)
% plot(step,airspeed);grid on
% title('Airspeed');xlabel('steps');ylabel('airspeed(m/s)');
% legend('airspeed')

