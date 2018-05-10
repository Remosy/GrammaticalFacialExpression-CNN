%% Before Face
figure;
scatter(SignerA_X,SignerA_Y,50,'MarkerEdgeColor',[0.3333 0 0],'MarkerFaceColor',[1.0000 0.3333 0],'LineWidth',1.5)
hold on;
scatter(SignerB_X,SignerB_Y,50,'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'LineWidth',1.5)

%% Nose Only

figure;
scatter(SignerA_X,SignerA_Y,50,'MarkerEdgeColor',[0.3333 0 0],'MarkerFaceColor',[1.0000 0.3333 0],'LineWidth',1.5)
hold on;
scatter(SignerB2_X,SignerB2_Y,50,'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'LineWidth',1.5)

%% Nose + Iris
SB2_X = SignerB_X
SB2_Y = SignerB_Y

SB2_X(17:26) = SB2_X(17:26) - dist_left_iris_X
SB2_X(91:95) = SB2_X(91:95) - dist_left_iris_X
SB2_X(1:8) = SB2_X(1:8) - dist_left_iris_X
SB2_X(88) = SB2_X(88) - dist_left_iris_X

SB2_Y(17:26) = SB2_Y(17:26) - dist_left_iris_Y
SB2_Y(91:95) = SB2_Y(91:95) - dist_left_iris_Y
SB2_Y(1:8) = SB2_Y(1:8) - dist_left_iris_Y
SB2_Y(88) = SB2_Y(88) - dist_left_iris_Y

SB2_X(27:36) = SB2_X(27:36) - dist_right_iris_X
SB2_X(96:100) = SB2_X(96:100) - dist_right_iris_X
SB2_X(9:16) = SB2_X(9:16) - dist_right_iris_X
SB2_X(89) = SB2_X(89) - dist_right_iris_X

SB2_Y(27:36) = SB2_Y(27:36) - dist_right_iris_Y
SB2_Y(96:100) = SB2_Y(96:100) - dist_right_iris_Y
SB2_Y(9:16) = SB2_Y(9:16) - dist_right_iris_Y
SB2_Y(89) = SB2_Y(89) - dist_right_iris_Y

SB2_X(37:48) = SB2_X(37:48) -  dis_X
SB2_Y(37:48) = SB2_Y(37:48) -  dis_Y

SB2_X(69:87) = SB2_X(69:87) - dis_X
SB2_Y(69:87) = SB2_Y(69:87) - dis_Y

SB2_X(49:68) = SB2_X(49:68) - dis_X
SB2_Y(49:68) = SB2_Y(49:68) - dis_Y

SB2_X(90) = SB2_X(90) - dis_X
SB2_Y(90) = SB2_Y(90) - dis_Y

figure;
scatter(SignerA_X,SignerA_Y,50,'MarkerEdgeColor',[0.3333 0 0],'MarkerFaceColor',[1.0000 0.3333 0],'MarkerFaceAlpha',.75,'LineWidth',1.5)
hold on;
scatter(SB2_X,SB2_Y,50,'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'MarkerFaceAlpha',.75,'LineWidth',1.5)
