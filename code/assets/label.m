clear; close all; clc;

% 指定输入文件夹和输出文件夹路径
input_folder = '/media/hdu/T7/大批数据/extracted_data/input_file/'; % 替换为你的输入文件夹路径
output_folder = '/media/hdu/T7/大批数据/extracted_data/label/'; % 替换为你的输出文件夹路径


% 获取输入文件夹中的所有.mat文件
mat_files = dir(fullfile(input_folder, '*.mat'));

% 获取文件总数
total_files = length(mat_files);

% 循环遍历每个.mat文件
for k = 1:total_files
    % 读取当前.mat文件
    input_file = fullfile(input_folder, mat_files(k).name);
    [~, name, ~] = fileparts(input_file);   %提取文件名

    data_struct = load(input_file);

    % 获取数据的变量名（假设只有一个变量）
    var_names = fieldnames(data_struct);
    data = data_struct.(var_names{1});

    % 定义各波段的变量名
    R = cell(1, 35);
    for i = 1:35
        R{i} = data(:,:,i);
    end

    % 计算光谱指数
    DD = (R{32} - R{31}) - (R{27} - R{21});
    TVI = 0.5 * (120 * (R{32} - R{12}) - 200 * (R{21} - R{12}));
    LCI = (R{34} - R{30}) ./ (R{34} + R{25});
    mND680 = (R{33} - R{25}) ./ (R{33} + R{25} - 2 .* R{1});
    mND705 = (R{32} - R{29}) ./ (R{32} + R{29} - 2 .* R{1});
    PSSR = R{33} ./ R{6};
    CRI550 = 1./R{7} - 1./R{12};
    CRI700 = 1./R{7} - 1./R{27};
    MCARI = ((R{28} - R{22}) - 0.2 .* (R{28} - R{11})) ./ (R{28} ./ R{22});
    L = 0.5;
    SAVI = (1 + L) .* (R{35} - R{17}) ./ (R{35} + R{17} + L);
    CI_green = R{32} ./ R{12} - 1;
    CI_red_edge = R{32} ./ R{30} - 1;
    NDVI = (R{35} - R{17}) ./ (R{35} + R{17});
    DVI = R{35} - R{17};
    a = 1.22;
    b = 0.03;
    X = 0.08;
    ATSAVI = a .* (R{33} - a .* R{21} - b) ./ (a .* R{33} + R{21} - a .* b + X .* (1 + a.^2));
    EVI = 2.5 .* (R{35} - R{17}) ./ (R{35} + 6 .* R{17} - 7.5 .* R{3} + 1);
    GI = R{14} ./ R{24};
    MSAVI = 0.5 .* (2 .* R{33} + 1 - sqrt((2 .* R{33} + 1).^2 - 8 .* (R{33} - R{21})));
    MSR = (R{33} ./ R{21} - 1) ./ sqrt(R{33} ./ R{21} + 1);
    MVTI1 = 1.2 .* (1.2 .* (R{33} - R{12}) - 2.5 .* (R{21} - R{12}));
    MVTI2 = (1.5 .* (1.2 .* (R{33} - R{12}) - 2.5 .* (R{21} - R{12}))) ./ sqrt((2 .* R{33} + 1).^2 - (6 .* R{33} - 5 .* sqrt(R{21})) - 0.5);
    OSAVI = 1.16 .* (R{33} - R{21}) ./ (R{33} + R{21} + 0.16);
    PSND = (R{33} - R{3}) ./ (R{33} + R{3});
    RDVI = (R{33} - R{21}) ./ sqrt(R{33} + R{21});
    SPVI = 0.4 .* (3.7 .* (R{33} - R{21}) - 1.2 .* abs(R{8} - R{21}));
    TCARI = 3 .* ((R{27} - R{21}) - 0.2 .* (R{27} - R{12}) .* (R{27} ./ R{21}));
    SR = R{35} ./ R{17};
    VARI_green = (R{12} - R{17}) ./ (R{12} + R{17});
    WDRVI = (0.1 .* R{35} - R{17}) ./ (0.1 .* R{35} + R{17});
    ARI = 1 ./ R{12} - 1 ./ R{27};
    BGI = R{2} ./ R{12};
    BRI = R{2} ./ R{26};

    % 将光谱指数结果存储到结构体中
    spectral_indices = struct('DD', DD, 'TVI', TVI, 'LCI', LCI, 'mND680', mND680, ...
                              'mND705', mND705, 'PSSR', PSSR, 'CRI550', CRI550, ...
                              'CRI700', CRI700, 'MCARI', MCARI, 'SAVI', SAVI, ...
                              'CI_green', CI_green, 'CI_red_edge', CI_red_edge, ...
                              'NDVI', NDVI, 'DVI', DVI, 'ATSAVI', ATSAVI, ...
                              'EVI', EVI, 'GI', GI, 'MSAVI', MSAVI, 'MSR', MSR, ...
                              'MVTI1', MVTI1, 'MVTI2', MVTI2, 'OSAVI', OSAVI, ...
                              'PSND', PSND, 'RDVI', RDVI, 'SPVI', SPVI, ...
                              'TCARI', TCARI, 'SR', SR, 'VARI_green', VARI_green, ...
                              'WDRVI', WDRVI, 'ARI', ARI, 'BGI', BGI, 'BRI', BRI);

    % 构建输出文件名
    output_file = fullfile(output_folder, [             '_indices.mat']);

    % 保存计算结果到新的.mat文件
    save(output_file, 'spectral_indices');

    % 显示进度
    fprintf('正在处理文件 %d / %d: %s\n', k, total_files, mat_files(k).name);
end

disp('所有文件已处理完成。');
