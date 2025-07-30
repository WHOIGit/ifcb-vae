% Set base directory for input and output
scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, 'data', 'oleander');
fileList = { ...
    fullfile('//sosiknas1/IFCB_products/Oleander/summary/count_group_class_withTS.mat'), ...
    fullfile('//sosiknas1/IFCB_products/Oleander/summary/carbon_group_class_withTS.mat') ...
};

% Ensure base directory exists
if ~exist(baseDir, 'dir')
    fprintf('Creating output directory: %s\n', baseDir);
    mkdir(baseDir);
end

% Flags to ensure class table and metadata are only saved once
savedMetaAndClassTable = false;

for f = 1:length(fileList)
    fprintf('\n[%d/%d] Loading file: %s\n', f, length(fileList), fileList{f});
    
    % Load .mat file
    load(fileList{f});
    
    % Clean and convert timestamps
    rawTimes = meta_data.sample_time;
    trimmedTimes = erase(rawTimes, '+00:00');
    sampleTimes = datetime(trimmedTimes, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'TimeZone', 'UTC');
    
    % Extract *_label variables into table
    fprintf('Extracting label tables...\n');
    labelTables = extractLabelTables();

    % Save class table and metadata only once
    if ~savedMetaAndClassTable
        fprintf('Saving class labels to: %s\n', fullfile(baseDir, 'ifcb_class.csv'));
        writetable(labelTables, fullfile(baseDir, 'ifcb_class.csv'));
        
        fprintf('Saving metadata to: %s\n', fullfile(baseDir, 'ifcb_metadata.csv'));
        writetable(meta_data, fullfile(baseDir, 'ifcb_metadata.csv'));
        
        savedMetaAndClassTable = true;
    end

    % Save count or carbon data tables if available
    saveIfExists('classcount_opt_adhoc_merge', fullfile(baseDir, 'ifcb_count_raw.csv'));
    saveIfExists('classC_opt_adhoc_merge', fullfile(baseDir, 'ifcb_carbon_raw.csv'));

    % Clear variables except loop control
    clearvars -except fileList f savedMetaAndClassTable baseDir
end

fprintf('\nAll files processed successfully.\n');

% --- Helper Functions ---

function tbl = extractLabelTables()
    vars = whos;
    labelVars = {vars(strcmp({vars.class}, 'cell') & endsWith({vars.name}, '_label')).name};

    tbl = table();
    for i = 1:numel(labelVars)
        name = labelVars{i};
        data = evalin('caller', name);
        label = erase(name, '_label');
        tmp = table(data, repmat({label}, size(data, 1), 1), ...
                    'VariableNames', {'class', 'label'});
        tbl = [tbl; tmp];
        fprintf('  Processed label group: %s (%d entries)\n', label, size(data, 1));
    end
end

function saveIfExists(varName, filePath)
    outDir = fileparts(filePath);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    if evalin('caller', ['exist(''', varName, ''', ''var'')']) && ...
       istable(evalin('caller', varName))
        fprintf('Saving variable "%s" to: %s\n', varName, filePath);
        writetable(evalin('caller', varName), filePath);
    else
        fprintf('Variable "%s" does not exist or is not a table. Skipping.\n', varName);
    end
end