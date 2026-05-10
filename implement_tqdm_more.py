import json

notebook_path = '/Users/macbook/Library/CloudStorage/GoogleDrive-nur.ichsan@gmail.com/My Drive/UNS/Bimbingan TA/Alwan/EDA-Cyberbullying-Sentiment-Analysis-SVM-main/test.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = cell['source']
    source_str = "".join(source)
    
    # 1. Update any remaining .apply(preprocess_text) to .progress_apply(...)
    if ".apply(preprocess_text)" in source_str:
        source = [line.replace(".apply(preprocess_text)", ".progress_apply(preprocess_text)") for line in source]
        cell['source'] = source
        source_str = "".join(source) # Update source_str for subsequent checks

    # 2. Section 6: oversample_text loop
    if "def oversample_text" in source_str:
        new_source = []
        for line in source:
            if "for label in df_train['label'].unique():" in line:
                new_source.append(line.replace("df_train['label'].unique():", "tqdm(df_train['label'].unique(), desc='Oversampling', leave=False):"))
            else:
                new_source.append(line)
        cell['source'] = new_source

    # 3. Section 8: Ensemble individual loops
    if "def run_ensemble_experiment" in source_str:
        new_source = []
        for line in source:
            if "for name, pipe in [('MNB', pipe_mnb)," in line:
                new_source.append(line.replace("[('MNB', pipe_mnb),", "tqdm([('MNB', pipe_mnb),")
                # Need to find the end of this list to close the parenthesis
            elif "('CNB', pipe_cnb)]:" in line:
                new_source.append(line.replace(")]:", "], desc='Base Models', leave=False):"))
            else:
                new_source.append(line)
        cell['source'] = new_source

    # 4. Section 11: Learning curve loop
    if "learning_curve(" in source_str:
        new_source = []
        for line in source:
            if "for (clf_name, clf), color in zip(classifiers_lc.items(), colors_lc):" in line:
                new_source.append(line.replace("zip(classifiers_lc.items(), colors_lc):", "tqdm(list(zip(classifiers_lc.items(), colors_lc)), desc='Learning Curves', leave=False):"))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Implemented additional tqdm progress bars in remaining candidates.")
