## SLIST

- Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder data/rsc15/raw 
- Run a configuration with the following command:
For example: ```python run_preprocesing.py conf/preprocess/window/rsc15.yml```

### Basic Usage
- Change the expeimental settings and the model hyperparameters using a configuration file `*.yml`. </br>
- When a configuration file in conf/in has been executed, it will be moved to the folder conf/out.
- Run `run_config.py` with configuaration folder arguments to train and test models. </br>
For example: ```python run_confg.py conf/in conf/out```

### Running SLIST
- The yml files for slist used in paper can be found in `conf/save_slist`


## CDRC-MSc-Recommender-Systems

Recommender Systems for CDRC Masters Dissertation Scheme.

Abstract: https://www.cdrc.ac.uk/wp-content/uploads/sites/72/2021/10/Sharon-Liu-MDS2021-Abstract-1.pdf


This is based on the session-based recommendation framework session-rec (https://github.com/rn5l/session-rec).

Algorithms included:
* STAMP (https://dl.acm.org/doi/10.1145/3219819.3219950)
* STAN (https://dl.acm.org/doi/10.1145/3331184.3331322)
* RecVAE (https://arxiv.org/abs/1912.11160)
* GRU4Rec+ (https://arxiv.org/abs/1706.03847)
* SLIST (https://arxiv.org/abs/2103.16104)
* SLIST Ext (proposed extension of SLIST for new items)

### How to Use

* **Data Preprocessing**: Performs data preprocessing and splitting, and generates the item features matrix.
    
    Data Preprocessing and splitting: `python run_preprocessing.py conf/preprocess/base_dataset.yml`
    
    Generate item features matrix: `python preprocessing/generate_item_matrix_cease.py`
    
* **Hyperparameter Optimization**: Performs Bayesian optimization and outputs the results for each iteration in a CSV file.
    
    Base models: `python run_config.py conf/opt/opt_beauty_slist.yml`
    
    KNN extension in SLIST Ext: `python grid_search_knn.py`
    
* **Model Evaluation**: Performs model evaluation and outputs the results in a CSV file.
    
    Example: `python run_config.py conf/evaluation/evaluate_beauty_models.yml`

## Project structure
```
.
├── [451K]  algorithms
│   ├── [ 16K]  baselines
│   │   ├── [5.0K]  ar.py
│   │   ├── [   0]  __init__.py
│   │   └── [6.6K]  sr.py
│   ├── [5.9K]  baselines.py
│   ├── [9.6K]  filemodel
│   │   ├── [3.1K]  filemodel.py
│   │   ├── [   0]  __init__.py
│   │   └── [2.5K]  resultfile.py
│   ├── [ 92K]  gru4rec
│   │   ├── [ 292]  gpu_ops.py
│   │   ├── [ 41K]  gru4rec.py
│   │   ├── [   0]  __init__.py
│   │   └── [ 46K]  ugru4rec.py
│   ├── [   0]  __init__.py
│   ├── [ 38K]  knn
│   │   ├── [   0]  __init__.py
│   │   ├── [ 17K]  sknn.py
│   │   └── [ 17K]  stan.py
│   ├── [ 31K]  narm
│   │   ├── [   0]  __init__.py
│   │   └── [ 27K]  narm.py
│   ├── [3.1K]  pop.py
│   ├── [ 26K]  recvae
│   │   ├── [ 17K]  model.py
│   │   └── [5.5K]  utils.py
│   ├── [ 16K]  slist
│   │   └── [ 12K]  slist.py
│   ├── [ 14K]  slist_ext.py
│   ├── [ 13K]  slist.py
│   ├── [ 17K]  sr_gnn
│   │   ├── [3.3K]  main.py
│   │   ├── [6.1K]  model.py
│   │   └── [3.9K]  utils.py
│   ├── [112K]  STAMP
│   │   ├── [ 21K]  basic_layer
│   │   │   ├── [9.4K]  FwNn3AttLayer.py
│   │   │   ├── [   0]  __init__.py
│   │   │   ├── [1.7K]  LinearLayer_3dim.py
│   │   │   ├── [1007]  LinearLayer.py
│   │   │   ├── [2.4K]  NN_adam.py
│   │   │   └── [2.4K]  NN.py
│   │   ├── [ 16K]  data_prepare
│   │   │   ├── [3.3K]  dataset_read.py
│   │   │   ├── [6.5K]  entity
│   │   │   │   ├── [   0]  __init__.py
│   │   │   │   ├── [1.7K]  samplepack.py
│   │   │   │   └── [ 854]  sample.py
│   │   │   ├── [   0]  __init__.py
│   │   │   └── [2.2K]  load_dict.py
│   │   ├── [   0]  __init__.py
│   │   ├── [ 37K]  model
│   │   │   ├── [   0]  __init__.py
│   │   │   └── [ 33K]  STAMP.py
│   │   └── [ 34K]  util
│   │       ├── [2.2K]  AccCalculater.py
│   │       ├── [ 521]  Activer.py
│   │       ├── [4.1K]  BatchData.py
│   │       ├── [ 10K]  batcher
│   │       │   ├── [6.3K]  equal_len
│   │       │   │   ├── [2.3K]  batcher_p.py
│   │       │   │   └── [   0]  __init__.py
│   │       │   └── [   0]  __init__.py
│   │       ├── [ 539]  Bitmap.py
│   │       ├── [3.4K]  Config.py
│   │       ├── [ 497]  FileDumpLoad.py
│   │       ├── [ 845]  Formater.py
│   │       ├── [   0]  __init__.py
│   │       ├── [1.1K]  Pooler.py
│   │       ├── [2.4K]  Printer.py
│   │       ├── [ 252]  Randomer.py
│   │       ├── [1.8K]  SoftmaxMask.py
│   │       └── [2.2K]  TensorGather.py
│   ├── [ 17K]  stan.py
│   └── [ 36K]  test
│       ├── [4.5K]  ar_my.py
│       ├── [5.1K]  ar.py
│       ├── [ 15K]  slist.py
│       └── [7.1K]  sr.py
├── [ 52K]  analysis
│   ├── [1.2K]  eda.py
│   ├── [2.3K]  evaluate_new_items.py
│   ├── [2.3K]  evaluate_scalability.py
│   ├── [2.6K]  kmeans_clustering.py
│   ├── [ 12K]  plot_scalability.py
│   ├── [ 11K]  plots.py
│   └── [ 17K]  product_eda.py
├── [260K]  conf
│   ├── [9.3K]  evaluation
│   │   ├── [1.4K]  evaluate_beauty_models.yml
│   │   ├── [1.2K]  evaluate_beauty_new_items.yml
│   │   ├── [1.2K]  evaluate_scalability2.yml
│   │   └── [1.4K]  evaluate_scalability.yml
│   ├── [4.9K]  in
│   │   └── [ 942]  rsc15_100k_test.yml
│   ├── [ 15K]  opt
│   │   ├── [1.9K]  opt_beauty_gru4rec.yml
│   │   ├── [1.9K]  opt_beauty_slist.yml
│   │   ├── [1.9K]  opt_beauty_stamp.yml
│   │   ├── [1.9K]  opt_beauty_stan.yml
│   │   ├── [2.2K]  opt_models.yml
│   │   └── [1.1K]  opt_recvae.yml
│   ├── [ 15K]  preprocess
│   │   ├── [ 407]  base_dataset.yml
│   │   ├── [ 409]  new_items_dataset.yml
│   │   ├── [5.0K]  single
│   │   │   ├── [ 269]  diginetica.yml
│   │   │   ├── [ 268]  rsc15_4.yml
│   │   │   ├── [ 269]  rsc15_64.yml
│   │   │   └── [ 229]  rsc15.yml
│   │   └── [5.0K]  window
│   │       ├── [ 365]  diginetica.yml
│   │       ├── [ 363]  retailrocket.yml
│   │       └── [ 346]  rsc15.yml
│   ├── [200K]  save
│   │   ├── [ 74K]  diginetica
│   │   │   ├── [ 31K]  single split
│   │   │   │   ├── [ 16K]  opt
│   │   │   │   │   ├── [1.2K]  single_digi_gru.yml
│   │   │   │   │   ├── [1.2K]  single_digi_knn.yml
│   │   │   │   │   ├── [ 993]  single_digi_narm.yml
│   │   │   │   │   ├── [ 940]  single_digi_sr.yml
│   │   │   │   │   ├── [1.0K]  single_digi_stamp.yml
│   │   │   │   │   ├── [1.2K]  single_wrongtime_digi_gru.yml
│   │   │   │   │   ├── [1.2K]  single_wrongtime_digi_knn.yml
│   │   │   │   │   ├── [1009]  single_wrongtime_digi_narm.yml
│   │   │   │   │   ├── [1.0K]  single_wrongtime_digi_nextitnet.yml
│   │   │   │   │   ├── [ 956]  single_wrongtime_digi_sr.yml
│   │   │   │   │   └── [1.0K]  single_wrongtime_digi_stamp.yml
│   │   │   │   ├── [1.2K]  single_digi_baselines.yml
│   │   │   │   ├── [1.4K]  single_digi_models.yml
│   │   │   │   ├── [1.5K]  single_multiple_digi_baselines.yml
│   │   │   │   ├── [1.4K]  single_multiple_digi_models.yml
│   │   │   │   ├── [1.4K]  single_multiple_wrongtime_digi_baselines.yml
│   │   │   │   ├── [1.6K]  single_multiple_wrongtime_digi_models.yml
│   │   │   │   ├── [1.3K]  single_wrongtime_digi_baselines.yml
│   │   │   │   └── [1.4K]  single_wrongtime_digi_models.yml
│   │   │   └── [ 39K]  window
│   │   │       ├── [ 10K]  opt
│   │   │       │   ├── [1.2K]  window_digi_gru.yml
│   │   │       │   ├── [1.2K]  window_digi_knn.yml
│   │   │       │   ├── [1011]  window_digi_narm.yml
│   │   │       │   ├── [1.0K]  window_digi_nextitnet.yml
│   │   │       │   ├── [ 958]  window_digi_sr.yml
│   │   │       │   └── [1.0K]  window_digi_stamp.yml
│   │   │       ├── [ 10K]  opt_wrongtime
│   │   │       │   ├── [1.2K]  window_wrongtime_digi_gru.yml
│   │   │       │   ├── [1.2K]  window_wrongtime_digi_knn.yml
│   │   │       │   ├── [1017]  window_wrongtime_digi_narm.yml
│   │   │       │   ├── [1.0K]  window_wrongtime_digi_nextitnet.yml
│   │   │       │   ├── [ 964]  window_wrongtime_digi_sr.yml
│   │   │       │   └── [1.0K]  window_wrongtime_digi_stamp.yml
│   │   │       ├── [1.3K]  window_digi_baselines.yml
│   │   │       ├── [1.7K]  window_digi_models.yml
│   │   │       ├── [1.5K]  window_multiple_digi_baselines.yml
│   │   │       ├── [1.8K]  window_multiple_digi_models.yml
│   │   │       ├── [1.5K]  window_wrongtime_digi_baselines.yml
│   │   │       ├── [1.6K]  window_wrongtime_digi_models.yml
│   │   │       ├── [2.6K]  window_wrongtime_multiple_digi_baselines.yml
│   │   │       └── [2.5K]  window_wrongtime_multiple_digi_models.yml
│   │   ├── [ 24K]  nowplaying
│   │   │   └── [ 20K]  window
│   │   │       ├── [8.3K]  opt
│   │   │       │   ├── [1.2K]  window_aotm_knn.yml
│   │   │       │   ├── [1.2K]  window_nowplaying_gru.yml
│   │   │       │   ├── [1.0K]  window_nowplaying_nextitnet.yml
│   │   │       │   └── [ 939]  window_nowplaying_sr.yml
│   │   │       ├── [1.4K]  window_multiple_nowplaying_baselines.yml
│   │   │       ├── [1.7K]  window_multiple_nowplaying_models.yml
│   │   │       ├── [1.2K]  window_multiple_nowplaying_stamp.yml
│   │   │       ├── [1.3K]  window_nowplaying_baselines.yml
│   │   │       ├── [1.6K]  window_nowplaying_models.yml
│   │   │       └── [1.0K]  window_nowplaying_stamp.yml
│   │   ├── [ 26K]  retailrocket
│   │   │   ├── [2.0K]  hybrids_window_retail.yml
│   │   │   └── [ 20K]  window
│   │   │       ├── [ 10K]  opt
│   │   │       │   ├── [1.2K]  window_retailrocket_gru.yml
│   │   │       │   ├── [1.0K]  window_retailrocket_knnidf.yml
│   │   │       │   ├── [1.2K]  window_retailrocket_knn.yml
│   │   │       │   ├── [ 996]  window_retailrocket_narm.yml
│   │   │       │   ├── [1021]  window_retailrocket_nextitnet.yml
│   │   │       │   └── [ 942]  window_retailrocket_sr.yml
│   │   │       ├── [1.4K]  window_multiple_retailr_baselines.yml
│   │   │       ├── [1.5K]  window_multiple_retailr_models.yml
│   │   │       ├── [1.3K]  window_retailr_baselines.yml
│   │   │       └── [1.4K]  window_retailr_models.yml
│   │   ├── [ 29K]  rsc15
│   │   │   ├── [1.7K]  hybrids_window_rsc15_multiple.yml
│   │   │   └── [ 23K]  window
│   │   │       ├── [9.3K]  opt
│   │   │       │   ├── [1.2K]  window_rsc15_gru.yml
│   │   │       │   ├── [1.0K]  window_rsc15_knnidf.yml
│   │   │       │   ├── [1.2K]  window_rsc15_knn.yml
│   │   │       │   ├── [1.0K]  window_rsc15_nextitnet.yml
│   │   │       │   └── [ 948]  window_rsc15_sr.yml
│   │   │       ├── [1.4K]  window_multiple_rsc15_baselines.yml
│   │   │       ├── [1.5K]  window_multiple_rsc15_models.yml
│   │   │       ├── [1.3K]  window_rsc15_baselines.yml
│   │   │       ├── [ 831]  window_rsc15_memory.yml
│   │   │       ├── [1.4K]  window_rsc15_models.yml
│   │   │       ├── [ 770]  window_rsc15_time_ct.yml
│   │   │       ├── [ 823]  window_rsc15_time_nextitnet.yml
│   │   │       └── [1.6K]  window_rsc15_time.yml
│   │   ├── [ 24K]  rsc15_4
│   │   │   └── [ 20K]  single split
│   │   │       ├── [9.2K]  opt
│   │   │       │   ├── [1.2K]  single_rsc15_4_gru.yml
│   │   │       │   ├── [1.2K]  single_rsc_15_4_knn.yml
│   │   │       │   ├── [ 989]  single_rsc15_4_narm.yml
│   │   │       │   ├── [ 936]  single_rsc_15_4_sr.yml
│   │   │       │   └── [1023]  single_rsc15_4_stamp.yml
│   │   │       ├── [1.4K]  single_multiple_rsc15_4_baselines.yml
│   │   │       ├── [1.3K]  single_multiple_rsc15_4_models.yml
│   │   │       ├── [1.3K]  single_rsc15_4_baselines.yml
│   │   │       ├── [ 861]  single_rsc15_4_ct.yml
│   │   │       └── [1.4K]  single_rsc15_4_models.yml
│   │   └── [ 20K]  rsc15_64
│   │       ├── [1.4K]  single_multiple_rsc15_64_baselines.yml
│   │       ├── [1.6K]  single_multiple_rsc15_64_models.yml
│   │       ├── [1.3K]  single_rsc15_64_baselines.yml
│   │       ├── [1.3K]  single_rsc15_64_models.yml
│   │       └── [ 10K]  single split
│   │           └── [6.1K]  opt
│   │               ├── [1.2K]  single_rsc_15_64_knn.yml
│   │               └── [ 945]  single_rsc_15_64_sr.yml
│   └── [ 11K]  save_slist
│       ├── [1.1K]  1fold_digi_swalk.yml
│       ├── [1.0K]  1fold_rsc15_4_swalk.yml
│       ├── [1.0K]  1fold_rsc15_64_slist.yml
│       ├── [1.1K]  5fold_digi_slist.yml
│       ├── [1.1K]  5fold_nowp_slist.yml
│       ├── [1.1K]  5fold_rr_slist.yml
│       └── [1.1K]  5fold_rsc_slist.yml
├── [4.5M]  data
│   └── [4.5M]  rsc15
│       └── [4.5M]  prepared
│           ├── [405K]  yoochoose-clicks-100k_test.txt
│           ├── [2.0M]  yoochoose-clicks-100k_train_full.txt
│           ├── [1.6M]  yoochoose-clicks-100k_train_tr.txt
│           └── [493K]  yoochoose-clicks-100k_train_valid.txt
├── [ 593]  environment_cpu.yml
├── [ 670]  environment_gpu.yml
├── [159K]  evaluation
│   ├── [7.7K]  evaluation_last.py
│   ├── [5.1K]  evaluation_multiple.py
│   ├── [ 17K]  evaluation.py
│   ├── [   0]  __init__.py
│   ├── [ 19K]  loader.py
│   └── [106K]  metrics
│       ├── [6.7K]  accuracy_ext.py
│       ├── [ 23K]  accuracy_grt.py
│       ├── [ 23K]  accuracy_leq.py
│       ├── [ 19K]  accuracy_multiple.py
│       ├── [8.2K]  accuracy.py
│       ├── [3.2K]  artist_coherence.py
│       ├── [3.1K]  artist_diversity.py
│       ├── [2.9K]  coverage.py
│       ├── [   0]  __init__.py
│       ├── [3.2K]  popularity.py
│       ├── [4.4K]  saver_next.py
│       ├── [3.1K]  saver.py
│       └── [2.6K]  time_memory_usage.py
├── [4.0K]  grid_search_knn.py
├── [ 11K]  helper
│   ├── [   0]  __init__.py
│   └── [7.5K]  stats.py
├── [369K]  images
│   ├── [264K]  img0.png
│   ├── [ 82K]  img1.png
│   └── [ 19K]  process_flow_prototype_1.svg
├── [ 92K]  nbs
│   ├── [ 25K]  M428282_RetailRocket_Data_Preprocessing.ipynb
│   └── [ 62K]  P293191_SLIST_on_Yoochoose_Preprocessed_Sample_Dataset.ipynb
├── [141K]  preprocessing
│   ├── [7.2K]  generate_item_matrix_cease.py
│   ├── [   0]  __init__.py
│   ├── [ 20K]  preprocess_diginetica.py
│   ├── [ 13K]  preprocess_dressipi.py
│   ├── [ 22K]  preprocessing.py
│   ├── [ 11K]  preprocess_music.py
│   ├── [ 11K]  preprocess_playlist.py
│   ├── [ 14K]  preprocess_retailrocket.py
│   ├── [ 15K]  preprocess_rsc15.py
│   ├── [ 12K]  preprocess_tmall.py
│   ├── [ 11K]  preprocess_windeln.py
│   └── [2.2K]  run_preprocessing.py
├── [2.1K]  README.md
├── [ 30K]  run_config.py
├── [2.6K]  run_preprocessing.py
├── [ 69K]  trial_codes
│   ├── [4.4K]  check_dataset.py
│   ├── [1.7K]  create_pseudo_prod2vec.py
│   ├── [4.7K]  generate_item_matrix_word2vec.py
│   ├── [ 11K]  preprocessing_extract.py
│   ├── [ 10K]  preprocessing_extract.txt
│   ├── [2.1K]  recreate_image.py
│   ├── [ 758]  run_multvae_trial.py
│   ├── [1.1K]  run_recvae_trial.py
│   ├── [2.8K]  run_slist_ext_trial.py
│   ├── [1.2K]  run_slist_trial.py
│   ├── [3.3K]  run_srgnn_trial.py
│   ├── [ 17K]  trial_knn
│   │   ├── [1.6K]  generate_ground_truth_b.py
│   │   ├── [2.4K]  generate_train_test.py
│   │   ├── [3.6K]  knn.py
│   │   └── [5.2K]  knn_utils.py
│   └── [4.2K]  troubleshoot_slist.py
└── [ 15K]  utils
    ├── [5.3K]  knn_utils.py
    ├── [4.3K]  sr_gnn_utils.py
    └── [1.3K]  utils.py

 6.1M used in 61 directories, 252 files
```