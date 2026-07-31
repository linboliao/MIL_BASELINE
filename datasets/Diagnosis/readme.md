#### 对比
#### train
##### PFM 对比
- 对比PFM CONCH、h-optimus-1、mstar、omiclip、UNI、UNI2、virchow2
- stains: 无
- MIL: AB MIL
- magnification: 20x
- 对比FPM: 
- 对比指标: 诊断性能, 特征提取耗时
- 路径前缀：/NAS145/liaolinbo/Data/MXB/CLS/feat_0_224/pt_files/（CONCH、h-optimus-1、mstar、omiclip、UNI、UNI2、virchow2）
#### Magnification 对比
- PFM: H-optimus-1
- MIL: AB MIL
- stains: 无
- 对比倍率: 20x, 10x, 5x
- 对比指标: 诊断性能, 特征提取耗时
- 路径前缀：/NAS145/liaolinbo/Data/MXB/CLS/feat_0_224（feat_0_448、feat_0_896）/pt_files/h-optimus-1
#### stains 对比
- PFM: H-optimus-1
- MIL: AB MIL
- magnification: 10x
- 对比染色: Macenko, Reinhard, Vahadane
- 对比指标: 诊断性能, 特征提取耗时
- 路径前缀：/NAS145/liaolinbo/Data/MXB/CLS/feat_0_448/stains/Macenko（Reinhard、Vahadane）/pt_files
##### MIL 对比
- PFM: H-optimus-1
- stains: Reinhard
- magnification: 10x
- 对比MIL: AB_MIL、TRANS_MIL、CLAM_MIL、WIKG_MIL、MAMBA_MIL、MICRO_MIL、MAMBA2D_MIL、AEM_MIL、MICO_MIL、MSM_MIL、TDA_MIL、GDF_MIL
- 对比指标: 诊断性能, 特征提取耗时
####
测试数据：NAS145/liaolinbo/Data/MXB/CLS测试/