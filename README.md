# **GFlow**

**UserName: YJxyzxyz**

## 工程记录

### 环境配置

先安装pytorch → Mast3R MSplat  → requiresment 

PS：UniMatch的函数也需要clone到本地

使用的版本 Python == **3.9.11** CUDA == **12.4** pytorch ==  **2.5.1** gpu版本 这个版本是否可以？

### 预训练的模型

Mast3R: **MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth**

UniMatch: **gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth**

### 数据集

使用 DAVIS 2017 TrainVal 480p

#### 深度图

解决：宽高调换

./scripts/organize_davis.sh

在这些数据集下跑深度图有问题：

1. blackswan
2. cat-girl
3. flamingo
4. kite-surf
5. paragliding
6. paragliding-launch
7. parkour
8. scooter-board
9. surf
10. varanus-cage
11. walking

会报错：

```powershell
Traceback (most recent call last):
  File "F:\sp24\GFlow-organized\utility\depth_mast3r.py", line 255, in <module>
    tyro.cli(main)
  File "E:\softwarezjq\anaconda\envs\4D\lib\site-packages\tyro\_cli.py", line 229, in cli
    return run_with_args_from_cli()
  File "F:\sp24\GFlow-organized\utility\depth_mast3r.py", line 71, in main
    inference_mast3r(model, device, images_list[i:i+seg_size], input_dir, cache_dir,
  File "F:\sp24\GFlow-organized\utility\depth_mast3r.py", line 118, in inference_mast3r
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=True)
  File "F:\sp24\GFlow-organized\./third_party/mast3r\mast3r\cloud_opt\sparse_ga.py", line 83, in get_dense_pts3d
    idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
  File "F:\sp24\GFlow-organized\./third_party/mast3r\mast3r\cloud_opt\sparse_ga.py", line 867, in anchor_depth_offsets
    assert (core_depth > 0).all()
AssertionError
```

在 assert (core_depth > 0).all() 这一行触发了 AssertionError 

疑问：意思是通过这个模型生成的深度图有负数？？ 导致了报错→运行环境的问题还是模型或者数据集的问题？

使用的硬件环境：

自己的笔记本 3060 6GB显存 （跑深度图时显示显存不够 Out of Memory）

工位的电脑 1660 6GB显存 （跑深度图时显示显存不够 Out of Memory）

公用3090 24GB显存 （足够）但是不知道跑出来的是否正确

注释掉Assertion的代码后，可以正常生成深度图

尝试print出core_depth

结果如下

```python
print(core_depth)
print("min:", core_depth.min())
print("max:", core_depth.max())
print("shape:", core_depth.shape)
```

```powershell
tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
min: tensor(1., device='cuda:0')
max: tensor(1., device='cuda:0')
shape: torch.Size([36864])
```

```powershell
tensor([0.9969, 0.9972, 1.0013,  ..., 1.0544, 1.0859, 0.9982], device='cuda:0')
min: tensor(0.3870, device='cuda:0')
max: tensor(1.9814, device='cuda:0')
shape: torch.Size([2304])
```

...

```powershell
tensor([0.9985, 0.9976, 1.0045,  ..., 0.9995, 1.0033, 1.0045], device='cuda:0')
min: tensor(-0.0652, device='cuda:0')
max: tensor(2.5606, device='cuda:0')
shape: torch.Size([2304])
```

确实是出现了负数

#### 光流图

./scripts/flow_unimatch.sh

所有数据集均能跑

但是不知道是否正确

#### 掩模图

./scripts/move_seg.sh parent_folder

所有数据集均能跑

但是不知道是否正确

### TAP-Vid DAVIS 

如果清空davis文件夹会生成数据集并包含tracking.pkl文件

但是数据集并不完全，只有下面这些有

![f56b744dfe8bc5cff580cc10914d54b8](D:\QQ-NT\Tencent Files\Tencent Files\718005487\nt_qq\nt_data\Pic\2024-11\Ori\f56b744dfe8bc5cff580cc10914d54b8.png)

这个pkl文件是否会影响代码效果？后续才用到

### RUN GFlow

运行fit_video文件后

会产生提示：

```powershell
F:\sp24\GFlow-organized\./third_party/mast3r\mast3r\model.py:24: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to
 construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `w
eights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the use
r via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
```

这个torch.load的提示是否影响加载？实际并不影响代码运行

会生成一系列结果：

![57fbe776d1c874c801041e4581dca394](D:\QQ-NT\Tencent Files\Tencent Files\718005487\nt_qq\nt_data\Pic\2024-11\Ori\57fbe776d1c874c801041e4581dca394.png)

这个结果是否正确？

在某些测试集上，如car-turn，运行后会报错

```powershell
Traceback (most recent call last):
  File "F:\Project\GFlow-organized\gflow\fit_video.py", line 409, in <module>
    tyro.cli(main)
  File "C:\Users\Admin\.conda\envs\4D\lib\site-packages\tyro\_cli.py", line 231, in cli
    return run_with_args_from_cli()
  File "F:\Project\GFlow-organized\gflow\fit_video.py", line 386, in main
    traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, occulasions=occulasions, filename="sequence_traj_vis", still_length=closest_points_still.shape[0])
  File "F:\Project\GFlow-organized\gflow\utils\traj_visualizer.py", line 136, in visualize
    res_video = self.draw_tracks_on_video(
  File "F:\Project\GFlow-organized\gflow\utils\traj_visualizer.py", line 224, in draw_tracks_on_video
    tracks[query_frame, still_length:, 1].min(),
  File "C:\Users\Admin\.conda\envs\4D\lib\site-packages\numpy\core\_methods.py", line 45, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation minimum which has no identity
```

定位到traj_visualizer.py文件中是在调用 tracks[query_frame, still_length:, 1].min()是使用了空数组，切片的结果是一个空数组

尝试：

在调用 min() 之前，添加一些调试输出

```python
   # Ensure tracks and still_length are valid
    if still_length > 0 and tracks[query_frame, still_length:, 1].size > 0:
        y_move_min, y_move_max = (
            tracks[query_frame, still_length:, 1].min(),
            tracks[query_frame, still_length:, 1].max(),
        )
        move_norm = plt.Normalize(y_move_min, y_move_max)
        for n in range(still_length, N):
            color = self.color_map(move_norm(tracks[query_frame, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)
    else:
        print(f"Warning: Empty array for tracks at query_frame {query_frame} and still_length {still_length}")
```

```powershell
Warning: Empty array for tracks at query_frame 0 and still_length 151
```

```powershell
Video saved to data\davis\car-turn\car-turn_logs_cam_init_only\2024_11_21-12_23_18\sequence_traj_vis.mp4
Warning: Empty array for tracks at query_frame 0 and still_length 0
```

文章对应模块

Shape of  Motion

First：

attempt auto 参数调整 跟优化相关的 比如 次数 学习率  

depthfucation 阈值

全局优化

1.重建的RGB视频

2.高斯点中心视频

3.优化后的

4.深度

5.第一帧的优化

viewer.py 可以看网页



将代码中的下采样倍数从八倍改成两倍之后可以正常生成深度图，但是分辨率不够，需要debug;

```python
# Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
scene = sparse_global_alignment(images_list, pairs, cache_dir,
                                model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                matching_conf_thr=matching_conf_thr, subsample=8, **kw)
```



subsample=2或者4是生成的深度图带有强烈的条纹纹影?



