
## 科研代码课作业 一

# 1修改VQ-GAN的代码，替换其中的minigpt 为nanogpt.

VQ-GAN的代码链接：https://github.com/AlexWang1900/pytorch_vq_gan_refactor

数据见：https://pan.baidu.com/s/1eQy-RuZyiWnl1FdoNgvxOg?pwd=5qgx 

提取码：5qgx 

下载解压后放入：/data/FFHQ_128/ *.png

nanogpt代码链接：https://github.com/karpathy/nanoGPT

完成标准：能成功运行training_transformer.py 训练10个EPOC以上不报错，/result 文件夹会有新生成的图片，清晰可见是人脸即可。

# 2修改VQ-GAN的代码，将Codebook模块替换为FSQ模块，

FSQ模块见论文：Finite scalar quantization:VQ-VAE made simple

VQ-GAN的代码链接：https://github.com/AlexWang1900/pytorch_vq_gan_refactor

FSQ 代码：自行寻找

完成标准： 能成功运行training_vqgan.py, 训练10个EPOCH 以上不报错 /result 文件夹会有新生成的图片，清晰可见是人脸即可。

# 3在问题2完成了的基础上，做出修改，实现完整的FSQ功能的VQGAN

完成标准：
能成功训练带有FSQ的training_transformer.py 训练10个EPOCH 以上不报错，/result 文件夹会有新生成的图片，清晰可见是人脸即可。

# 4 提交：

将最终代码压缩成.zip文件，不包含checkpoint,可以包含result文件夹

文件名: 

例如3个问题全部完成：homework1_q123.zip 

只做了问题1：homework1_q1.zip

做了问题1，2：homework1_q12.zip


# 评分标准：

第一个问题1分，第二个问题2分，第三个问题3分

能达到完成标准记分，不能达到完成标准不计分

3不能抄袭别人的答案，发现完全雷同的两份或更多作业，一起作废。

4，多人得分一样的情况下，按照代码提交先后决定名次。

## Train VQGAN on your own data:
### Training First Stage
1. (optional) Configure Hyperparameters in ```training_vqgan.py```
2. Set path to dataset in ```training_vqgan.py```
3. ```python training_vqgan.py```

### Training Second Stage
1. (optional) Configure Hyperparameters in ```training_transformer.py```
2. Set path to dataset in ```training_transformer.py```
3. ```python training_transformer.py```


## Citation
```bibtex
@misc{esser2021taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2021},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
