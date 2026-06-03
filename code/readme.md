### 这是论文复现代码的说明
在miniconda中，创建新环境：
conda create -n 新环境名 python=3.10
conda activate 新环境名
在当前目录下安装依赖：
conda env update -f environment.yml

### 修改变量以及参数
通过修改args.py来修改论文使用模型的各个超参数
必须要修改的是total_path，要将路径修改为模型实际存在的路径
然后是choose_model，将其修改为total_path的你想要的某个键，当然，pruner，protect_tail和total_rank的键也需要保持同步
only_pruner用于控制是否进行校准，如果只想要剪枝，就设置为True；反之设置为False

### 运行代码
python gini.py
然后整个系统会自动完成剪枝以及校准的步骤，最后输出新的模型，会保存在当前目录下的./model中，同时收集的隐藏向量会缓存在./hidden_state中，其元信息会存在相应的json文件中，如果想要删除某个模型的某个层之间的cache，可以查看json文件并手动删除相应的cache。它的作用是使得整个流程只需要收集一次隐藏向量即可，后面的训练就可以复用相应的训练数据，加快效率

