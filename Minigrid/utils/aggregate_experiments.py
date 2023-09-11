import argparse
import glob
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--seeds", help="list of seeds to aggregate over")
args = parser.parse_args()

all_csv_files = glob.glob("*csv")
seeds = args.seeds.split(",")

experiment_strings = [word[:-6] for word in all_csv_files]
experiment_strings = list(set(experiment_strings))
print(experiment_strings)

for experiment_string in experiment_strings:
    experiments = list(filter(lambda k: experiment_string in k, all_csv_files))
    N_rows, N_cols = pd.read_csv(experiments[0]).shape
    results = np.zeros((N_rows, N_cols))
    for csv_file in experiments:
        results += pd.read_csv(csv_file).values
    results /= len(experiments)
    np.save(experiment_string + "mean.npy", results)


'''
这段Python代码是用于读取多个CSV文件，并对每个实验（通过实验字符串进行标识）的结果进行平均，然后将平均后的结果保存为.npy文件。

以下是每个部分的详细说明：

argparse.ArgumentParser：首先，使用argparse库创建了一个命令行输入参数解析器。

parser.add_argument("--seeds", ...)：然后，定义了一个命令行参数 --seeds，它接受一组逗号分隔的种子值。

args = parser.parse_args()：解析命令行参数。

all_csv_files = glob.glob("*csv")：使用glob库获取当前目录下所有的CSV文件。

seeds = args.seeds.split(",")：将输入参数--seeds以逗号为分隔符转化为列表。

experiment_strings：去除文件名后缀.csv并对其去重，得到不重复的实验字符串。

然后，代码对每个实验字符串执行以下操作：

experiments = list(filter(lambda k: ...)：找出文件名包含当前实验字符串的所有CSV文件。

N_rows, N_cols = pd.read_csv(experiments[0]).shape：读取第一个CSV文件，获取其行数和列数。

results = np.zeros((N_rows, N_cols))：创建一个大小与CSV文件相同的零矩阵。

for csv_file in experiments：遍历所有的CSV文件，将每个文件的数据加到零矩阵上。

results /= len(experiments)：将累加的结果除以CSV文件的数量，得到每个实验的平均结果。

np.save(experiment_string + "mean.npy", results)：最后，将平均后的结果保存为.npy文件。

总结：这段代码的主要功能是对同一实验（通过文件名中的实验字符串识别）的多个CSV文件进行平均，然后将结果保存为.npy文件。这在处理多次实验结果、需要计算平均性能时非常有用。

'''





