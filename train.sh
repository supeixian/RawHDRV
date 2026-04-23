# 基础训练命令（4倍缩放，GPU 0）
python trainHDR_RVours_mask1.py --model RawHDRV --gpu_id 0 --scale 1 --continue_train
python trainHDR.py --model Nocache_HDR --gpu_id 0 --scale 1
python trainHDR.py --model Cache_HDR --gpu_id 0 --scale 1 --enable_memcache=True
python train_rawhdr.py --model Cache_HDR --gpu_id 1 --scale 1 
python trainHDR.py --model RRVSR_HDR_npy --gpu_id 0 --scale 1

python trainHDR_RVours_mask1.py --model Ours4 --gpu_id 0 --scale 1 --continue_train

# 继续训练命令（加载最新检查点）
python trainHDR.py --model Cache_HDR --gpu_id 0 --scale 1 --enable_memcache=True --continue_train
python trainHDR.py --model Nocache_HDR --gpu_id 0 --scale 1 --continue_train

# 多GPU训练（使用GPU 0和1）
python trainHDR.py --model RRVSR_HDR --gpu_id 0,1 --scale 1 


# 在项目根目录执行（假设数据存放在 datasets/Train/）
python data/preprocess_raw.py --data_root ./datasets/Train


# 删除所有以"temp_"开头的文件
python delete.py ./your_directory --prefix "temp_"
# 删除所有以".meta.npz"结尾的文件 
python delete.py ./your_directory --suffix ".meta.npz"
# 同时指定前缀和后缀
python delete.py ./your_directory --prefix "temp_" --suffix ".tmp"

# 测试指令
python testHDR.py --model Nocache_HDR --gpu_id 0 --save_image True
python testrawHDR.py --model Cache_HDR --gpu_id 0 --save_image True
python testHDR_RVours_maskf.py --model Ours_5_1 --gpu_id 1 --save_image True