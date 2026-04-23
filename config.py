import argparse

def get_train_config(args):
    opt = argparse.Namespace()
    # 修改目标曝光值为实际存在的EV值
    opt.target_exposure = 3  # 原值为6，现改为3以匹配EV_-3到EV_3
    # 添加参数范围校验
    assert -3 <= opt.target_exposure <= 3, "曝光值需在[-3,3]范围内"

    model = args.model
    gpu_id = args.gpu_id
    scale = 1
    continue_train = args.continue_train

    opt_parser = argparse.ArgumentParser(description='Training module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--init_iters', type=int, default=0)
    opt_parser.add_argument('--num_iters', type=int, default=300e3)
    opt_parser.add_argument('--lr', type=float, default=1e-4)
    opt_parser.add_argument('--N_frames', type=int, default=5)
    opt_parser.add_argument('--batch_size', type=int, default=4)
    opt_parser.add_argument('--valid_batch_size', type=int, default=1)
    opt_parser.add_argument('--n_workers', type=int, default=4)
    opt_parser.add_argument('--LR_size', type=int, default=128)
    opt_parser.add_argument('--target_exposure', type=int, default=0)
    opt_parser.add_argument('--clip_stride', type=int, default=1)
    opt_parser.add_argument('--crop_size', type=int, default=256)
    opt_parser.add_argument('--enable_memcache', type=bool, default=True,
                          help='启用内存缓存加速数据加载')
    opt_parser.add_argument('--memcache_size', type=int, default=200,
                          help='内存缓存的最大样本数（需小于系统内存容量）')
    opt_parser.add_argument('--enable_dynamic_levels', type=bool, default=True,help='启用自动电平检测')
    opt_parser.add_argument('--train_root', type=str, default='./datasets/Train/')
    opt_parser.add_argument('--test_root', type=str, default='./datasets/Test/')

    opt_parser.add_argument('--train_paths_LR_RAW', type=str, default='./datasets/Train/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--train_paths_LR_RGB', type=str, default='./datasets/Train/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--train_paths_HR_RGB', type=str, default='./datasets/Train/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))

    opt_parser.add_argument('--test_paths_LR_RAW', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_LR_RGB', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_HR_RGB', type=str, default='./datasets/Test/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))
    
    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/')
    opt_parser.add_argument('--model', type=str, default=model)
    opt = opt_parser.parse_args()

    return opt

def get_test_config(args):
    model = args.model
    gpu_id = args.gpu_id
    scale = 1
    save_image = args.save_image


    opt_parser = argparse.ArgumentParser(description='Testing module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--N_frames', type=int, default=5)
    opt_parser.add_argument('--batch_size', type=int, default=1)
    opt_parser.add_argument('--n_workers', type=int, default=4)

    opt_parser.add_argument('--test_paths_LR_RAW', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_raw/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_LR_RGB', type=str, default='./datasets/Test/{0:d}X/lr_{1:d}_rgb/'.format(scale, scale))
    opt_parser.add_argument('--test_paths_HR_RGB', type=str, default='./datasets/Test/{0:d}X/hr_{1:d}_rgb/'.format(scale, scale))

    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/', help='weight_savepath')
    opt_parser.add_argument('--model', type=str, default=model, help='base model')
    opt_parser.add_argument('--save_image', type=bool, default=save_image)
    opt = opt_parser.parse_args()

    return opt
