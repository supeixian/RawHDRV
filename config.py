import argparse

def get_train_config(args):
    model = args.model
    gpu_id = args.gpu_id

    opt_parser = argparse.ArgumentParser(description='Training module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--number_epochs', type=int, default=100)
    opt_parser.add_argument('--lr', type=float, default=1e-4)
    opt_parser.add_argument('--N_frames', type=int, default=3)
    opt_parser.add_argument('--batch_size', type=int, default=4)
    opt_parser.add_argument('--valid_batch_size', type=int, default=1)
    opt_parser.add_argument('--target_exposure', type=int, default=0)
    opt_parser.add_argument('--crop_size', type=int, default=128)
    opt_parser.add_argument('--train_root', type=str, default='./datasets/Train/')
    opt_parser.add_argument('--test_root', type=str, default='./datasets/Test/')
    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/')
    opt_parser.add_argument('--model', type=str, default=model)
    opt_parser.add_argument('--continue_train', action='store_true', help='Resume training from checkpoint')
    opt = opt_parser.parse_args()

    return opt

def get_test_config(args):
    model = args.model
    gpu_id = args.gpu_id
    save_image = args.save_image

    opt_parser = argparse.ArgumentParser(description='Testing module')
    opt_parser.add_argument('--gpu_id', type=str, default=gpu_id)
    opt_parser.add_argument('--N_frames', type=int, default=3)
    opt_parser.add_argument('--batch_size', type=int, default=1)
    opt_parser.add_argument('--n_workers', type=int, default=4)
    opt_parser.add_argument('--target_exposure', type=int, default=0)
    opt_parser.add_argument('--crop_size', type=int, default=512)
    opt_parser.add_argument('--train_root', type=str, default='./datasets/Train/')
    opt_parser.add_argument('--test_root', type=str, default='./datasets/Test/')
    opt_parser.add_argument('--weight_savepath', type=str, default='./weight_checkpoints/', help='weight_savepath')
    opt_parser.add_argument('--model', type=str, default=model, help='base model')
    opt_parser.add_argument('--save_image', type=bool, default=save_image)

    # Use defaults only for test config.
    opt = opt_parser.parse_args([])

    # Fixed-coordinate cropping is disabled by default.
    opt.crop_coords = None
    opt.crop_size_custom = None
	
    return opt
