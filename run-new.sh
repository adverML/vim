
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py  --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/spatial/resnet18_train.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py  --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/spatial/resnet18_test.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py  --batch 1024 --fc_save_path /home/lorenzp/workspace/competence_estimation/features/cifar10/spatial


CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode "|phase|" --batch 1024 --out_file "/home/lorenzp/workspace/competence_estimation/features/cifar10/|phase|/resnet18_train.npy"
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode "|phase|" --batch 1024 --out_file "/home/lorenzp/workspace/competence_estimation/features/cifar10/|phase|/resnet18_test.npy"
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode "|phase|" --batch 1024 --fc_save_path "/home/lorenzp/workspace/competence_estimation/features/cifar10/|phase|"


CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode phase --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/phase/resnet18_train.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode phase --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/phase/resnet18_test.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode phase --batch 1024 --fc_save_path /home/lorenzp/workspace/competence_estimation/features/cifar10/phase


CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode magnitude --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/magnitude/resnet18_train.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode magnitude --batch 1024 --out_file /home/lorenzp/workspace/competence_estimation/features/cifar10/magnitude/resnet18_test.npy 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm-new.py --mode magnitude --batch 1024 --fc_save_path /home/lorenzp/workspace/competence_estimation/features/cifar10/magnitude


# lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_phase_2023-10-17_08:36:12.pt
# lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_magnitude_2023-10-16_20:05:23.pt
# lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_|phase|_2023-10-17_08:36:00.pt
