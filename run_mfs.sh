# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS" 
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS"   
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS"  
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS"  

# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy

CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfpgd --preprocess "MFS" 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack fgsm    --preprocess "MFS"   
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack l2df    --preprocess "MFS"  
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfdf  --preprocess "MFS"  

CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfpgd --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack fgsm    --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack l2df    --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfdf  --preprocess "MFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy