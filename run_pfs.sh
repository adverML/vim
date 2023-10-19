
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS" 
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS"   
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS"  
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS"  

# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
# CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096  --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy




CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfpgd --preprocess "PFS" 
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack fgsm    --preprocess "PFS"   
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack l2df    --preprocess "PFS"  
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfdf  --preprocess "PFS"  

CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfpgd --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack fgsm    --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack l2df    --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy
CUDA_VISIBLE_DEVICES=0; python extract_feature_timm.py --batch 4096 --attack linfdf  --preprocess "PFS" --out_file  /home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy