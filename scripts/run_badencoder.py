import os

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')

def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, clean_encoder='model_1000.pth'):

    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'python badencoder.py \
    --lr 0.001 \
    --batch_size 256 \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder Y:/BadEncoder/output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --start_epoch 1 \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file ./trigger/{trigger}'
    os.system(cmd)

# --pretrained_encoder X:\Directory\code\Invisiable-Construction-Learning-Backdoor\output\cifar10\stl10_backdoored_encoder\model_100.pth
# badencoder断开所以需要从续点重新训练，因此修改⬇️的代码为⬆️，并且把start_epoch修改为101
# --pretrained_encoder Y:/BadEncoder/output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
# run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck')
# run_finetune(0, 'cifar10', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority')
run_finetune(0, 'cifar10', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'svhn_one')
# run_finetune(0, 'cifar10', 'cifar10', 'cifar10', 'trigger_pt_white_21_10_ap_replace.npz', 'cifar10_airplane')
