import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    cmd = f"python training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu}"

    os.system(cmd)



#run_eval(0, 'cifar10', 'stl10', 'Y:/BadEncoder/output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
#run_eval(0, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')
#run_eval(0, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one')

run_eval(0, 'cifar10', 'stl10', r'X:\Directory\code\Invisiable-Construction-Learning-Backdoor\output\cifar10\stl10_backdoored_encoder\model_250.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
#run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
#run_eval(0, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')
