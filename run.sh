#data='mvtec_ad'
#model='cosface'

#for i in 0 1 2 3 4 5 6 7 8 9
#do
#    echo $i
#    python run_metric_learning.py train with configs/$data/$model.yaml -f dataset.params.fold_idx=$i
#    python run_metric_learning.py inference with configs/$data/$model.yaml -f dataset.params.fold_idx=$i
#done

#for model in 'l2_autoencoder' 'ssim_autoencoder'
#    for c in 'carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper'
#    do
#        echo $c >> result.txt
#        python run_autoencoder.py train with configs/$data/$model.yaml -f dataset.params.target=$c 
#        python run_autoencoder.py inference with configs/$data/$model.yaml -f dataset.params.target=$c >> result.txt
#    done
#done

data='cifar10'

# metric learning
#for model in 'cosface' 'sphereface' 'arcface' 'adacos'
#do    
#    for t in 'plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'
#        do
#        echo $t>> result.txt
#        python run_metric_learning.py train with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t
#        python run_metric_learning.py inference with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t >> result.txt
#        rm -r train_dirs/$data/$model
#    done
#done

# autoencoder
for model in 'l2_autoencoder' 'ssim_autoencoder'
do
    for t in 'plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'
    do
        echo $t >> result.txt
        python run_autoencoder.py train with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t
        python run_autoencoder.py inference with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t >> result.txt
        rm -r train_dirs/$data/ssimae
        rm -r train_dirs/$data/l2ae
    done
done

# gan
"""
for model in 'f-anogan'
do
    for t in 'plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'
    do
        echo $t >> result.txt
        python run_gan.py train with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t
        python run_gan.py inference with configs/$data/$model.yaml -f dataset.params.anomaly_classes=$t >> result.txt
        rm -r train_dirs/$data/sagan
    done
done
"""
