rm *npy
rm *png
rm -r storage/*
rm *mp4
frames_before_resets=(8000000000)
environment=MiniGrid-MultiRoom-N4-S5-v0
randomise_env=False
frames=1400000
random_seeds=(96)

for frames_before_reset in ${frames_before_resets[@]}; do  
    #random_action=True
    random_action=False
    save_interval=2000
    visualizing=False
   
    reward_weighting=10

    icm_lr=0.001
    #icm_lr=0.0001
    
    #noisy_tv=(False)
    noisy_tv=(True)
    
    curiosity=(True)
    #curiosity=(False)
    
    uncertainty=(True)
    #uncertainty=(False)
    normalise_reward=True

    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train_a2c --algo a2c --visualizing $visualizing --random_action False --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_${environment} --icm_lr $icm_lr --reward_weighting $reward_weighting --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --frames_before_reset $frames_before_reset 
                done
            done
        done
    done
    wait

    #python3 -m scripts.plot --environment $environment
done
