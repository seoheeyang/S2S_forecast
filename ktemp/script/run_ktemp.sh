#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf22-env

#TRAIN_ITR=20000
TRAIN_UP=3

# 시작 시간 기록
start_time=$(date +%s)
echo "Start Time: $(date)"

for repeat_ in 300; do
   for ((ens=1; ens<=30; ens+=2)); do
      
      # ---------------------------------------------
      # [Job 1] (GPU 1)
      # ---------------------------------------------
      ens_odd=$ens
      py_file_odd="ktemp_run_${ens_odd}.py"
      
      echo ">>> Launching Ensemble $ens_odd on GPU 1"
      
      sed "s/ENSEMBLE/$ens_odd/g" inception.py > tmp1_${ens_odd}
      sed "s/TRAIN_ITERS/$repeat_/g" tmp1_${ens_odd} > tmp2_${ens_odd}
      sed "s/TRAIN_UPDATE/$TRAIN_UP/g" tmp2_${ens_odd} > $py_file_odd
      
      (
         export CUDA_VISIBLE_DEVICES=1
         python $py_file_odd
         rm -f tmp*_${ens_odd} $py_file_odd
      ) &
      pid_odd=$!

      # ---------------------------------------------
      # [Job 2] (GPU 0)
      # ---------------------------------------------
      ens_even=$((ens+1))
      if [ $ens_even -le 30 ]; then
          py_file_even="ktemp_run_${ens_even}.py"
          
          echo ">>> Launching Ensemble $ens_even on GPU 0"
          
          sed "s/ENSEMBLE/$ens_even/g" inception.py > tmp1_${ens_even}
          sed "s/TRAIN_ITERS/$repeat_/g" tmp1_${ens_even} > tmp2_${ens_even}
          sed "s/TRAIN_UPDATE/$TRAIN_UP/g" tmp2_${ens_even} > $py_file_even
          
          (
             export CUDA_VISIBLE_DEVICES=0
             python $py_file_even
             rm -f tmp*_${ens_even} $py_file_even
          ) &
          pid_even=$!
      fi

      wait $pid_odd $pid_even
      echo "Finished batch ($ens_odd, $ens_even). Proceeding..."
      
   done
done

end_time=$(date +%s)
total_diff=$((end_time - start_time))

echo "End Time: $(date)"
printf "Total Execution Time: %02d:%02d:%02d\n" $((total_diff/3600)) $((total_diff%3600/60)) $((total_diff%60))
