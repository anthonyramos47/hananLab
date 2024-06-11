#Windows source /home/anthony/anaconda3/etc/profile.d/conda.sh
# Mac
source /Users/cisneras/opt/anaconda3/etc/profile.d/conda.sh

#Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_dir="/Users/cisneras/hanan/hananLab/QS_project/experiments/"

#Exp_name="Nice_result.pickle"
#Exp_name=$1
#!/bin/bash

for  Exp_name in Last_ex_2 Last_ex_3_inv_1 Last_ex_3_inv Last_ex_4_inv_1 Last_ex_4_inv Last_ex_5_inv_1 Last_ex_5_inv Last_ex_5
do
    Pickle_name=$Exp_name".pickle"

    conda activate hananJ

    python QS_project/triangulation.py $1

    conda deactivate
    conda activate geo


    file_path=$Exp_dir$Exp_name
    echo $Exp_dir$Exp_name

    python Quad_remesh/stress_remesher.py "$file_path" 0.5 0.01 0

    conda deactivate
    conda activate hananJ

    python QS_project/cross_field_sep.py $Exp_name


done


git add *
git commit -m "$Exp_name Upload for remeshing"
git push origin LastC