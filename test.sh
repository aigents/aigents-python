#python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=2 -ds=4 -mg=50000000 -mt=108000 -sm="exp(-d)" -tu=0 -cc=0.0 -dr=4 -r="human"
# mt suffix ds
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=2  -ds=$3 -mg=5000000 -mt=$1 -hf=1 -sm="exp(-d)" > sim_test5/exp$1_s2ds$3ss10lm2sc2cs2tu0er0mc0_$2_fire.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=3  -ds=$3 -mg=5000000 -mt=$1 -hf=1 -sm="exp(-d)" > sim_test5/exp$1_s3ds$3ss10lm2sc2cs2tu0er0mc0_$2_fire.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=41 -ds=$3 -mg=5000000 -mt=$1 -hf=1 -sm="exp(-d)" > sim_test5/exp$1_s41ds$3ss10lm2sc2cs2tu0er0mc0_$2_fire.txt

