python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=$1 -mg=50000000 -mt=$2 -sm="exp(-d)" -tu=-1 -cc=0.1 -se="leftright" > sim_test4/lr_exp$2_s$1ss10lm2sc2cs2tu-1er0mc0cc01.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=$1 -mg=50000000 -mt=$2 -sm="exp(-d)" -tu=-1 -cc=0.1 -se="na"        > sim_test4/na_exp$2_s$1ss10lm2sc2cs2tu-1er0mc0cc01.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=$1 -mg=50000000 -mt=$2 -sm="exp(-d)" -tu=0  -cc=0.1 -se="leftright" > sim_test4/lr_exp$2_s$1ss10lm2sc2cs2tu0er0mc0cc01.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=$1 -mg=50000000 -mt=$2 -sm="exp(-d)" -tu=0  -cc=0.1 -se="na"        > sim_test4/na_exp$2_s$1ss10lm2sc2cs2tu0er0mc0cc01.txt

